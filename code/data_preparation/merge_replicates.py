#!/usr/bin/env python3
"""Merge ENCODE replicates and process target/control pairs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import secrets
import string
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import poseigen_binmeths as bm
import poseigen_binmeths.bin_utils as bu
import poseigen_chisel.poseigen_chisel as chis
import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import betaprime

from download_encode import (
    EXPERIMENT_ACCESSION_PATTERN,
    FILE_ACCESSION_PATTERN,
    ensure_metadata_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODE_ROOT = PROJECT_ROOT / "data" / "ENCODE"
WIN_ROOT = ENCODE_ROOT / "win"
GENOME_ROOT = PROJECT_ROOT / "data" / "genomes"

SHRINK_WINDOW_SIZE = 0.01
SHRINK_STEP = 0.005
SHRINK_SAMPLE_SIZE = 100_000
HISTOGRAM_BINS = 1_000
BETA_PRIME_SAMPLE_SIZE = 100_000
BETA_PRIME_DRAWS = 1_000
BETA_PRIME_CHUNK_SIZE = 512
VST_DIAGNOSTIC_SAMPLE_SIZE = 100_000
VST_DIAGNOSTIC_SEED = 42
REPLICATE_QC_SAMPLE_SIZE = 100_000
REPLICATE_QC_SEED = 42
EPSILON = 1e-12


@dataclass
class PairGroup:
    target_dataset: str
    control_dataset: str
    target_files: list[str]
    control_files: list[str]
    pair_id: str = ""


@dataclass
class ProcessingPlan:
    replicate_groups: dict[str, list[dict[str, str]]]
    target_datasets: set[str]
    pair_groups: list[PairGroup]
    unresolved_controls: list[tuple[str, str]]


def parse_accessions(values: list[str], input_file: Path | None) -> list[str]:
    if values and input_file:
        raise ValueError("Provide accessions directly or with --input-file, not both.")
    if input_file:
        text = input_file.read_text(encoding="utf-8-sig")
        accessions = [part for part in re.split(r"[\s,]+", text) if part]
    else:
        accessions = values
    if not accessions:
        raise ValueError("Provide at least one ENCODE file accession.")

    invalid = [
        accession for accession in accessions
        if not FILE_ACCESSION_PATTERN.fullmatch(accession)
    ]
    if invalid:
        raise ValueError(
            "Invalid ENCODE file accession(s): " + ", ".join(invalid)
        )
    return list(dict.fromkeys(accessions))


def read_metadata(metadata_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ENCODE metadata file not found: {metadata_path}")
    ensure_metadata_columns(metadata_path)

    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter="\t")
        fieldnames = reader.fieldnames
        required = {
            "File name",
            "Dataset",
            "Target",
            "Controls",
            "Replicate ID",
            "Pair ID",
        }
        missing = required.difference(fieldnames or [])
        if missing:
            raise ValueError(
                f"{metadata_path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]

    seen: set[str] = set()
    for row in rows:
        accession = (row.get("File name") or "").strip()
        if not accession:
            continue
        if accession in seen:
            raise ValueError(f"Duplicate ENCODE file accession in metadata: {accession}")
        seen.add(accession)
    return fieldnames, rows


def _parse_controls(value: str | None) -> list[str]:
    if not value:
        return []
    accessions = EXPERIMENT_ACCESSION_PATTERN.findall(value)
    return list(dict.fromkeys(accessions))


def build_processing_plan(
    accessions: list[str], rows: list[dict[str, str]]
) -> ProcessingPlan:
    rows_by_accession = {
        (row.get("File name") or "").strip(): row
        for row in rows
        if (row.get("File name") or "").strip()
    }
    missing_accessions = [name for name in accessions if name not in rows_by_accession]
    if missing_accessions:
        raise ValueError(
            "Accession(s) not found in meta_info.txt: "
            + ", ".join(missing_accessions)
        )

    rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset = (row.get("Dataset") or "").strip()
        if dataset:
            rows_by_dataset[dataset].append(row)

    seed_datasets = list(
        dict.fromkeys(
            (rows_by_accession[accession].get("Dataset") or "").strip()
            for accession in accessions
        )
    )
    if any(not dataset for dataset in seed_datasets):
        raise ValueError("An input accession has no Dataset in meta_info.txt.")

    target_datasets = {
        dataset
        for dataset in seed_datasets
        if any((row.get("Target") or "").strip() for row in rows_by_dataset[dataset])
    }

    pair_target_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    unresolved_controls: list[tuple[str, str]] = []
    replicate_datasets = list(seed_datasets)

    for target_dataset in seed_datasets:
        target_rows = [
            row for row in rows_by_dataset[target_dataset]
            if (row.get("Target") or "").strip()
        ]
        for row in target_rows:
            controls = _parse_controls(row.get("Controls"))
            if not controls:
                continue
            missing_controls = [
                control for control in controls
                if not rows_by_dataset.get(control)
            ]
            if missing_controls:
                unresolved_controls.extend(
                    (target_dataset, control) for control in missing_controls
                )
                continue
            for control_dataset in controls:
                pair_target_rows[(target_dataset, control_dataset)].append(row)
                if control_dataset not in replicate_datasets:
                    replicate_datasets.append(control_dataset)

    replicate_groups = {
        dataset: rows_by_dataset[dataset]
        for dataset in replicate_datasets
    }
    pair_groups = [
        PairGroup(
            target_dataset=target_dataset,
            control_dataset=control_dataset,
            target_files=list(dict.fromkeys(
                (row.get("File name") or "").strip() for row in pair_rows
            )),
            control_files=[
                (row.get("File name") or "").strip()
                for row in rows_by_dataset[control_dataset]
            ],
        )
        for (target_dataset, control_dataset), pair_rows in pair_target_rows.items()
    ]

    return ProcessingPlan(
        replicate_groups=replicate_groups,
        target_datasets=target_datasets,
        pair_groups=pair_groups,
        unresolved_controls=list(dict.fromkeys(unresolved_controls)),
    )


def _split_codes(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _new_identifier(used: set[str]) -> str:
    while True:
        identifier = "".join(
            secrets.choice(string.ascii_uppercase) for _ in range(6)
        )
        if identifier not in used:
            used.add(identifier)
            return identifier


def assign_identifiers(
    rows: list[dict[str, str]],
    plan: ProcessingPlan,
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    used_codes: set[str] = set()
    for row in rows:
        used_codes.update(_split_codes(row.get("Replicate ID")))
        used_codes.update(_split_codes(row.get("Pair ID")))

    code_owners: dict[str, str] = {}
    replicate_existing: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        dataset = (row.get("Dataset") or "").strip()
        for identifier in _split_codes(row.get("Replicate ID")):
            if not re.fullmatch(r"[A-Z]{6}", identifier):
                raise ValueError(f"Invalid Replicate ID in metadata: {identifier!r}")
            owner = f"replicate:{dataset}"
            if identifier in code_owners and code_owners[identifier] != owner:
                raise ValueError(f"Identifier {identifier} is reused by multiple groups.")
            code_owners[identifier] = owner
            replicate_existing[dataset].add(identifier)

    replicate_ids: dict[str, str] = {}
    for dataset in plan.replicate_groups:
        existing = replicate_existing.get(dataset, set())
        if len(existing) > 1:
            raise ValueError(
                f"Dataset {dataset} already has multiple Replicate IDs: "
                + ", ".join(sorted(existing))
            )
        replicate_ids[dataset] = (
            next(iter(existing)) if existing else _new_identifier(used_codes)
        )
        code_owners[replicate_ids[dataset]] = f"replicate:{dataset}"

    pair_existing: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if not (row.get("Target") or "").strip():
            continue
        dataset = (row.get("Dataset") or "").strip()
        controls = _parse_controls(row.get("Controls"))
        identifiers = _split_codes(row.get("Pair ID"))
        if not identifiers:
            continue
        if len(controls) != len(identifiers):
            raise ValueError(
                f"Pair ID values for {row.get('File name')} do not align with "
                "its Controls values."
            )
        for control_dataset, identifier in zip(controls, identifiers):
            if not re.fullmatch(r"[A-Z]{6}", identifier):
                raise ValueError(f"Invalid Pair ID in metadata: {identifier!r}")
            key = (dataset, control_dataset)
            owner = f"pair:{dataset}:{control_dataset}"
            if identifier in code_owners and code_owners[identifier] != owner:
                raise ValueError(f"Identifier {identifier} is reused by multiple groups.")
            code_owners[identifier] = owner
            pair_existing[key].add(identifier)

    pair_ids: dict[tuple[str, str], str] = {}
    for pair in plan.pair_groups:
        key = (pair.target_dataset, pair.control_dataset)
        existing = pair_existing.get(key, set())
        if len(existing) > 1:
            raise ValueError(
                f"Target/control pair {key} already has multiple Pair IDs: "
                + ", ".join(sorted(existing))
            )
        pair_ids[key] = (
            next(iter(existing)) if existing else _new_identifier(used_codes)
        )

    for dataset, group_rows in plan.replicate_groups.items():
        for row in group_rows:
            row["Replicate ID"] = replicate_ids[dataset]

    for row in rows:
        target_dataset = (row.get("Dataset") or "").strip()
        if not (row.get("Target") or "").strip():
            row["Pair ID"] = ""
            continue
        if target_dataset not in plan.target_datasets:
            continue
        controls = _parse_controls(row.get("Controls"))
        if controls and all((target_dataset, control) in pair_ids for control in controls):
            row["Pair ID"] = ", ".join(
                pair_ids[(target_dataset, control)] for control in controls
            )
        else:
            row["Pair ID"] = ""

    for pair in plan.pair_groups:
        pair.pair_id = pair_ids[(pair.target_dataset, pair.control_dataset)]

    return replicate_ids, pair_ids


def write_metadata(
    metadata_path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(metadata_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_window_currents(plan: ProcessingPlan, win_root: Path) -> None:
    missing = [
        (row.get("File name") or "").strip()
        for group in plan.replicate_groups.values()
        for row in group
        if not (win_root / f"{(row.get('File name') or '').strip()}.p").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing window currents in {win_root}: "
            + ", ".join(dict.fromkeys(missing))
            + ". Run generate_encode_currents.py for these files first."
        )


def _read_current(path: Path) -> list[np.ndarray]:
    current = se.PickleLoad(str(path))
    if not isinstance(current, list) or not current:
        raise ValueError(f"Expected a non-empty chromosome current in {path}.")
    arrays = [np.asarray(chromosome, dtype=np.float32) for chromosome in current]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError(f"Current arrays in {path} must be two-dimensional.")
    return arrays


def _merge_current_paths(
    paths: list[Path],
    target_total_reads: float | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not paths:
        raise ValueError("Cannot merge an empty replicate collection.")

    means: list[np.ndarray] | None = None
    sum_squares: list[np.ndarray] | None = None
    count = 0
    for path in paths:
        current = _read_current(path)
        total_reads = sum(float(np.sum(chromosome, dtype=np.float64)) for chromosome in current)
        if not np.isfinite(total_reads) or total_reads <= 0:
            raise ValueError(f"Current {path} has a non-positive or invalid total.")
        if target_total_reads is not None:
            current = [
                chromosome * (target_total_reads / total_reads)
                for chromosome in current
            ]

        if means is None:
            means = [chromosome.copy() for chromosome in current]
            sum_squares = [np.zeros_like(chromosome) for chromosome in current]
        elif len(current) != len(means):
            raise ValueError(f"Chromosome counts do not match for {path}.")

        count += 1
        for index, chromosome in enumerate(current):
            if chromosome.shape != means[index].shape:
                raise ValueError(
                    f"Chromosome current shape mismatch for {path} at index {index}: "
                    f"{chromosome.shape} vs {means[index].shape}."
                )
            delta = chromosome - means[index]
            means[index] += delta / count
            sum_squares[index] += delta * (chromosome - means[index])
        del current

    if means is None or sum_squares is None:
        raise ValueError("No replicate currents were loaded.")
    if count == 1:
        standard_deviations = [
            np.full_like(mean, np.nan, dtype=np.float32) for mean in means
        ]
    else:
        standard_deviations = [
            np.sqrt(np.maximum(total / (count - 1), 0)).astype(np.float32)
            for total in sum_squares
        ]
    return means, standard_deviations


def _average_total_reads(paths: list[Path]) -> float:
    totals = []
    for path in paths:
        current = _read_current(path)
        total = sum(float(np.sum(chromosome, dtype=np.float64)) for chromosome in current)
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Control current {path} has an invalid total.")
        totals.append(total)
        del current
    if not totals:
        raise ValueError("No control currents are available for read scaling.")
    return float(np.mean(totals))


def _pickle_pair(
    directory: Path, identifier: str, suffix: str
) -> Path:
    return directory / f"{identifier}_{suffix}.p"


def _save_mean_std(
    means: list[np.ndarray],
    stds: list[np.ndarray],
    mean_path: Path,
    std_path: Path,
    rewrite: bool = False,
) -> None:
    if rewrite or not mean_path.is_file():
        se.PickleDump(means, str(mean_path))
    if rewrite or not std_path.is_file():
        se.PickleDump(stds, str(std_path))


def merge_replicate_groups(
    plan: ProcessingPlan,
    replicate_ids: dict[str, str],
    win_root: Path,
    merged_root: Path,
    rewrite: bool,
) -> None:
    for dataset, rows in plan.replicate_groups.items():
        identifier = replicate_ids[dataset]
        mean_path = _pickle_pair(merged_root, identifier, "mean")
        std_path = _pickle_pair(merged_root, identifier, "std")
        if not rewrite and mean_path.is_file() and std_path.is_file():
            print(f"{identifier}: using cached replicate mean/std.")
            continue
        paths = [
            win_root / f"{(row.get('File name') or '').strip()}.p"
            for row in rows
        ]
        means, stds = _merge_current_paths(paths)
        _save_mean_std(means, stds, mean_path, std_path, rewrite)
        if len(paths) == 1:
            print(
                f"Warning: {identifier} contains one replicate; its sample std "
                "is recorded as NaN.",
                file=sys.stderr,
            )
        print(f"{identifier}: merged {len(paths)} replicate current(s).")


def merge_target_pair(
    pair: PairGroup,
    win_root: Path,
    merged_root: Path,
    rewrite: bool,
) -> tuple[Path, Path]:
    mean_path = _pickle_pair(merged_root, pair.pair_id, "mean")
    std_path = _pickle_pair(merged_root, pair.pair_id, "std")
    if not rewrite and mean_path.is_file() and std_path.is_file():
        return mean_path, std_path

    control_paths = [
        win_root / f"{accession}.p" for accession in pair.control_files
    ]
    target_paths = [
        win_root / f"{accession}.p" for accession in pair.target_files
    ]
    control_mean_reads = _average_total_reads(control_paths)
    means, stds = _merge_current_paths(
        target_paths, target_total_reads=control_mean_reads
    )
    _save_mean_std(means, stds, mean_path, std_path, rewrite)
    print(
        f"{pair.pair_id}: merged {len(target_paths)} target replicate(s), "
        f"scaled to average control reads ({control_mean_reads:.3g})."
    )
    return mean_path, std_path


def _sample_unblacklisted_positions(
    blacklist: list[np.ndarray],
    sample_size: int,
    seed: int,
) -> list[np.ndarray]:
    eligible_counts = [
        int(np.count_nonzero(np.asarray(chromosome).reshape(-1) > 0))
        for chromosome in blacklist
    ]
    total_eligible = sum(eligible_counts)
    if total_eligible == 0:
        raise ValueError("No unblacklisted bins are available for replicate QC.")

    sample_count = min(sample_size, total_eligible)
    rng = np.random.default_rng(seed)
    if total_eligible <= sample_size:
        ranks = np.arange(total_eligible, dtype=np.int64)
    else:
        ranks = np.empty(sample_count, dtype=np.int64)
        selected_ranks: set[int] = set()
        # Floyd sampling avoids allocating a permutation of all genome bins.
        for sample_index, upper_bound in enumerate(
            range(total_eligible - sample_count, total_eligible)
        ):
            draw = int(rng.integers(upper_bound + 1))
            rank = upper_bound if draw in selected_ranks else draw
            selected_ranks.add(rank)
            ranks[sample_index] = rank
        ranks.sort()
    offsets = np.concatenate(([0], np.cumsum(eligible_counts)))
    chromosome_indices = np.searchsorted(offsets[1:], ranks, side="right")
    positions_by_chromosome: list[np.ndarray] = []
    for index, chromosome in enumerate(blacklist):
        local_ranks = ranks[chromosome_indices == index] - offsets[index]
        if not len(local_ranks):
            positions_by_chromosome.append(np.empty(0, dtype=np.int64))
            continue
        eligible_positions = np.flatnonzero(
            np.asarray(chromosome).reshape(-1) > 0
        )
        positions_by_chromosome.append(eligible_positions[local_ranks])
    return positions_by_chromosome


def _sample_target_replicates(
    pair: PairGroup,
    win_root: Path,
    blacklist: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    positions_by_chromosome = _sample_unblacklisted_positions(
        blacklist,
        REPLICATE_QC_SAMPLE_SIZE,
        REPLICATE_QC_SEED,
    )
    sample_count = sum(len(positions) for positions in positions_by_chromosome)
    control_paths = [win_root / f"{accession}.p" for accession in pair.control_files]
    control_total_reads = _average_total_reads(control_paths)
    sampled_values = np.empty((len(pair.target_files), sample_count), dtype=np.float64)

    for replicate_index, accession in enumerate(pair.target_files):
        current = _read_current(win_root / f"{accession}.p")
        if len(current) != len(blacklist):
            raise ValueError(
                f"{pair.pair_id}: replicate {accession} and blacklist chromosome "
                "counts differ."
            )
        if any(
            chromosome.shape[0] != np.asarray(blacklist[index]).size
            or chromosome.shape[1] != 1
            for index, chromosome in enumerate(current)
        ):
            raise ValueError(
                f"{pair.pair_id}: replicate {accession} does not align with "
                "blacklist bins."
            )

        total_reads = sum(
            float(np.sum(chromosome, dtype=np.float64))
            for chromosome in current
        )
        if not np.isfinite(total_reads) or total_reads <= 0:
            raise ValueError(
                f"{pair.pair_id}: replicate {accession} has an invalid total."
            )
        scale = control_total_reads / total_reads
        offset = 0
        for chromosome, positions in zip(current, positions_by_chromosome):
            if not len(positions):
                continue
            values = chromosome[positions, 0].astype(np.float64) * scale
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(
                    f"{pair.pair_id}: replicate {accession} has invalid signal "
                    "values."
                )
            sampled_values[
                replicate_index,
                offset : offset + len(positions),
            ] = values
            offset += len(positions)
    return sampled_values, control_total_reads


def _replicate_pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float | int | None]:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if not len(left):
        raise ValueError("No finite sampled values are available for replicate QC.")

    left_log = np.log1p(left)
    right_log = np.log1p(right)
    left_centered = left_log - np.mean(left_log)
    right_centered = right_log - np.mean(right_log)
    denominator = np.sqrt(
        np.sum(left_centered**2) * np.sum(right_centered**2)
    )
    correlation = (
        float(np.sum(left_centered * right_centered) / denominator)
        if denominator > 0
        else None
    )
    return {
        "sample_count": int(len(left)),
        "pearson_log1p": correlation,
        "zero_positive_disagreement_fraction": float(
            np.mean((left == 0) != (right == 0))
        ),
        "both_positive_fraction": float(np.mean((left > 0) & (right > 0))),
    }


def _save_target_replicate_concordance(
    pair: PairGroup,
    win_root: Path,
    qc_root: Path,
    blacklist: list[np.ndarray],
    rewrite: bool,
) -> None:
    qc_root.mkdir(parents=True, exist_ok=True)
    figures_dir = qc_root / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cache_path = qc_root / f"{pair.pair_id}_target_replicate_concordance.npz"
    if rewrite or not cache_path.is_file():
        sampled_values, control_total_reads = _sample_target_replicates(
            pair,
            win_root,
            blacklist,
        )
        np.savez_compressed(
            cache_path,
            values=sampled_values,
            files=np.asarray(pair.target_files, dtype=str),
            control_total_reads=np.asarray(control_total_reads),
        )

    with np.load(cache_path) as cache:
        replicate_files = [str(value) for value in cache["files"]]
        if replicate_files != pair.target_files:
            raise ValueError(
                f"{pair.pair_id}: replicate QC cache does not match target files; "
                "rerun merge_replicates.py with --rewrite."
            )
        sampled_values = np.asarray(cache["values"], dtype=np.float64)
        control_total_reads = float(cache["control_total_reads"])
    if sampled_values.ndim != 2 or sampled_values.shape[0] != len(replicate_files):
        raise ValueError(f"{pair.pair_id}: invalid replicate QC cache shape.")

    pairwise: list[dict[str, Any]] = []
    pair_indices = list(combinations(range(len(replicate_files)), 2))
    for left_index, right_index in pair_indices:
        metrics = _replicate_pair_metrics(
            sampled_values[left_index],
            sampled_values[right_index],
        )
        pairwise.append(
            {
                "replicate_a": replicate_files[left_index],
                "replicate_b": replicate_files[right_index],
                **metrics,
            }
        )

    columns = min(3, max(1, len(pair_indices)))
    rows = (len(pair_indices) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4.5 * rows),
        squeeze=False,
    )
    flattened_axes = axes.reshape(-1)
    maximum_log_value = float(np.log1p(np.max(sampled_values)))
    maximum_log_value = max(maximum_log_value, 1.0)
    for axis, (left_index, right_index), metrics in zip(
        flattened_axes,
        pair_indices,
        pairwise,
    ):
        left_log = np.log1p(sampled_values[left_index])
        right_log = np.log1p(sampled_values[right_index])
        axis.hexbin(
            left_log,
            right_log,
            gridsize=55,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        axis.plot(
            [0, maximum_log_value],
            [0, maximum_log_value],
            color="white",
            linestyle="--",
            linewidth=1,
        )
        correlation = metrics["pearson_log1p"]
        correlation_label = (
            f"{correlation:.3f}" if correlation is not None else "n/a"
        )
        disagreement = metrics["zero_positive_disagreement_fraction"]
        axis.set(
            title=(
                f"{replicate_files[left_index]} vs "
                f"{replicate_files[right_index]}\n"
                f"log1p r={correlation_label}; zero/positive disagreement="
                f"{disagreement:.1%}"
            ),
            xlabel=f"log1p normalized signal ({replicate_files[left_index]})",
            ylabel=f"log1p normalized signal ({replicate_files[right_index]})",
            xlim=(0, maximum_log_value),
            ylim=(0, maximum_log_value),
        )
        axis.grid(alpha=0.15)
    for axis in flattened_axes[len(pair_indices) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{pair.pair_id}: target replicate concordance "
        f"({sampled_values.shape[1]:,} sampled unblacklisted bins; "
        f"scaled to {control_total_reads:.3g} control reads)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        figures_dir / f"{pair.pair_id}_target_replicate_concordance.png",
        bbox_inches="tight",
    )
    plt.close(figure)

    report = {
        "pair_id": pair.pair_id,
        "normalization": (
            "Each target replicate was scaled to the mean total reads across "
            "the paired control replicates."
        ),
        "sampled_unblacklisted_bins": int(sampled_values.shape[1]),
        "sampling_seed": REPLICATE_QC_SEED,
        "sampled_without_replacement": True,
        "control_mean_total_reads": control_total_reads,
        "replicates": replicate_files,
        "pairwise": pairwise,
    }
    with (qc_root / f"{pair.pair_id}_target_replicate_concordance.json").open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")


def _fit_std_predictor(
    means: list[np.ndarray],
    stds: list[np.ndarray],
    rng: np.random.Generator,
):
    mean_values = np.vstack(means).reshape(-1)
    std_values = np.vstack(stds).reshape(-1)
    valid = (
        (mean_values > 0)
        & (std_values > 0)
        & np.isfinite(mean_values)
        & np.isfinite(std_values)
    )
    mean_values = mean_values[valid]
    std_values = std_values[valid]
    if len(mean_values) < 2:
        raise ValueError(
            "Not enough positive finite mean/std observations to shrink standard "
            "deviation; at least two replicates with variable signal are required."
        )

    assignments = bu.HistBinning(
        mean_values, bins=min(HISTOGRAM_BINS, len(mean_values))
    )
    weights = np.asarray(bu.BinWeights(assignments), dtype=np.float64).reshape(-1)
    weight_sum = weights.sum()
    if (
        len(weights) != len(mean_values)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isfinite(weight_sum)
        or weight_sum <= 0
    ):
        raise ValueError("Could not derive valid mean-balanced sampling weights.")
    weights /= weight_sum

    sample_count = SHRINK_SAMPLE_SIZE
    sample_indices = rng.choice(
        len(mean_values), size=sample_count, replace=True, p=weights
    )
    log_means = np.log(mean_values[sample_indices] + EPSILON)
    log_stds = np.log(std_values[sample_indices])
    centers, rolling_std = se.RollingVal(
        log_means,
        log_stds,
        mode=[np.median, {}],
        win_size=SHRINK_WINDOW_SIZE,
        step=SHRINK_STEP,
        extended=True,
    )
    valid_curve = np.isfinite(centers) & np.isfinite(rolling_std)
    centers = np.asarray(centers)[valid_curve]
    rolling_std = np.asarray(rolling_std)[valid_curve]
    if len(centers) < 2:
        raise ValueError("Not enough non-empty windows to fit a std predictor.")

    order = np.argsort(centers)
    centers = centers[order]
    rolling_std = rolling_std[order]
    centers, unique_indices = np.unique(centers, return_index=True)
    rolling_std = rolling_std[unique_indices]
    if len(centers) < 2:
        raise ValueError("Std predictor requires at least two distinct mean windows.")

    smoothed_std = gaussian_filter1d(
        rolling_std, sigma=(SHRINK_WINDOW_SIZE / SHRINK_STEP) / 2
    )
    return interp1d(
        centers,
        smoothed_std,
        kind="linear",
        bounds_error=False,
        fill_value=(smoothed_std[0], smoothed_std[-1]),
    )


def shrink_standard_deviation(
    identifier: str,
    mean_path: Path,
    std_path: Path,
    shrunk_root: Path,
    rng: np.random.Generator,
    rewrite: bool,
) -> list[np.ndarray]:
    predictor_path = _pickle_pair(shrunk_root, identifier, "std_predictor")
    shrunk_path = _pickle_pair(shrunk_root, identifier, "std_shrunk")
    if not rewrite and shrunk_path.is_file():
        return se.PickleLoad(str(shrunk_path))

    means = se.PickleLoad(str(mean_path))
    stds = se.PickleLoad(str(std_path))
    if not rewrite and predictor_path.is_file():
        predictor = se.PickleLoad(str(predictor_path))
    else:
        predictor = _fit_std_predictor(means, stds, rng)
        se.PickleDump(predictor, str(predictor_path))

    shrunk = [
        np.exp(
            predictor(np.log(np.asarray(mean).reshape(-1) + EPSILON))
        ).reshape(-1, 1)
        for mean in means
    ]
    if any(not np.all(np.isfinite(chromosome)) for chromosome in shrunk):
        raise ValueError(f"{identifier}: std shrinkage produced non-finite values.")
    se.PickleDump(shrunk, str(shrunk_path))
    return shrunk


def _gamma_parameters(
    means: list[np.ndarray], stds: list[np.ndarray]
) -> list[np.ndarray]:
    if len(means) != len(stds):
        raise ValueError("Mean and std chromosome counts do not match.")
    parameters = []
    for mean, std in zip(means, stds):
        mean_values = np.asarray(mean).reshape(-1)
        std_values = np.asarray(std).reshape(-1)
        if mean_values.shape != std_values.shape:
            raise ValueError("Mean and std shapes do not match.")
        alpha, theta = mex.Gamma_params(mean_values, std_values)
        parameters.append(np.column_stack((alpha, theta)))
    return parameters


def calculate_beta_prime(
    pair: PairGroup,
    merged_root: Path,
    shrunk_root: Path,
    betaprime_root: Path,
    control_replicate_id: str,
    rewrite: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    mode_path = _pickle_pair(betaprime_root, pair.pair_id, "gbp_mode")
    parameters_path = _pickle_pair(betaprime_root, pair.pair_id, "gbp_exp")
    if not rewrite and mode_path.is_file() and parameters_path.is_file():
        return (
            se.PickleLoad(str(mode_path)),
            se.PickleLoad(str(parameters_path)),
        )

    target_mean = se.PickleLoad(str(_pickle_pair(merged_root, pair.pair_id, "mean")))
    control_mean = se.PickleLoad(
        str(_pickle_pair(merged_root, control_replicate_id, "mean"))
    )
    target_std = se.PickleLoad(
        str(_pickle_pair(shrunk_root, pair.pair_id, "std_shrunk"))
    )
    control_std = se.PickleLoad(
        str(_pickle_pair(shrunk_root, control_replicate_id, "std_shrunk"))
    )

    target_gamma = _gamma_parameters(target_mean, target_std)
    control_gamma = _gamma_parameters(control_mean, control_std)
    if not (
        len(target_gamma) == len(control_gamma)
        and all(
            target.shape == control.shape
            for target, control in zip(target_gamma, control_gamma)
        )
    ):
        raise ValueError(f"{pair.pair_id}: target/control current shapes differ.")

    modes = [
        mex.GB2_Mode(
            target[:, 0],
            target[:, 1],
            control[:, 0],
            control[:, 1],
        ).reshape(-1, 1)
        for target, control in zip(target_gamma, control_gamma)
    ]
    parameters = [
        np.hstack((target, control))
        for target, control in zip(target_gamma, control_gamma)
    ]
    if rewrite or not mode_path.is_file():
        se.PickleDump(modes, str(mode_path))
    if rewrite or not parameters_path.is_file():
        se.PickleDump(parameters, str(parameters_path))
    print(f"{pair.pair_id}: calculated generalized beta-prime parameters.")
    return modes, parameters


def _estimate_beta_prime_mad(
    parameters: np.ndarray,
    rng: np.random.Generator,
    sample_size: int = BETA_PRIME_SAMPLE_SIZE,
    chunk_size: int = BETA_PRIME_CHUNK_SIZE,
) -> np.ndarray:
    if parameters.ndim != 2 or parameters.shape[1] != 4:
        raise ValueError("Beta-prime parameters must have shape (observations, 4).")
    if sample_size <= 0 or chunk_size <= 0:
        raise ValueError("Beta-prime sampling sizes must be positive.")

    estimates = np.empty(len(parameters), dtype=np.float64)
    for start in range(0, len(parameters), chunk_size):
        stop = min(start + chunk_size, len(parameters))
        values = parameters[start:stop]
        draws = betaprime.rvs(
            values[:, 0, None],
            values[:, 2, None],
            size=(len(values), sample_size),
            random_state=rng,
        )
        draws *= (values[:, 1] / values[:, 3])[:, None]
        medians = np.median(draws, axis=1)
        deviations = np.median(
            np.abs(draws - medians[:, None]), axis=1
        ) * 1.4826
        if not np.all(np.isfinite(deviations)):
            raise ValueError(
                f"Beta-prime MAD sampling produced non-finite values in rows "
                f"{start}:{stop}."
            )
        estimates[start:stop] = deviations
        if stop == len(parameters) or stop % (chunk_size * 10) == 0:
            print(f"Beta-prime MAD: {stop:,}/{len(parameters):,} observations")
    return estimates


def _save_vst_figures(
    pair: PairGroup,
    vst_root: Path,
    transform: Any,
    win_root: Path,
    control_replicate_id: str,
    merged_root: Path,
    betaprime_root: Path,
    blacklist: list[np.ndarray],
    rewrite: bool,
) -> None:
    centers = np.asarray(transform.x, dtype=np.float64).reshape(-1)
    vst_values = np.asarray(transform.y, dtype=np.float64).reshape(-1)
    if (
        len(centers) < 2
        or len(centers) != len(vst_values)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(vst_values))
        or np.any(np.diff(centers) <= 0)
        or np.any(np.diff(vst_values) <= 0)
    ):
        raise ValueError(f"{pair.pair_id}: cannot plot an invalid VST transform.")

    figures_dir = vst_root / "figs"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(centers, vst_values, color="tab:blue")
    axis.set(
        title=f"{pair.pair_id}: variance-stabilizing transform",
        xlabel="Raw beta-prime mode",
        ylabel="VST value",
    )
    axis.grid(alpha=0.25)
    figure.savefig(
        figures_dir / f"{pair.pair_id}_vst_transform.png",
        bbox_inches="tight",
    )
    plt.close(figure)

    mode_intervals = np.diff(centers)
    vst_intervals = np.diff(vst_values)
    implied_scale = mode_intervals / vst_intervals
    interval_centers = (centers[:-1] + centers[1:]) / 2
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(interval_centers, implied_scale, color="tab:orange")
    axis.set(
        title=f"{pair.pair_id}: scale implied by the VST",
        xlabel="Raw beta-prime mode",
        ylabel="Raw-mode units per VST unit",
    )
    axis.grid(alpha=0.25)
    figure.savefig(
        figures_dir / f"{pair.pair_id}_vst_scale.png",
        bbox_inches="tight",
    )
    plt.close(figure)

    observed_cache = figures_dir / f"{pair.pair_id}_vst_observed_spread.npz"
    if rewrite or not observed_cache.is_file():
        observed = _observed_replicate_spreads(
            pair,
            transform,
            win_root,
            control_replicate_id,
            merged_root,
            betaprime_root,
            blacklist,
        )
        np.savez_compressed(observed_cache, **observed)

    with np.load(observed_cache) as observed:
        raw_centers = observed["raw_centers"]
        raw_spreads = observed["raw_spreads"]
        transformed_centers = observed["transformed_centers"]
        transformed_spreads = observed["transformed_spreads"]
        site_count = int(observed["site_count"])
        outside_fraction = float(observed["outside_fraction"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, x_values, y_values, title, xlabel in (
        (
            axes[0],
            raw_centers,
            raw_spreads,
            "Observed replicate ratios",
            "Median raw target/control ratio",
        ),
        (
            axes[1],
            transformed_centers,
            transformed_spreads,
            "Observed ratios after VST",
            "Median transformed ratio",
        ),
    ):
        axis.scatter(
            x_values,
            y_values,
            s=4,
            alpha=0.12,
            color="tab:blue",
            rasterized=True,
        )
        order = np.argsort(x_values)
        groups = np.array_split(order, min(40, len(order)))
        trend_x = np.asarray([np.median(x_values[group]) for group in groups])
        trend_y = np.asarray([np.median(y_values[group]) for group in groups])
        axis.plot(
            trend_x,
            trend_y,
            color="tab:red",
            linewidth=1.8,
            label="Median across equal-count bins",
        )
        axis.set(title=title, xlabel=xlabel, ylabel="Observed robust spread (1.4826 x MAD)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Unit-spread reference",
    )
    axes[1].legend()
    figure.suptitle(
        f"{pair.pair_id}: observed spread before/after VST "
        f"({site_count:,} sites; {outside_fraction:.1%} ratios outside fit range)"
    )
    figure.tight_layout()
    figure.savefig(figures_dir / f"{pair.pair_id}_vst_spread.png", bbox_inches="tight")
    plt.close(figure)


def _observed_replicate_spreads(
    pair: PairGroup,
    transform: Any,
    win_root: Path,
    control_replicate_id: str,
    merged_root: Path,
    betaprime_root: Path,
    blacklist: list[np.ndarray],
) -> dict[str, np.ndarray]:
    control_mean = se.PickleLoad(
        str(_pickle_pair(merged_root, control_replicate_id, "mean"))
    )
    target_mode = se.PickleLoad(
        str(_pickle_pair(betaprime_root, pair.pair_id, "gbp_mode"))
    )
    if not (
        len(control_mean) == len(target_mode) == len(blacklist)
    ):
        raise ValueError(f"{pair.pair_id}: current and blacklist chromosome counts differ.")

    control_values = np.vstack(control_mean).reshape(-1)
    mode_values = np.vstack(target_mode).reshape(-1)
    blacklist_values = np.vstack(blacklist).reshape(-1)
    if not (
        len(control_values) == len(mode_values) == len(blacklist_values)
    ):
        raise ValueError(f"{pair.pair_id}: flattened current lengths do not align.")

    valid_control = np.isfinite(control_values) & (control_values > 0)
    valid_mode = np.isfinite(mode_values) & (mode_values > 0)
    if not valid_control.any() or not valid_mode.any():
        raise ValueError(f"{pair.pair_id}: no positive control or target mode values.")
    control_min, control_max = np.percentile(
        control_values[valid_control], [5, 100]
    )
    mode_min, mode_max = np.percentile(mode_values[valid_mode], [1, 100])
    valid = (
        valid_control
        & valid_mode
        & (control_values >= control_min)
        & (control_values <= control_max)
        & (mode_values >= mode_min)
        & (mode_values <= mode_max)
        & (blacklist_values > 0)
    )
    valid_indices = np.flatnonzero(valid)
    modes = mode_values[valid]
    if len(modes) < 2:
        raise ValueError(f"{pair.pair_id}: too few unblacklisted values for VST diagnostics.")

    assignments = bu.HistBinning(
        modes, bins=min(HISTOGRAM_BINS, len(modes))
    )
    sample_count = min(VST_DIAGNOSTIC_SAMPLE_SIZE, len(modes))
    global_random_state = np.random.get_state()
    try:
        sampled_mode_indices = bm.Harpoon(
            assignments,
            select=sample_count,
            random_state=VST_DIAGNOSTIC_SEED,
        )
    finally:
        np.random.set_state(global_random_state)
    sampled_mode_indices = np.unique(
        np.concatenate(
            (
                np.asarray(sampled_mode_indices, dtype=np.int64),
                [int(np.argmin(modes)), int(np.argmax(modes))],
            )
        )
    )
    sampled_global_indices = np.sort(valid_indices[sampled_mode_indices])
    chromosome_lengths = [np.asarray(chromosome).size for chromosome in target_mode]
    chromosome_offsets = np.concatenate(([0], np.cumsum(chromosome_lengths)))
    chromosome_indices = np.searchsorted(
        chromosome_offsets[1:], sampled_global_indices, side="right"
    )
    local_indices = sampled_global_indices - chromosome_offsets[chromosome_indices]

    target_currents = [
        _read_current(win_root / f"{accession}.p")
        for accession in pair.target_files
    ]
    control_currents = [
        _read_current(win_root / f"{accession}.p")
        for accession in pair.control_files
    ]
    if not target_currents or not control_currents:
        raise ValueError(f"{pair.pair_id}: observed spread requires target and control files.")
    current_sets = target_currents + control_currents
    if any(len(current) != len(chromosome_lengths) for current in current_sets):
        raise ValueError(f"{pair.pair_id}: replicate current chromosome counts differ.")

    def total_reads(current: list[np.ndarray]) -> float:
        return sum(float(np.sum(chromosome, dtype=np.float64)) for chromosome in current)

    target_totals = np.asarray([total_reads(current) for current in target_currents])
    control_total = float(np.mean([total_reads(current) for current in control_currents]))
    if (
        not np.isfinite(control_total)
        or control_total <= 0
        or not np.all(np.isfinite(target_totals))
        or np.any(target_totals <= 0)
    ):
        raise ValueError(f"{pair.pair_id}: invalid replicate totals for observed spread.")
    target_scales = control_total / target_totals

    raw_centers: list[np.ndarray] = []
    raw_spreads: list[np.ndarray] = []
    transformed_centers: list[np.ndarray] = []
    transformed_spreads: list[np.ndarray] = []
    outside_count = 0
    ratio_count = 0
    transform_min = float(np.asarray(transform.x)[0])
    transform_max = float(np.asarray(transform.x)[-1])

    for chromosome_index, chromosome_length in enumerate(chromosome_lengths):
        selected = chromosome_indices == chromosome_index
        if not selected.any():
            continue
        positions = local_indices[selected]
        if any(
            current[chromosome_index].shape[0] != chromosome_length
            or current[chromosome_index].shape[1] != 1
            for current in current_sets
        ):
            raise ValueError(
                f"{pair.pair_id}: replicate currents do not align at chromosome "
                f"index {chromosome_index}."
            )

        target_values = np.column_stack(
            [
                current[chromosome_index][positions, 0] * scale
                for current, scale in zip(target_currents, target_scales)
            ]
        )
        control_values = np.column_stack(
            [current[chromosome_index][positions, 0] for current in control_currents]
        )
        valid_pairs = (
            np.isfinite(target_values[:, :, None])
            & (target_values[:, :, None] >= 0)
            & np.isfinite(control_values[:, None, :])
            & (control_values[:, None, :] > 0)
        )
        ratios = np.divide(
            target_values[:, :, None],
            control_values[:, None, :],
            out=np.full(valid_pairs.shape, np.nan, dtype=np.float64),
            where=valid_pairs,
        )
        pair_counts = np.sum(valid_pairs, axis=(1, 2))
        usable_sites = pair_counts >= 3
        if not usable_sites.any():
            continue

        observed_ratios = ratios[usable_sites]
        raw_median = np.nanmedian(observed_ratios, axis=(1, 2))
        raw_mad = 1.4826 * np.nanmedian(
            np.abs(observed_ratios - raw_median[:, None, None]),
            axis=(1, 2),
        )
        transformed_ratios = np.full_like(observed_ratios, np.nan)
        valid_ratios = np.isfinite(observed_ratios)
        transformed_ratios[valid_ratios] = transform(observed_ratios[valid_ratios])
        transformed_median = np.nanmedian(transformed_ratios, axis=(1, 2))
        transformed_mad = 1.4826 * np.nanmedian(
            np.abs(transformed_ratios - transformed_median[:, None, None]),
            axis=(1, 2),
        )
        finite = (
            np.isfinite(raw_median)
            & np.isfinite(raw_mad)
            & np.isfinite(transformed_median)
            & np.isfinite(transformed_mad)
        )
        raw_centers.append(raw_median[finite])
        raw_spreads.append(raw_mad[finite])
        transformed_centers.append(transformed_median[finite])
        transformed_spreads.append(transformed_mad[finite])
        outside_count += int(
            np.sum(
                valid_ratios
                & ((observed_ratios < transform_min) | (observed_ratios > transform_max))
            )
        )
        ratio_count += int(np.sum(valid_ratios))

    if not raw_centers or ratio_count == 0:
        raise ValueError(f"{pair.pair_id}: no usable observed replicate ratios for VST.")
    return {
        "raw_centers": np.concatenate(raw_centers),
        "raw_spreads": np.concatenate(raw_spreads),
        "transformed_centers": np.concatenate(transformed_centers),
        "transformed_spreads": np.concatenate(transformed_spreads),
        "site_count": np.asarray(sum(map(len, raw_centers)), dtype=np.int64),
        "outside_fraction": np.asarray(outside_count / ratio_count),
    }


def load_autosome_blacklist(genome: str = "hg38") -> list[np.ndarray]:
    genome_root = GENOME_ROOT / genome
    basic_ids_path = genome_root / f"{genome}_basic_id_sizes.p"
    blacklist_path = genome_root / f"{genome}_basic_BLC_LR50.p"
    missing = [
        path for path in (basic_ids_path, blacklist_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {genome} blacklist resource(s): "
            + ", ".join(str(path) for path in missing)
        )
    basic_ids, _ = se.PickleLoad(str(genome_root / f"{genome}_basic_id_sizes"))
    if len(basic_ids) < 4:
        raise ValueError(
            f"{genome} basic chromosome list is too short to select autosomes."
        )
    autosome_ids = basic_ids[:-3]
    blacklist = se.PickleLoad(str(genome_root / f"{genome}_basic_BLC_LR50"))
    autosome_blacklist = chis.CurrentSelect(blacklist, basic_ids, autosome_ids)
    if len(autosome_blacklist) != len(autosome_ids):
        raise ValueError(
            f"Could not align {genome} blacklist currents to autosomes."
        )
    return autosome_blacklist


def calculate_vst(
    pair: PairGroup,
    control_replicate_id: str,
    win_root: Path,
    merged_root: Path,
    betaprime_root: Path,
    vst_root: Path,
    blacklist: list[np.ndarray],
    rng: np.random.Generator,
    rewrite: bool,
) -> None:
    transform_path = _pickle_pair(vst_root, pair.pair_id, "vst_transform")
    if transform_path.is_file() and not rewrite:
        transform = se.PickleLoad(str(transform_path))
        _save_vst_figures(
            pair,
            vst_root,
            transform,
            win_root,
            control_replicate_id,
            merged_root,
            betaprime_root,
            blacklist,
            rewrite,
        )
        print(f"{pair.pair_id}: using cached VST transform.")
        return

    control_mean = se.PickleLoad(
        str(_pickle_pair(merged_root, control_replicate_id, "mean"))
    )
    target_mode = se.PickleLoad(
        str(_pickle_pair(betaprime_root, pair.pair_id, "gbp_mode"))
    )
    target_parameters = se.PickleLoad(
        str(_pickle_pair(betaprime_root, pair.pair_id, "gbp_exp"))
    )
    if not (
        len(control_mean) == len(target_mode) == len(target_parameters) == len(blacklist)
    ):
        raise ValueError(f"{pair.pair_id}: current and blacklist chromosome counts differ.")

    control_values = np.vstack(control_mean).reshape(-1)
    mode_values = np.vstack(target_mode).reshape(-1)
    parameter_values = np.vstack(target_parameters)
    blacklist_values = np.vstack(blacklist).reshape(-1)
    if not (
        len(control_values) == len(mode_values)
        and len(mode_values) == len(parameter_values)
        and len(parameter_values) == len(blacklist_values)
    ):
        raise ValueError(f"{pair.pair_id}: flattened current lengths do not align.")

    valid_control = np.isfinite(control_values) & (control_values > 0)
    valid_mode = np.isfinite(mode_values) & (mode_values > 0)
    if not valid_control.any() or not valid_mode.any():
        raise ValueError(f"{pair.pair_id}: no positive control or target mode values.")
    control_min, control_max = np.percentile(
        control_values[valid_control], [5, 100]
    )
    mode_min, mode_max = np.percentile(mode_values[valid_mode], [1, 100])
    valid = (
        valid_control
        & valid_mode
        & (control_values >= control_min)
        & (control_values <= control_max)
        & (mode_values >= mode_min)
        & (mode_values <= mode_max)
        & (blacklist_values > 0)
    )
    modes = mode_values[valid]
    parameters = parameter_values[valid]
    if len(modes) < 2:
        raise ValueError(f"{pair.pair_id}: too few unblacklisted values for VST.")

    histogram_assignments = bu.HistBinning(
        modes, bins=min(HISTOGRAM_BINS, len(modes))
    )
    weights = np.asarray(
        bu.BinWeights(histogram_assignments), dtype=np.float64
    ).reshape(-1)
    weight_sum = weights.sum()
    if (
        len(weights) != len(modes)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isfinite(weight_sum)
        or weight_sum <= 0
    ):
        raise ValueError(f"{pair.pair_id}: invalid VST sampling weights.")
    weights /= weight_sum
    sample_count = min(BETA_PRIME_SAMPLE_SIZE, len(modes))
    sample_indices = rng.choice(
        len(modes), size=sample_count, replace=False, p=weights
    )
    sampled_modes = modes[sample_indices]
    sampled_parameters = parameters[sample_indices]
    mad_values = _estimate_beta_prime_mad(
        sampled_parameters,
        rng,
        sample_size=BETA_PRIME_DRAWS,
    )

    centers, rolling_mad = se.RollingVal(
        sampled_modes,
        mad_values,
        mode=[np.median, {}],
        win_size=SHRINK_WINDOW_SIZE,
        step=SHRINK_STEP,
        extended=True,
    )
    valid_curve = (
        np.isfinite(centers)
        & np.isfinite(rolling_mad)
        & (rolling_mad > 0)
    )
    centers = np.asarray(centers)[valid_curve]
    rolling_mad = np.asarray(rolling_mad)[valid_curve]
    if len(centers) < 2:
        raise ValueError(f"{pair.pair_id}: not enough rolling MAD windows for VST.")

    order = np.argsort(centers)
    centers = centers[order]
    rolling_mad = rolling_mad[order]
    centers, unique_indices = np.unique(centers, return_index=True)
    rolling_mad = rolling_mad[unique_indices]
    if len(centers) < 2:
        raise ValueError(f"{pair.pair_id}: VST requires two distinct mode windows.")

    smoothed_mad = gaussian_filter1d(
        rolling_mad, sigma=(SHRINK_WINDOW_SIZE / SHRINK_STEP) / 2
    )
    inverse_sigma = 1.0 / (smoothed_mad + EPSILON)
    vst_values = np.zeros_like(centers, dtype=np.float64)
    vst_values[1:] = cumulative_trapezoid(inverse_sigma, centers)
    vst_values -= vst_values[0]
    transform = interp1d(
        centers,
        vst_values,
        kind="linear",
        bounds_error=False,
        fill_value=(vst_values[0], vst_values[-1]),
    )
    se.PickleDump(transform, str(transform_path))
    _save_vst_figures(
        pair,
        vst_root,
        transform,
        win_root,
        control_replicate_id,
        merged_root,
        betaprime_root,
        blacklist,
        rewrite,
    )
    print(f"{pair.pair_id}: saved VST transform and diagnostic figures.")


def process_plan(
    plan: ProcessingPlan,
    replicate_ids: dict[str, str],
    win_root: Path,
    merged_root: Path,
    shrunk_root: Path,
    betaprime_root: Path,
    vst_root: Path,
    blacklist: list[np.ndarray],
    rewrite: bool,
    seed: int | None,
) -> None:
    for directory in (merged_root, shrunk_root, betaprime_root, vst_root):
        directory.mkdir(parents=True, exist_ok=True)

    merge_replicate_groups(plan, replicate_ids, win_root, merged_root, rewrite)
    rng = np.random.default_rng(seed)
    shrunk_cache: dict[str, list[np.ndarray]] = {}

    for pair in plan.pair_groups:
        if len(pair.target_files) < 2:
            raise ValueError(
                f"{pair.pair_id}: target dataset {pair.target_dataset} has fewer "
                "than two replicates; sample variance cannot be estimated."
            )
        if len(pair.control_files) < 2:
            raise ValueError(
                f"{pair.pair_id}: control dataset {pair.control_dataset} has fewer "
                "than two replicates; sample variance cannot be estimated."
            )

        target_mean_path, target_std_path = merge_target_pair(
            pair, win_root, merged_root, rewrite
        )
        _save_target_replicate_concordance(
            pair,
            win_root,
            vst_root.parent / "replicate_qc",
            blacklist,
            rewrite,
        )
        control_replicate_id = replicate_ids[pair.control_dataset]
        control_mean_path = _pickle_pair(merged_root, control_replicate_id, "mean")
        control_std_path = _pickle_pair(merged_root, control_replicate_id, "std")

        target_shrunk = shrink_standard_deviation(
            pair.pair_id,
            target_mean_path,
            target_std_path,
            shrunk_root,
            rng,
            rewrite,
        )
        if control_replicate_id not in shrunk_cache:
            shrunk_cache[control_replicate_id] = shrink_standard_deviation(
                control_replicate_id,
                control_mean_path,
                control_std_path,
                shrunk_root,
                rng,
                rewrite,
            )
        del target_shrunk

        calculate_beta_prime(
            pair,
            merged_root,
            shrunk_root,
            betaprime_root,
            control_replicate_id,
            rewrite,
        )
        calculate_vst(
            pair,
            control_replicate_id,
            win_root,
            merged_root,
            betaprime_root,
            vst_root,
            blacklist,
            rng,
            rewrite,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge ENCODE replicate currents, pair target datasets with their "
            "controls, shrink standard deviations, and calculate beta-prime/VST."
        )
    )
    parser.add_argument(
        "accessions",
        nargs="*",
        help="ENCODE file accessions (for example, ENCFF564SVK).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Text file of accessions separated by tabs, commas, or newlines.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Recompute outputs even when cached files already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the statistical sampling steps.",
    )
    parser.add_argument(
        "--genome",
        default="hg38",
        help="Genome name used for prepared resources (default: hg38).",
    )
    args = parser.parse_args(argv)

    metadata_path = ENCODE_ROOT / "meta_info.txt"
    try:
        accessions = parse_accessions(args.accessions, args.input_file)
        for directory in (
            ENCODE_ROOT / "merged",
            ENCODE_ROOT / "shrunk_std",
            ENCODE_ROOT / "betaprime",
            ENCODE_ROOT / "vst",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        fieldnames, rows = read_metadata(metadata_path)
        plan = build_processing_plan(accessions, rows)
        validate_window_currents(plan, WIN_ROOT)
        replicate_ids, _ = assign_identifiers(rows, plan)
        write_metadata(metadata_path, fieldnames, rows)
        blacklist = (
            load_autosome_blacklist(args.genome) if plan.pair_groups else []
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))

    for target_dataset, control_dataset in plan.unresolved_controls:
        print(
            f"Warning: no metadata files found for control dataset "
            f"{control_dataset} referenced by {target_dataset}; no pair will be made.",
            file=sys.stderr,
        )

    try:
        process_plan(
            plan,
            replicate_ids,
            WIN_ROOT,
            ENCODE_ROOT / "merged",
            ENCODE_ROOT / "shrunk_std",
            ENCODE_ROOT / "betaprime",
            ENCODE_ROOT / "vst",
            blacklist,
            args.rewrite,
            args.seed,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"merge_replicates: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
