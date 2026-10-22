#!/usr/bin/env python3
"""Merge ENCODE replicates and process target/control pairs."""

from __future__ import annotations

import argparse
import csv
import re
import secrets
import string
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODE_ROOT = PROJECT_ROOT / "data" / "ENCODE"
WIN_ROOT = ENCODE_ROOT / "win"
GENOME_ROOT = PROJECT_ROOT / "data" / "genomes" / "hg38"

SHRINK_WINDOW_SIZE = 0.01
SHRINK_STEP = 0.005
SHRINK_SAMPLE_SIZE = 100_000
HISTOGRAM_BINS = 1_000
BETA_PRIME_SAMPLE_SIZE = 100_000
BETA_PRIME_DRAWS = 1_000
BETA_PRIME_CHUNK_SIZE = 512
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
        if not (win_root / f"{(row.get('File name') or '').strip()}_win.p").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing window currents in {win_root}: "
            + ", ".join(dict.fromkeys(missing))
            + ". Run processing_initial.py for these files first."
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
            win_root / f"{(row.get('File name') or '').strip()}_win.p"
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
        win_root / f"{accession}_win.p" for accession in pair.control_files
    ]
    target_paths = [
        win_root / f"{accession}_win.p" for accession in pair.target_files
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


def load_autosome_blacklist() -> list[np.ndarray]:
    basic_ids_path = GENOME_ROOT / "hg38_basic_id_sizes.p"
    blacklist_path = GENOME_ROOT / "hg38_basic_BLC_LR50.p"
    missing = [
        path for path in (basic_ids_path, blacklist_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing hg38 blacklist resource(s): "
            + ", ".join(str(path) for path in missing)
        )
    basic_ids, _ = se.PickleLoad(str(GENOME_ROOT / "hg38_basic_id_sizes"))
    if len(basic_ids) < 4:
        raise ValueError("hg38 basic chromosome list is too short to select autosomes.")
    autosome_ids = basic_ids[:-3]
    blacklist = se.PickleLoad(str(GENOME_ROOT / "hg38_basic_BLC_LR50"))
    autosome_blacklist = chis.CurrentSelect(blacklist, basic_ids, autosome_ids)
    if len(autosome_blacklist) != len(autosome_ids):
        raise ValueError("Could not align hg38 blacklist currents to autosomes.")
    return autosome_blacklist


def calculate_vst(
    pair: PairGroup,
    control_replicate_id: str,
    merged_root: Path,
    betaprime_root: Path,
    vst_root: Path,
    blacklist: list[np.ndarray],
    rng: np.random.Generator,
    rewrite: bool,
) -> None:
    transform_path = _pickle_pair(vst_root, pair.pair_id, "vst_transform")
    if transform_path.is_file() and not rewrite:
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
    print(f"{pair.pair_id}: saved variance-stabilizing transform.")


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
        blacklist = load_autosome_blacklist() if plan.pair_groups else []
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
        print(f"processing_replicates: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
