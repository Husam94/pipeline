#!/usr/bin/env python3
"""Group overlapping sequence windows and generate experiment splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import poseigen_binmeths as bm
import poseigen_chisel.poseigen_chisel as chis
import poseigen_seaside.basics as se


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments"
DEFAULT_NUM_TRANSLATION_WINDOWS = 3
SPLIT_PROPORTIONS = [0.55, 0.15, 0.15, 0.15]
SPLIT_SCORE_EPSILON = 1e-12
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def _progress(message: str) -> None:
    print(f"[data_split] {message}", flush=True)


def _pickle_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required processed data not found: {path}")
    return se.PickleLoad(str(path))


def _pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    se.PickleDump(value, str(path))


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _normalize_proportions(proportions: np.ndarray, axis: int) -> np.ndarray:
    proportions = np.asarray(proportions, dtype=np.float64)
    minima = np.min(proportions, axis=axis, keepdims=True)
    ranges = np.ptp(proportions, axis=axis, keepdims=True)
    return np.divide(
        proportions - minima,
        ranges,
        out=np.zeros_like(proportions),
        where=ranges != 0,
    )


def overlap_groups(
    intervals: np.ndarray,
    chromosome_ids: np.ndarray,
) -> np.ndarray:
    intervals = np.asarray(intervals, dtype=np.int64)
    chromosome_ids = np.asarray(chromosome_ids, dtype=str)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("Window intervals must have shape (sites, 2).")
    if len(chromosome_ids) != len(intervals):
        raise ValueError("Chromosome IDs and window intervals must align.")
    if not len(intervals):
        return np.empty(0, dtype=np.int64)

    groups = np.empty(len(intervals), dtype=np.int64)
    group_id = -1
    for chromosome in np.unique(chromosome_ids):
        indices = np.flatnonzero(chromosome_ids == chromosome)
        order = indices[
            np.argsort(intervals[indices, 0], kind="stable")
        ]
        group_id += 1
        groups[order[0]] = group_id
        group_end = intervals[order[0], 1]

        for index in order[1:]:
            start, end = intervals[index]
            if start - group_end > 0:
                group_id += 1
                group_end = end
            else:
                group_end = max(group_end, end)
            groups[index] = group_id

    return groups


def assign_overlap_groups(markers: Any, sequence_size: int) -> Any:
    if isinstance(sequence_size, bool) or not isinstance(sequence_size, int):
        raise ValueError("seq_size must be a positive integer.")
    if sequence_size <= 0:
        raise ValueError("seq_size must be a positive integer.")
    missing_columns = {"loc", "subseq"}.difference(
        getattr(markers, "columns", ())
    )
    if missing_columns:
        raise ValueError(
            "Selected sites are missing overlap-group columns: "
            + ", ".join(sorted(missing_columns))
        )

    locations = np.asarray(markers["loc"], dtype=np.int64)
    chromosomes = np.asarray(markers["subseq"], dtype=str)
    left = sequence_size // 2
    starts = locations - left
    intervals = np.column_stack((starts, starts + sequence_size))
    grouped_markers = markers.copy()
    grouped_markers["group"] = overlap_groups(intervals, chromosomes)
    return grouped_markers


def _score_split(
    split: list[np.ndarray],
    packed_bind: np.ndarray,
    categories: np.ndarray,
    all_bin_counts: np.ndarray,
    bin_sizes: np.ndarray,
) -> float:
    split_scores = []
    for indices in split:
        counts = bm.BinCount(
            packed_bind[indices],
            uni=categories,
            prop=True,
            bin_sizes=bin_sizes,
        )
        relative_error = np.abs(counts - all_bin_counts) / (
            all_bin_counts + SPLIT_SCORE_EPSILON
        )
        split_scores.append(float(np.mean(bin_sizes * relative_error)))
    return float(np.mean(split_scores))


def _validate_split_partition(
    split: Any,
    groups: np.ndarray,
    source: str,
) -> list[np.ndarray]:
    if not isinstance(split, (list, tuple)) or len(split) != len(SPLIT_PROPORTIONS):
        raise ValueError(f"{source} split has a missing or invalid partition.")
    validated_split = []
    for indices in split:
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or not len(indices)
        ):
            raise ValueError(f"{source} split contains an empty or invalid partition.")
        validated_split.append(indices.astype(np.int64, copy=False))

    all_site_indices = np.concatenate(validated_split)
    if not np.array_equal(np.sort(all_site_indices), np.arange(len(groups))):
        raise ValueError(
            f"{source} split does not partition every selected site exactly once. "
            "Use --redo-splits to generate a new split."
        )
    split_group_sets = [set(groups[indices]) for indices in validated_split]
    for left_index, left_groups in enumerate(split_group_sets):
        if any(
            left_groups.intersection(split_group_sets[right_index])
            for right_index in range(left_index + 1, len(split_group_sets))
        ):
            raise ValueError(
                f"{source} split assigns an overlap group to multiple partitions. "
                "Use --redo-splits to generate a new split."
            )
    return validated_split


def _search_splits(
    experiment_name: str,
    horizontal_scores: np.ndarray,
    horizontal_bind: np.ndarray,
    categories: np.ndarray,
    bin_sizes: np.ndarray,
    groups: np.ndarray,
    split_candidates: int,
    split_cycles: int,
    seed: int | None,
) -> tuple[list[np.ndarray], float, list[float]]:
    all_bin_counts = bm.BinCount(
        horizontal_bind,
        uni=categories,
        prop=True,
        bin_sizes=bin_sizes,
    )
    if seed is not None:
        np.random.seed(seed)
    best_overall_split: list[np.ndarray] | None = None
    best_overall_score = float("inf")
    cycle_scores: list[float] = []
    max_workers = min(4, max(1, split_candidates))
    _progress(
        f"{experiment_name}: searching {split_candidates:,} candidate "
        f"splits across {split_cycles} cycle(s)."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cycle in range(split_cycles):
            _progress(
                f"{experiment_name}: starting split-search cycle "
                f"{cycle + 1}/{split_cycles}."
            )
            best_cycle_split: list[np.ndarray] | None = None
            best_cycle_score = float("inf")
            next_report = max(1, split_candidates // 10)
            for offset in range(0, split_candidates, 128):
                count = min(128, split_candidates - offset)
                candidates = bm.SplitGen_Random(
                    len(horizontal_scores),
                    num_splits=count,
                    proportions=SPLIT_PROPORTIONS,
                    group=groups,
                )
                if count == 1:
                    candidates = [candidates]
                scores = list(
                    executor.map(
                        lambda split: _score_split(
                            split,
                            horizontal_bind,
                            categories,
                            all_bin_counts,
                            bin_sizes,
                        ),
                        candidates,
                    )
                )
                best_index = int(np.argmin(scores))
                if scores[best_index] < best_cycle_score:
                    best_cycle_score = scores[best_index]
                    best_cycle_split = [
                        np.asarray(indices).copy()
                        for indices in candidates[best_index]
                    ]
                processed = offset + count
                if processed >= next_report or processed == split_candidates:
                    _progress(
                        f"{experiment_name}: split cycle "
                        f"{cycle + 1}/{split_cycles}, "
                        f"{processed:,}/{split_candidates:,} candidates checked."
                    )
                    next_report = processed + max(1, split_candidates // 10)
            if best_cycle_split is None:
                raise RuntimeError("Split generation returned no candidates.")
            cycle_scores.append(best_cycle_score)
            _progress(
                f"Split cycle {cycle + 1}/{split_cycles}: "
                f"best score {best_cycle_score:.6g}"
            )
            if best_cycle_score < best_overall_score:
                best_overall_score = best_cycle_score
                best_overall_split = best_cycle_split

    if best_overall_split is None:
        raise RuntimeError("No split was selected.")
    return (
        _validate_split_partition(best_overall_split, groups, "Generated"),
        best_overall_score,
        cycle_scores,
    )


def make_splits(
    experiment_dir: Path,
    markers: Any,
    bin_sizes: np.ndarray,
    sequence_size: int,
    split_candidates: int,
    split_cycles: int,
    seed: int | None,
    num_trans_wins: int,
    redo_splits: bool,
) -> None:
    if split_candidates <= 0 or split_cycles <= 0:
        raise ValueError("Split candidates and cycles must be positive.")
    if isinstance(num_trans_wins, bool) or num_trans_wins <= 0:
        raise ValueError("num_trans_wins must be a positive integer.")

    markers = assign_overlap_groups(markers, sequence_size)
    groups = np.asarray(markers["group"])
    if len(np.unique(groups)) < len(SPLIT_PROPORTIONS):
        raise ValueError(
            "At least four distinct overlap groups are needed to create all splits."
        )
    _pickle_dump(markers, experiment_dir / "flag" / "selected_sites.p")
    _progress(
        f"{experiment_dir.name}: assigned {len(np.unique(groups)):,} overlap "
        f"groups across {len(markers):,} selected sites."
    )

    bind_path = experiment_dir / "packs" / "tpack_output_bind.p"
    if not bind_path.is_file():
        raise FileNotFoundError("Missing translation-augmented output binding pack.")
    translated_bind = _pickle_load(bind_path)
    horizontal_bind = np.asarray(
        chis.StackTransAugsHorizontal(
            translated_bind,
            num_shifts=num_trans_wins,
        )
    )
    categories = np.unique(horizontal_bind)
    split_dir = experiment_dir / "split"
    split_figs = (
        experiment_dir
        / "output"
        / "figures"
        / "data_preparation"
        / "split"
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split.p"
    reuse_existing_split = split_path.is_file() and not redo_splits
    if reuse_existing_split:
        _progress(
            f"{experiment_dir.name}: existing split found; reusing it. "
            "Pass --redo-splits to search for a new split."
        )
        best_overall_split = _validate_split_partition(
            _pickle_load(split_path),
            groups,
            "Existing",
        )
        cycle_scores: list[float] = []
    else:
        score_path = experiment_dir / "packs" / "tpack_output_scores.p"
        if not score_path.is_file():
            raise FileNotFoundError(
                "Missing translation-augmented output score pack."
            )
        translated_scores = _pickle_load(score_path)
        horizontal_scores = np.asarray(
            chis.StackTransAugsHorizontal(
                translated_scores,
                num_shifts=num_trans_wins,
            )
        )
        if len(groups) != len(horizontal_scores):
            raise ValueError("Grouped split inputs do not align with selected sites.")
        (
            best_overall_split,
            best_overall_score,
            cycle_scores,
        ) = _search_splits(
            experiment_dir.name,
            horizontal_scores,
            horizontal_bind,
            categories,
            bin_sizes,
            groups,
            split_candidates,
            split_cycles,
            seed,
        )
        _pickle_dump(best_overall_split, split_path)
        split_metadata = {
            "best_score": best_overall_score,
            "cycle_scores": cycle_scores,
            "proportions": SPLIT_PROPORTIONS,
            "split_candidates_per_cycle": split_candidates,
            "split_cycles": split_cycles,
            "seed": seed,
        }
        _write_json(split_metadata, split_dir / "split_metadata.json")
        _write_json(
            split_metadata,
            experiment_dir
            / "output"
            / "tables"
            / "data_preparation"
            / "split_metadata.json",
        )

    expanded_split = bm.SplitExpander(
        1,
        num_trans_wins,
        best_overall_split,
        ma=None,
    )
    _pickle_dump(expanded_split, split_dir / "split_transaug.p")

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        ["Train", "Stop", "Eval", "Test"],
        [len(indices) for indices in best_overall_split],
        color="slateblue",
    )
    axis.set(title="Selected sites by split", ylabel="Sites")
    _save_figure(figure, split_figs / "split_sizes.png")

    if cycle_scores:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(np.arange(1, len(cycle_scores) + 1), cycle_scores, marker="o")
        axis.set(
            title="Best split score by cycle",
            xlabel="Split-search cycle",
            ylabel="Weighted relative bin error",
        )
        _save_figure(figure, split_figs / "split_scores.png")

    split_labels = ["Train", "Stop", "Eval", "Test"]
    split_bin_proportions = np.asarray(
        [
            bm.BinCount(
                horizontal_bind[indices],
                uni=categories,
                prop=True,
            )
            for indices in best_overall_split
        ]
    )
    for axis_index, normalization in ((0, "column"), (1, "row")):
        normalized_proportions = _normalize_proportions(
            split_bin_proportions,
            axis=axis_index,
        )
        figure, axis = plt.subplots(figsize=(10, 6))
        image = axis.imshow(
            normalized_proportions,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axis, label="Normalized Proportion")
        axis.set_xlabel("Bin")
        axis.set_ylabel("Split")
        axis.set_yticks(range(len(split_labels)))
        axis.set_yticklabels(split_labels)
        axis.set_xticks(range(split_bin_proportions.shape[1]))
        axis.set_xticklabels(range(1, split_bin_proportions.shape[1] + 1))
        axis.set_title(
            f"Bin Proportions by Split ({normalization.title()}-wise Normalized)"
        )
        figure.tight_layout()
        _save_figure(
            figure,
            split_figs / f"bin_proportions_{normalization}_normalized.png",
        )

    output_scores = np.asarray(
        _pickle_load(experiment_dir / "packs" / "pack_output_scores.p")
    )
    if not output_scores.size:
        raise ValueError("Output score pack is empty; cannot plot its distribution.")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(output_scores.reshape(-1), bins=50, color="steelblue")
    axis.set(
        title="Output value distribution",
        xlabel="VST value",
        ylabel="Values",
    )
    _save_figure(figure, split_figs / "output_value_distribution.png")

    _progress(f"{experiment_dir.name}: split files and figures saved to {split_dir}.")


def _read_experiment(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment JSON file not found: {path}")
    with path.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    if not isinstance(experiment, dict):
        raise ValueError(f"{path} must contain one JSON experiment object.")
    return experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assign overlap groups to selected sequence windows and reuse or "
            "generate grouped train/stop/eval/test splits."
        )
    )
    parser.add_argument("experiment_file", type=Path, help="Experiment JSON file")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Experiment outputs directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--split-candidates",
        type=int,
        default=10_000,
        help="Random candidates per cycle with --redo-splits (default: 10000)",
    )
    parser.add_argument(
        "--split-cycles",
        type=int,
        default=10,
        help="Split-search cycles with --redo-splits (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducible split search with --redo-splits",
    )
    parser.add_argument(
        "--redo-splits",
        action="store_true",
        help="Search for and replace existing splits instead of reusing them",
    )
    args = parser.parse_args(argv)

    try:
        experiment = _read_experiment(args.experiment_file)
        name = experiment.get("experiment_name")
        if not isinstance(name, str) or not EXPERIMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError("experiment_name must be a valid directory name.")
        sequence_size = experiment.get("seq_size")
        num_trans_wins = experiment.get(
            "num_trans_wins", DEFAULT_NUM_TRANSLATION_WINDOWS
        )
        if num_trans_wins is None:
            num_trans_wins = DEFAULT_NUM_TRANSLATION_WINDOWS
        if (
            isinstance(sequence_size, bool)
            or not isinstance(sequence_size, int)
            or sequence_size <= 0
        ):
            raise ValueError("seq_size must be a positive integer.")
        if (
            isinstance(num_trans_wins, bool)
            or not isinstance(num_trans_wins, int)
            or num_trans_wins <= 0
        ):
            raise ValueError("num_trans_wins must be a positive integer.")
        if args.split_candidates <= 0 or args.split_cycles <= 0:
            raise ValueError("--split-candidates and --split-cycles must be positive.")

        experiment_dir = args.output_root / name
        markers = _pickle_load(experiment_dir / "flag" / "selected_sites.p")
        bin_sizes = np.asarray(
            _pickle_load(experiment_dir / "packs" / "output_bin_sizes.p")
        )
        _progress(f"{name}: generating overlap-aware data splits.")
        make_splits(
            experiment_dir,
            markers,
            bin_sizes,
            sequence_size,
            args.split_candidates,
            args.split_cycles,
            args.seed,
            num_trans_wins,
            args.redo_splits,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"data_split: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
