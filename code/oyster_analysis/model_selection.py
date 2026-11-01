#!/usr/bin/env python3
"""Select the Oyster fit recipe using the existing bootstrap RMSE scores."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import poseigen_seaside.basics as se

from ensemble_analysis import (
    DEFAULT_EXPERIMENT_FILE,
    DEFAULT_EXPERIMENT_ROOT,
    model_name_for_experiment,
    read_experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_LOW = 5
BOOTSTRAP_HIGH = 95
XP_SPECS = (
    ("xp2", "loss"),
    ("xp3", "without_replacement"),
    ("xp3B", "with_replacement"),
)
TRACKER_FIELDS = (
    "experiment_name",
    "model_name",
    "selected_xp_run",
    "selected_condition",
    "selected_model_config",
    "median_rmse",
    "baseline_median_rmse",
    "relative_change_median_percent",
    "relative_change_low_percent",
    "relative_change_high_percent",
    "selection_reason",
    "selected_at_utc",
)


def _native_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _load_pickle(path_without_suffix: Path) -> Any:
    return se.PickleLoad(str(path_without_suffix))


def _validated_rmse(scores: Any, source: str) -> np.ndarray:
    if not isinstance(scores, dict) or "rmses" not in scores:
        raise ValueError(f"Bootstrap RMSE scores are missing from {source}.")
    values = np.asarray(scores["rmses"], dtype=float).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"Bootstrap RMSE scores in {source} are invalid.")
    return values


def _relative_bootstrap_change(
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    if np.any(baseline <= 0):
        raise ValueError("Baseline bootstrap RMSE values must be positive.")
    changes = ((candidate[:, None] / baseline[None, :]) - 1.0) * 100.0
    low, median, high = np.percentile(
        changes,
        [BOOTSTRAP_LOW, 50, BOOTSTRAP_HIGH],
    )
    return {
        "median_percent": float(median),
        "low_percent": float(low),
        "high_percent": float(high),
        "statistically_better": bool(high < 0),
    }


def _fit_candidate(
    experiment_dir: Path,
    xp_run: str,
    xp_key: str,
    condition: Any,
    condition_type: str,
    model_config: str,
    scores: Any,
) -> dict[str, Any]:
    condition = _native_scalar(condition)
    condition_name = str(condition)
    if xp_key == "xp2":
        label = f"loss={condition_name}"
    else:
        label = f"{condition_type}={condition_name}"

    fit_folder = (
        experiment_dir
        / "xps"
        / xp_run
        / model_config
        / condition_name
    )
    if not fit_folder.is_dir():
        raise FileNotFoundError(f"Fitted model condition is missing: {fit_folder}")
    if not (fit_folder / "Mets.p").is_file():
        raise FileNotFoundError(
            f"Fit repeat metrics are missing: {fit_folder / 'Mets.p'}"
        )

    rmse_values = _validated_rmse(scores, f"{xp_run}/{model_config}/{condition_name}")
    return {
        "xp_run": xp_run,
        "xp_key": xp_key,
        "condition": condition,
        "condition_name": condition_name,
        "condition_label": label,
        "model_config": model_config,
        "fit_folder": fit_folder.relative_to(experiment_dir).as_posix(),
        "median_rmse": float(np.median(rmse_values)),
        "mean_rmse": float(np.mean(rmse_values)),
        "low_rmse": float(np.percentile(rmse_values, BOOTSTRAP_LOW)),
        "high_rmse": float(np.percentile(rmse_values, BOOTSTRAP_HIGH)),
        "_rmse_values": rmse_values,
    }


def _experiment_candidates(
    experiment_dir: Path,
    model_name: str,
) -> list[dict[str, Any]]:
    xps_dir = experiment_dir / "xps"
    candidates: list[dict[str, Any]] = []
    model_prefix = f"{model_name}_"

    for xp_key, condition_type in XP_SPECS:
        xpx_dict = _load_pickle(xps_dir / f"{xp_key}_dict")
        if not isinstance(xpx_dict, dict):
            raise ValueError(f"Unexpected XP metadata in {xps_dir / f'{xp_key}_dict.p'}.")
        trial = xpx_dict.get("trial")
        if isinstance(trial, bool) or not isinstance(trial, int):
            raise ValueError(f"Invalid trial number in {xp_key}_dict.")
        xp_run = f"{xp_key}_{trial}"
        ensemb_dict = _load_pickle(xps_dir / xp_run / "ensemb_dict")
        if not isinstance(ensemb_dict, dict):
            raise ValueError(f"Unexpected bootstrap scores in {xps_dir / xp_run}.")

        model_configs = [
            name
            for name in xpx_dict.get("modx_names_full", [])
            if isinstance(name, str) and name.startswith(model_prefix)
        ]
        if not model_configs:
            raise ValueError(
                f"{xp_run} contains no top-refined configuration for {model_name}."
            )

        combinations = xpx_dict.get("combs")
        if not isinstance(combinations, list) or not combinations:
            raise ValueError(f"{xp_key}_dict has no experiment conditions.")
        for model_config in model_configs:
            model_scores = ensemb_dict.get(model_config)
            if not isinstance(model_scores, dict):
                raise ValueError(
                    f"Bootstrap scores are missing for {xp_run}/{model_config}."
                )
            for combination in combinations:
                if (
                    not isinstance(combination, (list, tuple))
                    or len(combination) != 2
                    or not isinstance(combination[1], (list, tuple))
                    or not combination[1]
                ):
                    raise ValueError(f"Malformed condition in {xp_key}_dict.")
                condition = _native_scalar(combination[1][0])
                if condition not in model_scores:
                    raise ValueError(
                        f"Bootstrap RMSE is missing for "
                        f"{xp_run}/{model_config}/{condition}."
                    )
                candidate = _fit_candidate(
                    experiment_dir,
                    xp_run,
                    xp_key,
                    condition,
                    condition_type,
                    model_config,
                    model_scores[condition],
                )
                candidate["is_baseline_alias"] = (
                    (xp_key == "xp3" and condition == 1.0)
                    or (xp_key == "xp3B" and condition == 5)
                )
                candidates.append(candidate)

    return candidates


def _write_tracker(tracker_path: Path, row: dict[str, Any]) -> Path:
    latest_by_experiment: dict[str, dict[str, str]] = {}
    if tracker_path.is_file():
        with tracker_path.open(encoding="utf-8", newline="") as tracker_file:
            for existing in csv.DictReader(tracker_file):
                name = existing.get("experiment_name")
                if name:
                    latest_by_experiment[name] = existing

    latest_by_experiment[row["experiment_name"]] = {
        field: str(row.get(field, ""))
        for field in TRACKER_FIELDS
    }
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = tracker_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as tracker_file:
        writer = csv.DictWriter(tracker_file, fieldnames=TRACKER_FIELDS)
        writer.writeheader()
        writer.writerows(
            latest_by_experiment[name]
            for name in sorted(latest_by_experiment)
        )
    temporary_path.replace(tracker_path)
    return tracker_path


def select_best_recipe(
    experiment_dir: Path,
    tracker_path: Path,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    model_name = model_name_for_experiment(experiment)
    candidates = _experiment_candidates(experiment_dir, model_name)
    baseline_candidates = [
        candidate
        for candidate in candidates
        if candidate["xp_key"] == "xp2"
        and candidate["condition"] == "MSE-R"
    ]
    if not baseline_candidates:
        raise ValueError(
            "The default MSE-R fit is missing; complete model_fit_exp.py first."
        )

    baseline_by_config = {
        candidate["model_config"]: candidate
        for candidate in baseline_candidates
    }
    baseline = min(baseline_candidates, key=lambda candidate: candidate["median_rmse"])
    alternatives: list[dict[str, Any]] = []
    for candidate in candidates:
        if (
            candidate["xp_key"] == "xp2"
            and candidate["condition"] == "MSE-R"
        ) or candidate["is_baseline_alias"]:
            continue
        reference = baseline_by_config.get(candidate["model_config"])
        if reference is None:
            raise ValueError(
                f"No MSE-R reference exists for {candidate['model_config']}."
            )
        comparison = _relative_bootstrap_change(
            candidate["_rmse_values"],
            reference["_rmse_values"],
        )
        candidate["comparison_to_mse_r"] = comparison
        alternatives.append(candidate)

    statistically_better = [
        candidate
        for candidate in alternatives
        if candidate["comparison_to_mse_r"]["statistically_better"]
    ]
    selected = (
        min(statistically_better, key=lambda candidate: candidate["median_rmse"])
        if statistically_better
        else baseline
    )
    selected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        }

    selected_public = public_candidate(selected)
    report = {
        "schema_version": 1,
        "experiment_name": experiment["experiment_name"],
        "model_name": model_name,
        "selected_at_utc": selected_at,
        "method": {
            "metric": "reciprocal-bin-weighted RMSE",
            "evaluation_split_index": 2,
            "bootstrap_unit": "model repeats, using the existing XP scoring method",
            "rmse_interval_percentiles": [BOOTSTRAP_LOW, BOOTSTRAP_HIGH],
            "interval_confidence_percent": BOOTSTRAP_HIGH - BOOTSTRAP_LOW,
            "selection_rule": (
                "Keep MSE-R unless an alternative's existing repeat-bootstrap "
                "relative RMSE interval is entirely below zero; among qualifying "
                "alternatives, choose the lowest median RMSE."
            ),
        },
        "baseline": public_candidate(baseline),
        "selected": selected_public,
        "candidates": [public_candidate(candidate) for candidate in candidates],
        "selection_reason": (
            "statistically_better_than_mse_r"
            if statistically_better
            else "mse_r_default"
        ),
    }
    output_path = experiment_dir / "output" / "tables" / "model_selection.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, allow_nan=False)
        output_file.write("\n")

    relative_change = selected.get("comparison_to_mse_r", {})
    tracker_row = {
        "experiment_name": experiment["experiment_name"],
        "model_name": model_name,
        "selected_xp_run": selected["xp_run"],
        "selected_condition": selected["condition_label"],
        "selected_model_config": selected["model_config"],
        "median_rmse": selected["median_rmse"],
        "baseline_median_rmse": baseline["median_rmse"],
        "relative_change_median_percent": relative_change.get("median_percent", 0),
        "relative_change_low_percent": relative_change.get("low_percent", 0),
        "relative_change_high_percent": relative_change.get("high_percent", 0),
        "selection_reason": report["selection_reason"],
        "selected_at_utc": selected_at,
    }
    tracker_path = _write_tracker(tracker_path, tracker_row)
    report["tracker_path"] = str(tracker_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Oyster XP conditions by bootstrap RMSE, retaining MSE-R "
            "unless an alternative is statistically better."
        )
    )
    parser.add_argument(
        "experiment_file",
        nargs="?",
        type=Path,
        default=DEFAULT_EXPERIMENT_FILE,
        help=f"Single-experiment JSON file (default: {DEFAULT_EXPERIMENT_FILE})",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help=f"Parent directory for experiment data (default: {DEFAULT_EXPERIMENT_ROOT})",
    )
    args = parser.parse_args(argv)

    try:
        experiment = read_experiment(args.experiment_file)
        experiment_dir = args.experiment_root.resolve() / experiment["experiment_name"]
        report = select_best_recipe(
            experiment_dir,
            PROJECT_ROOT / "model_selection_tracker.csv",
            experiment,
        )
        selected = report["selected"]
        print(
            f"Selected {selected['xp_run']}/{selected['model_config']}/"
            f"{selected['condition_name']} for {report['experiment_name']!r} "
            f"(median RMSE={selected['median_rmse']:.6g}; "
            f"reason={report['selection_reason']}).",
            flush=True,
        )
        print(f"Saved selection report: {experiment_dir / 'xps' / 'model_selection.json'}")
        print(f"Updated tracker: {report['tracker_path']}")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"model_selection: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
