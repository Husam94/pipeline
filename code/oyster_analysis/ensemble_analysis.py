#!/usr/bin/env python3
"""Build the bootstrap-weighted Oyster repeat ensemble for an experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

import poseigen_compass as co
import poseigen_seaside.basics as se


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_FILE = PROJECT_ROOT / "data" / "experiments.json"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "experiments"
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")
MODEL_NAME_BY_INPUTS = {False: "Oyster", True: "Dual_Oyster"}
DEFAULT_XP_RUN = "xp2_0"
DEFAULT_LOSS = "MSE-R"


def read_experiment(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment JSON file not found: {path}")
    with path.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    if not isinstance(experiment, dict):
        raise ValueError(f"{path} must contain one JSON experiment object.")
    name = experiment.get("experiment_name")
    if not isinstance(name, str) or not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError("The experiment JSON has an invalid experiment_name.")
    input_signals = experiment.get("input_signals", [])
    if input_signals is None:
        input_signals = []
    if not isinstance(input_signals, list) or any(
        not isinstance(signal, str) or not signal.strip()
        for signal in input_signals
    ):
        raise ValueError("input_signals must be a list of non-empty signal IDs.")
    experiment["input_signals"] = input_signals
    return experiment


def model_name_for_experiment(experiment: dict[str, Any]) -> str:
    return MODEL_NAME_BY_INPUTS[bool(experiment["input_signals"])]


def model_analysis_dir(experiment_dir: Path, model_config: str) -> Path:
    return experiment_dir / "output" / "oyster_models" / model_config


def _load_model_selection(
    experiment_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    path = experiment_dir / "output" / "tables" / "model_selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Model selection is missing: {path}. Run model_selection.py first."
        )
    with path.open(encoding="utf-8") as selection_file:
        selection = json.load(selection_file)
    if not isinstance(selection, dict):
        raise ValueError(f"Unexpected model selection report: {path}.")
    selected = selection.get("selected")
    if (
        selection.get("model_name") != model_name
        or not isinstance(selected, dict)
        or not isinstance(selected.get("fit_folder"), str)
        or not isinstance(selected.get("model_config"), str)
    ):
        raise ValueError(
            f"Model selection in {path} does not match {model_name}."
        )
    return selection


def _json_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return _json_metadata(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def resolve_fit_folder(
    experiment_dir: Path,
    model_name: str,
    xp_run: str | None = None,
    loss: str | None = None,
) -> tuple[Path, dict[str, Any], int, str]:
    model_dir = experiment_dir / "search" / model_name
    ro_args_path = model_dir / "RO_args.p"
    if not ro_args_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_name} optimization arguments: {ro_args_path}. "
            "Run model_search.py first."
        )
    ro_args = se.PickleLoad(str(ro_args_path))
    ro_path = Path(ro_args["pn"])
    if not ro_path.is_absolute():
        ro_path = experiment_dir / ro_path
    ro_args["pn"] = str(ro_path)
    best_index_path = ro_path / "best_config_idx.p"
    if not best_index_path.is_file():
        raise FileNotFoundError(
            f"Missing best configuration index: {best_index_path}. "
            "Complete model_search.py's top-model refinement first."
        )
    best_index = int(se.PickleLoad(str(best_index_path)))
    match = re.fullmatch(r"RO_(\d+)", ro_path.name)
    if match is None:
        raise ValueError(f"Could not determine optimizer iteration from {ro_path}.")
    config_name = f"{model_name}_{match.group(1)}_{best_index}"
    if xp_run is None and loss is None:
        selected = _load_model_selection(experiment_dir, model_name)["selected"]
        if selected["model_config"] != config_name:
            raise ValueError(
                f"Selected model configuration {selected['model_config']} does "
                f"not match the current search result {config_name}."
            )
        fit_folder = Path(selected["fit_folder"])
        if not fit_folder.is_absolute():
            fit_folder = experiment_dir / fit_folder
    else:
        fit_folder = (
            experiment_dir
            / "xps"
            / (xp_run or DEFAULT_XP_RUN)
            / config_name
            / (loss or DEFAULT_LOSS)
        )
    if not fit_folder.is_dir():
        raise FileNotFoundError(
            f"Fitting results not found: {fit_folder}. "
            "Run model_fit_exp.py first and check the XP run/loss."
        )
    return fit_folder, ro_args, best_index, config_name


def make_weighted_ensemble(
    experiment_dir: Path,
    model_name: str,
    xp_run: str | None,
    loss: str | None,
    repeats: int,
    bootstrap: int,
    top: int,
    min_weight: float,
) -> Path:
    selection = (
        _load_model_selection(experiment_dir, model_name)
        if xp_run is None and loss is None
        else None
    )
    fit_folder, ro_args, best_index, config_name = resolve_fit_folder(
        experiment_dir, model_name, xp_run, loss
    )
    metrics_path = fit_folder / "Mets.p"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Model repeat metrics not found: {metrics_path}")
    metrics = se.PickleLoad(str(metrics_path))
    if not isinstance(metrics, (list, tuple)) or not metrics or not isinstance(
        metrics[0], dict
    ):
        raise ValueError(f"Unexpected metrics format in {metrics_path}.")
    if any(index not in metrics[0] for index in range(repeats)):
        raise ValueError(
            f"{metrics_path} does not contain scores for all {repeats} repeats."
        )

    all_weights = np.asarray(
        co.BootstrapWeights(
            str(fit_folder) + "/",
            repeats=repeats,
            bootstrap=bootstrap,
            top=top,
        ),
        dtype=float,
    )
    if all_weights.shape != (repeats,) or not np.all(np.isfinite(all_weights)):
        raise ValueError("BootstrapWeights returned invalid repeat weights.")
    selected_indices = np.flatnonzero(all_weights >= min_weight)
    if not len(selected_indices):
        raise ValueError(
            f"No model repeat met the minimum bootstrap weight {min_weight:g}; "
            "lower --min-weight or increase --bootstrap."
        )
    selected_weights = all_weights[selected_indices]
    selected_weights /= selected_weights.sum()
    source_model_paths = [
        (fit_folder / f"0_{index}_Mod").resolve() for index in selected_indices
    ]
    for model_path in source_model_paths:
        if not Path(f"{model_path}.pt").is_file():
            raise FileNotFoundError(f"Weighted model checkpoint not found: {model_path}.pt")

    ensemble_dir = model_analysis_dir(experiment_dir, config_name)
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    optimizer_path = Path(ro_args["pn"])
    if not optimizer_path.is_absolute():
        optimizer_path = experiment_dir / optimizer_path
    model_config = se.PickleLoad(str(optimizer_path / "best_config.p"))
    if not isinstance(model_config, dict):
        raise ValueError(
            f"Unexpected best model configuration in {optimizer_path / 'best_config.p'}."
        )
    with (ensemble_dir / "model_config.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(_json_metadata(model_config), config_file, indent=2)
        config_file.write("\n")

    model_files_dir = ensemble_dir / "models"
    model_files_dir.mkdir(parents=True, exist_ok=True)
    model_paths: list[str] = []
    for repeat_index, source_path in zip(selected_indices, source_model_paths):
        destination = model_files_dir / f"repeat_{repeat_index}_Mod"
        shutil.copy2(
            Path(f"{source_path}.pt"),
            Path(f"{destination}.pt"),
        )
        model_paths.append(destination.relative_to(ensemble_dir).as_posix())
    selected_fit = selection["selected"] if selection is not None else {}
    source_xp_run = selected_fit.get("xp_run", xp_run or DEFAULT_XP_RUN)
    source_condition = selected_fit.get("condition", loss or DEFAULT_LOSS)
    source_condition_label = selected_fit.get(
        "condition_label",
        loss or DEFAULT_LOSS,
    )
    source_fit_folder = fit_folder.resolve()
    relative_fit_folder = source_fit_folder.relative_to(
        experiment_dir.resolve()
    ).as_posix()
    manifest = {
        "experiment_name": experiment_dir.name,
        "model_name": model_name,
        "optimizer_config": config_name,
        "optimizer_best_config_index": best_index,
        "model_config_file": "model_config.json",
        "xp_run": source_xp_run,
        "loss": source_condition_label,
        "fit_condition": source_condition,
        "source_fit_folder": relative_fit_folder,
        "source_fit_folder_relative": relative_fit_folder,
        "selection_reason": (
            selection["selection_reason"]
            if selection is not None
            else "manual_override"
        ),
        "repeats": repeats,
        "bootstrap": bootstrap,
        "top": top,
        "minimum_weight": min_weight,
        "repeat_indices": selected_indices.tolist(),
        "weights": selected_weights.tolist(),
        "source_model_paths_relative": [
            Path(f"{path}.pt").relative_to(experiment_dir.resolve()).as_posix()
            for path in source_model_paths
        ],
        "model_paths": model_paths,
    }
    with (ensemble_dir / "ensemble_manifest.json").open(
        "w", encoding="utf-8"
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")
    se.PickleDump(all_weights, str(ensemble_dir / "bootstrap_weights.p"))
    se.PickleDump(manifest, str(ensemble_dir / "ensemble_manifest.p"))
    weights_path = (
        experiment_dir
        / "output"
        / "tables"
        / "oyster_models"
        / config_name
        / "ensemble_weights.csv"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open(
        "w", encoding="utf-8", newline=""
    ) as weights_file:
        writer = csv.writer(weights_file)
        writer.writerow(["repeat_index", "bootstrap_weight", "ensemble_weight"])
        selected_map = dict(zip(selected_indices.tolist(), selected_weights.tolist()))
        for index, weight in enumerate(all_weights):
            writer.writerow([index, weight, selected_map.get(index, 0.0)])
    return ensemble_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the repeat-weighted Oyster ensemble used for interpretation. "
            "Input signals select Dual_Oyster; no inputs select Oyster."
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
    parser.add_argument(
        "--xp-run",
        help="Override the selected fitting run; omit both overrides to use model_selection.json.",
    )
    parser.add_argument(
        "--loss",
        help="Override the selected fit condition; omit both overrides to use model_selection.json.",
    )
    parser.add_argument("--repeats", type=int, default=15, help="Model repeats (default: 15)")
    parser.add_argument(
        "--bootstrap", type=int, default=500, help="Bootstrap samples (default: 500)"
    )
    parser.add_argument("--top", type=int, default=3, help="Top repeats per bootstrap (default: 3)")
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        help="Keep repeats with raw bootstrap weight at least this value (default: 0.05)",
    )
    args = parser.parse_args(argv)
    if args.repeats <= 0 or args.bootstrap <= 0 or args.top <= 0:
        parser.error("--repeats, --bootstrap, and --top must be positive.")
    if not np.isfinite(args.min_weight) or not 0 <= args.min_weight <= 1:
        parser.error("--min-weight must be between 0 and 1.")

    try:
        experiment = read_experiment(args.experiment_file)
        experiment_dir = (
            args.experiment_root.resolve() / experiment["experiment_name"]
        )
        model_name = model_name_for_experiment(experiment)
        ensemble_dir = make_weighted_ensemble(
            experiment_dir,
            model_name,
            args.xp_run,
            args.loss,
            args.repeats,
            args.bootstrap,
            args.top,
            args.min_weight,
        )
        print(
            f"Created {model_name} weighted ensemble for "
            f"{experiment['experiment_name']!r}: {ensemble_dir}",
            flush=True,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"ensemble_analysis: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
