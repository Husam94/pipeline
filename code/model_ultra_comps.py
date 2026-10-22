#!/usr/bin/env python3
"""Compare Ultra model searches for one processed experiment."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import poseigen_seaside.basics as se


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_FILE = PROJECT_ROOT / "data" / "experiments.json"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "experiments"
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def read_experiment(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment JSON file not found: {path}")
    with path.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    if not isinstance(experiment, dict):
        raise ValueError(f"{path} must contain one JSON experiment object.")
    name = experiment.get("experiment_name")
    if not isinstance(name, str) or not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "experiment_name must start with a letter or digit and contain only "
            "letters, digits, spaces, '.', '_' or '-'."
        )
    return experiment


def opt_dict_topcans(
    opt_dict: dict[str, Any],
    metrics: dict[Any, Any],
    candidates: dict[Any, Any],
    boots: int = 100,
    params_of_interest: list[str] | None = None,
    top: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    if boots <= 0 or top <= 0:
        raise ValueError("boots and top must be positive integers.")
    if 0 not in metrics or 0 not in candidates:
        raise ValueError("Optimization outputs must have outer key 0.")
    metric_values = metrics[0]
    candidate_values = candidates[0]
    candidate_keys = list(candidate_values)
    if not candidate_keys:
        raise ValueError("Optimization output contains no candidates.")
    if any(key not in metric_values for key in candidate_keys):
        raise ValueError("Optimization metrics are missing one or more candidates.")
    if params_of_interest is None:
        params_of_interest = list(opt_dict)
    missing_params = [name for name in params_of_interest if name not in opt_dict]
    if missing_params:
        raise ValueError(
            "Optimization dictionary is missing comparison parameters: "
            + ", ".join(missing_params)
        )

    scores = np.asarray(
        [
            float(np.asarray(metric_values[key][0][0]).reshape(-1)[0])
            for key in candidate_keys
        ]
    )
    random = rng or np.random.default_rng()
    sampled_keys = np.asarray(candidate_keys, dtype=object)
    result = {name: np.empty((boots, len(opt_dict[name][0]))) for name in params_of_interest}
    candidate_count = len(candidate_keys)

    for bootstrap in range(boots):
        sampled_indices = random.integers(0, candidate_count, size=candidate_count)
        sampled_scores = scores[sampled_indices]
        top_indices = sampled_indices[np.argsort(sampled_scores)[:top]]
        sampled_candidates = [candidate_values[sampled_keys[index]] for index in sampled_indices]
        top_candidates = [candidate_values[sampled_keys[index]] for index in top_indices]

        for parameter in params_of_interest:
            possible_values = opt_dict[parameter][0]
            for value_index, value in enumerate(possible_values):
                expected_fraction = sum(
                    candidate.get(parameter) == value
                    for candidate in sampled_candidates
                ) / candidate_count
                observed_fraction = sum(
                    candidate.get(parameter) == value for candidate in top_candidates
                ) / len(top_candidates)
                result[parameter][bootstrap, value_index] = (
                    (observed_fraction - expected_fraction) / expected_fraction
                    if expected_fraction
                    else 0.0
                )
    return result


def opt_dict_ranks_table(
    rank_results: dict[str, np.ndarray],
    opt_dict: dict[str, Any],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["parameter", "value", "median_percent", "p5_percent", "p95_percent"]
        )
        for parameter, scores in rank_results.items():
            if parameter not in opt_dict:
                raise ValueError(
                    f"Optimization dictionary is missing parameter {parameter!r}."
                )
            values = opt_dict[parameter][0]
            if len(values) != scores.shape[1]:
                raise ValueError(
                    f"Rank results for {parameter!r} do not match its option values."
                )
            medians = np.median(scores, axis=0) * 100
            lower = np.percentile(scores, 5, axis=0) * 100
            upper = np.percentile(scores, 95, axis=0) * 100
            for value, median, p5, p95 in zip(values, medians, lower, upper):
                writer.writerow([parameter, value, median, p5, p95])


def combine_RO_outs(
    candidate_dicts: list[dict[Any, Any]],
    metric_dicts: list[dict[Any, Any]],
    types: list[str],
    params_of_interest: list[str] | None = None,
) -> tuple[dict[int, dict[Any, Any]], dict[int, dict[Any, Any]]]:
    if not (len(candidate_dicts) == len(metric_dicts) == len(types)):
        raise ValueError("Each model type must have one candidate and metric output.")
    combined_candidates: dict[Any, Any] = {}
    combined_metrics: dict[Any, Any] = {}
    next_index = 0

    for candidates, model_type, metrics in zip(candidate_dicts, types, metric_dicts):
        if 0 not in candidates or 0 not in metrics:
            raise ValueError(f"{model_type} optimization output has no outer key 0.")
        if params_of_interest is None:
            if not candidates[0]:
                raise ValueError(f"{model_type} optimization output has no candidates.")
            selected_params = list(next(iter(candidates[0].values())))
        else:
            selected_params = params_of_interest
        for key, candidate in candidates[0].items():
            if key not in metrics[0]:
                raise ValueError(f"{model_type} metrics are missing candidate {key}.")
            filtered = {
                name: value
                for name, value in candidate.items()
                if name in selected_params
            }
            combined_candidates[next_index] = {**filtered, "type": model_type}
            combined_metrics[next_index] = metrics[0][key]
            next_index += 1

    return {0: combined_candidates}, {0: combined_metrics}


def load_model_outputs(
    opt_dir: Path, model_name: str
) -> tuple[dict[str, Any], dict[Any, Any], dict[Any, Any]]:
    model_dir = opt_dir / model_name
    args_path = model_dir / "RO_args.p"
    if not args_path.is_file():
        raise FileNotFoundError(
            f"Missing optimization arguments for {model_name}: {args_path}. "
            "Run model_opt.py first."
        )
    model_args = se.PickleLoad(str(args_path))
    result_path = Path(model_args["pn"]) / "Out.p"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Missing optimization results for {model_name}: {result_path}. "
            "Run model_opt.py first."
        )
    opt_dict_path = Path(model_args["pn"]) / "opt_dict.p"
    if not opt_dict_path.is_file():
        raise FileNotFoundError(
            f"Missing optimization dictionary for {model_name}: {opt_dict_path}."
        )
    opt_dict = se.PickleLoad(str(opt_dict_path))
    candidates, metrics = se.PickleLoad(str(result_path))
    return opt_dict, candidates, metrics


def build_comparisons(
    experiment_dir: Path,
    has_input_signals: bool,
    boots: int,
    top: int,
    seed: int | None,
) -> list[Path]:
    opt_dir = experiment_dir / "opt"
    tables_dir = opt_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    dual_prefix = "Dual_" if has_input_signals else ""
    model_types = ["Global Att.", "Sing. Dense", "Recurrent", "CPM"]
    model_names = [
        dual_prefix + "Ultra_" + suffix
        for suffix in ["Att", "Single", "Recurrent", "CPM"]
    ]
    rng = np.random.default_rng(seed)
    generated: list[Path] = []

    attention_opt, attention_candidates, attention_metrics = load_model_outputs(
        opt_dir, model_names[0]
    )
    attention_clean = dict(attention_opt)
    attention_clean.update(
        {
            "PR_layer": [["PerDim", "Diagonal", "Regular", "Cross"], "cat"],
            "activation_f": [["None", "Asinh", "Asinh_L"], "cat"],
        }
    )
    if has_input_signals:
        attention_parameters = [
            "batchsize",
            "A_kE_ck_base",
            "B_kE_ck_base",
            "A_P_ck",
            "B_P_ck",
            "PR_layer",
        ]
        universal_parameters = [
            "A_kE_ck_base",
            "B_kE_ck_base",
            "late_comb",
        ]
    else:
        attention_parameters = [
            "batchsize",
            "kE_ck_base",
            "P_ck",
            "PR_layer",
        ]
        universal_parameters = ["kE_ck_base"]

    attention_ranks = opt_dict_topcans(
        attention_opt,
        attention_metrics,
        attention_candidates,
        boots=boots,
        params_of_interest=attention_parameters,
        top=top,
        rng=rng,
    )
    attention_table = tables_dir / f"{dual_prefix}Ultra_Att.csv"
    opt_dict_ranks_table(attention_ranks, attention_clean, attention_table)
    generated.append(attention_table)

    recurrent_opt, recurrent_candidates, recurrent_metrics = load_model_outputs(
        opt_dir, model_names[2]
    )
    recurrent_clean = dict(recurrent_opt)
    recurrent_clean.update(
        {
            "PR_layer": [["GRU", "LSTM"], "cat"],
            "activations": [["ReLU", "LeakyReLU"], "cat"],
            "activation_f": [["None", "Asinh", "Asinh_L"], "cat"],
        }
    )
    recurrent_parameters = (
        ["batchsize", "PR_layer", "PR_num_layers", "PR_bi"]
    )
    recurrent_ranks = opt_dict_topcans(
        recurrent_opt,
        recurrent_metrics,
        recurrent_candidates,
        boots=boots,
        params_of_interest=recurrent_parameters,
        top=top,
        rng=rng,
    )
    recurrent_table = tables_dir / f"{dual_prefix}Ultra_Recurrent.csv"
    opt_dict_ranks_table(recurrent_ranks, recurrent_clean, recurrent_table)
    generated.append(recurrent_table)

    all_candidates: list[dict[Any, Any]] = []
    all_metrics: list[dict[Any, Any]] = []
    for model_name in model_names:
        _, candidates, metrics = load_model_outputs(opt_dir, model_name)
        all_candidates.append(candidates)
        all_metrics.append(metrics)

    combined_candidates, combined_metrics = combine_RO_outs(
        all_candidates,
        all_metrics,
        model_types,
        params_of_interest=universal_parameters,
    )
    universal_opt = {
        parameter: attention_opt[parameter]
        for parameter in universal_parameters
    }
    universal_opt["type"] = [model_types, "cat"]
    universal_ranks = opt_dict_topcans(
        universal_opt,
        combined_metrics,
        combined_candidates,
        boots=boots,
        params_of_interest=None,
        top=top,
        rng=rng,
    )
    universal_table = tables_dir / f"{dual_prefix}Ultra_Universal.csv"
    opt_dict_ranks_table(universal_ranks, universal_opt, universal_table)
    generated.append(universal_table)
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Ultra model search results for one experiment and save "
            "bootstrap rank tables under experiments/<name>/opt/tables/."
        )
    )
    parser.add_argument(
        "experiment_file",
        nargs="?",
        type=Path,
        default=DEFAULT_EXPERIMENT_FILE,
        help=f"Experiment JSON file (default: {DEFAULT_EXPERIMENT_FILE})",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help=f"Parent directory for experiment data (default: {DEFAULT_EXPERIMENT_ROOT})",
    )
    parser.add_argument(
        "--boots",
        type=int,
        default=500,
        help="Number of bootstrap samples per table (default: 500).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top sampled candidates to compare (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible bootstrap tables.",
    )
    args = parser.parse_args(argv)
    if args.boots <= 0 or args.top <= 0:
        parser.error("--boots and --top must be positive.")

    try:
        experiment = read_experiment(args.experiment_file)
        experiment_dir = (
            args.experiment_root.resolve() / experiment["experiment_name"]
        )
        if not experiment_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment directory not found: {experiment_dir}. "
                "Run model_opt.py first or pass --experiment-root."
            )
        input_signals = experiment.get("input_signals", [])
        if input_signals is None:
            input_signals = []
        if not isinstance(input_signals, list):
            raise ValueError("input_signals must be a list in the experiment JSON.")
        print(
            f"Comparing model optimization results for "
            f"{experiment['experiment_name']!r}.",
            flush=True,
        )
        tables = build_comparisons(
            experiment_dir,
            bool(input_signals),
            args.boots,
            args.top,
            args.seed,
        )
        for table in tables:
            print(f"Saved comparison table: {table}", flush=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"model_ultra_comps: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
