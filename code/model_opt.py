#!/usr/bin/env python3
"""Optimize biological models using one experiment's packs and splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.optim import Adam
import torch.nn as nn

import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex
import poseigen_compass as co
import poseigen_trident.utils as tu
import poseigen_trident.preass as preass
import poseigen_trident.deepstarr as deepstarr
import poseigen_trident.othermodels as othmods
import poseigen_oyster.model as oyster_models


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


def load_experiment_data(
    experiment_dir: Path,
) -> tuple[list[Any], Any, Any]:
    paths = {
        "packs": experiment_dir / "packs" / "tri_tpacks.p",
        "bin_sizes": experiment_dir / "packs" / "output_bin_sizes.p",
        "split": experiment_dir / "split" / "split_transaug.p",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment input data: "
            + ", ".join(str(path) for path in missing)
            + ". Run processing_exp_data.py first."
        )

    tri_tpacks = se.PickleLoad(str(paths["packs"]))
    bin_sizes = se.PickleLoad(str(paths["bin_sizes"]))
    split_transaug = se.PickleLoad(str(paths["split"]))
    if not isinstance(tri_tpacks, list) or len(tri_tpacks) not in (4, 5):
        raise ValueError(
            f"{paths['packs']} must contain four packs without input signals "
            "or five packs with input signals."
        )
    return tri_tpacks, bin_sizes, split_transaug


def single_oyster_search_space(
    opt_base_dict: dict[str, Any], input_shape: tuple[int, ...]
) -> dict[str, Any]:
    return {
        **opt_base_dict,
        "dim_i": [[input_shape], "cat"],
        "dim_f": [[(1, 1, 1)], "cat"],
        "DS": [[False, True], "cat"],
        "DS_wwp_func": [[nn.MaxPool2d, nn.AvgPool2d], "cat"],
        "DS_act": [[False, True], "cat"],
        "kE_k": [[11, 13, 15], "cat"],
        "kE_cf_m": [[50, 100, 150], "cat"],
        "kE_cf_ns": [[1, 1.1, 1.2], "cat"],
        "kE_ck_base": [[None, 5, 7, 9], "cat"],
        "kE_OneByOne": [[True, False], "cat"],
        "kE_ck_grouped": [[True, False], "cat"],
        "P_ck": [[None, 10, 25, 50, 100], "cat"],
        "P_center": [[True, False], "cat"],
        "O_mods": [[0, 1, 2], "cat"],
        "O_mods_ns": [[0.3, 0.5], "cat"],
        "O_cf_pu_m": [[None, 1, 1.5, 2], "cat"],
        "O_cf_ns": [[1, 1.1, 1.2], "cat"],
        "O_ck_base": [[None, 5, 7, 9], "cat"],
        "O_pool_k2s": [[0, 0.5], "cat"],
        "O_pool_func": [[nn.MaxPool2d, nn.AvgPool2d], "cat"],
        "O_actb4pool": [[True, False], "cat"],
        "O_dropout": [[0, 0.1, 0.2], "cat"],
        "O_bias": [[False, True], "cat"],
        "activations": [[nn.ReLU(), nn.LeakyReLU()], "cat"],
        "activation_f": [[None, tu.AsinhAct(), tu.AsinhAct_L()], "cat"],
        "batchnorm": [[None, "before"], "cat"],
    }


def dual_oyster_search_space(
    opt_base_dict: dict[str, Any],
    dna_input_shape: tuple[int, ...],
    signal_input_shape: tuple[int, ...],
) -> dict[str, Any]:
    search_space = {
        **opt_base_dict,
        "dim_f": [[(1, 1, 1)], "cat"],
        "P_center": [[True, False], "cat"],
        "DS_wwp_func": [[nn.MaxPool2d], "cat"],
        "DS_act": [[False], "cat"],
        "O_mods": [[0, 1, 2], "cat"],
        "O_mods_ns": [[0.3, 0.5], "cat"],
        "O_cf_pu_m": [[None, 1, 1.5, 2], "cat"],
        "O_cf_ns": [[1, 1.1, 1.2], "cat"],
        "O_ck_base": [[None, 5, 7, 9], "cat"],
        "O_pool_k2s": [[0, 0.5], "cat"],
        "O_pool_func": [[nn.MaxPool2d, nn.AvgPool2d], "cat"],
        "O_actb4pool": [[True, False], "cat"],
        "O_dropout": [[0, 0.1, 0.2], "cat"],
        "O_bias": [[False, True], "cat"],
        "activations": [[nn.ReLU(), nn.LeakyReLU()], "cat"],
        "activation_f": [[None, tu.AsinhAct(), tu.AsinhAct_L()], "cat"],
        "batchnorm": [[None, "before"], "cat"],
    }
    for prefix, input_shape, kernel_sizes, filter_multipliers in (
        ("A_", dna_input_shape, [12, 20], [50, 150]),
        ("B_", signal_input_shape, [5, 10, 15], [10, 20, 30]),
    ):
        search_space.update(
            {
                f"{prefix}dim_i": [[input_shape], "cat"],
                f"{prefix}DS": [[True, False], "cat"],
                f"{prefix}kE_k": [kernel_sizes, "cat"],
                f"{prefix}kE_cf_m": [filter_multipliers, "cat"],
                f"{prefix}kE_cf_ns": [[1, 1.1, 1.2], "cat"],
                f"{prefix}kE_ck_base": [[None, 4, 5, 6, 7, 8], "cat"],
                f"{prefix}kE_OneByOne": [[True, False], "cat"],
                f"{prefix}kE_ck_grouped": [[True, False], "cat"],
                f"{prefix}P_ck": [[None, 6, 20], "cat"],
            }
        )
    return search_space


def TopOfTop(pn_RO, TCR_args, modo):
    topcan = co.TopCan(
        pn_RO,
        num_top=5,
        reduce_func=np.min,
        smallest=True,
        perround=False,
    )
    topcan_cd = co.TopCanDict(pn_RO, topcan, comb_rounds=True)
    topcan_idxs = list(topcan_cd.keys())

    for epo_samp in [1.0]:
        topo = f"Top_{epo_samp}"
        pn_RO_top = se.NewFolder(pn_RO + topo)
        top_dict = {
            "allconfigs": [topo],
            "pn": pn_RO,
            "combs": [
                [index, [candidate]]
                for index, candidate in enumerate(topcan_cd.keys())
            ],
            "TCR_repeats": TCR_args["repeats"],
        }
        se.PickleDump(top_dict, pn_RO_top + "top_dict")

        for candidate_name, candidate in topcan_cd.items():
            print(f"CANDIDATE: {candidate_name}", flush=True)
            candidate_folder = se.NewFolder(pn_RO_top + candidate_name)
            repeated_args = {**TCR_args, "pathname": candidate_folder}
            repeated_args["trainer_args"]["EUS_prop"] = epo_samp
            print(
                f"Using EUS_prop={repeated_args['trainer_args']['EUS_prop']}",
                flush=True,
            )
            tu.TridentCanRepeater(modo, candidate, data=None, **repeated_args)
    return topcan_idxs


def run_optimization(
    experiment_dir: Path,
    tri_tpacks: list[Any],
    bin_sizes: Any,
    split_transaug: Any,
    budget: int,
) -> None:
    opt_dir = experiment_dir / "opt"
    optfolder = se.NewFolder(str(opt_dir))
    if len(tri_tpacks) == 4:
        d_x, d_y_ms, d_s, d_b = tri_tpacks
    else:
        d_x1, d_y_ms, d_s, d_b, d_x2 = tri_tpacks

    mse_mode = [mex.AError, {"expo": 2}]
    fit_args = {"MSE_mode": mse_mode}
    flips = (
        [[-2, -1]]
        if len(tri_tpacks) == 4
        else [[-2, -1], [-2, -1]]
    )
    fit_args["TT_base"] = {
        "loss_mode": mse_mode,
        "metrics_mode": mse_mode,
        "bw_mode": "recip",
        "EUS_prop": 0.05,
        "EUS_mode": "recip",
        "smallest": True,
        "batchsize": 512,
        "batchsize_infer": 2024,
        "opt": Adam,
        "maxepochs": 100,
        "patience": 10,
        "pickup": True,
        "duds": None,
        "statusprints": 1,
        "flips": flips,
    }
    fit_args["TCS_base"] = {
        "trainer": tu.TridentTrainer_BW,
        "smallest": None,
        "get_predictions": True,
        "pred_rewrite": False,
        "add_pred_args": {"batchsize": 2024},
        "score_on": 1,
        "score_predictions": False,
    }
    fit_args["TCR_base"] = {
        "Splits": None,
        "pickup": False,
        "savemodels": True,
        "returnmodel": True,
    }
    se.PickleDump(fit_args, optfolder + "fit_args")

    opt_base_dict = {
        "batchsize": [[128, 256, 512], "cat"],
        "learningrate": [
            se.GeomNumSpacing(0.0001, 0.01, 15, plier=1.2),
            "cat",
        ],
    }
    ultra_dict_general = {
        **opt_base_dict,
        "kE_OneByOne": [[True, False], "cat"],
        "PR_pool_func": [[nn.MaxPool2d, nn.AvgPool2d], "cat"],
        "PR_cf_m": [[1.5, 2], "cat"],
        "O_mods": [[1, 2, 3], "cat"],
        "O_cf_pu_m": [[1, 1.5, 2], "cat"],
        "O_dropout": [[0, 0.1, 0.2], "cat"],
        "activations": [[nn.ReLU(), nn.LeakyReLU()], "cat"],
        "activation_f": [[None, tu.AsinhAct(), tu.AsinhAct_L()], "cat"],
        "batchnorm": [[None, "before"], "cat"],
    }
    recurrent_dict = {
        "PR_layer": [[nn.GRU, nn.LSTM], "cat"],
        "PR_num_layers": [[1, 2, 3], "cat"],
        "PR_bi": [[True, False], "cat"],
    }
    attention_dict = {
        "PR_layer": [[
            tu.PerDimAttentionPooling,
            tu.DiagonalAttentionPooling,
            tu.RegularAttentionPooling,
            tu.CrossAttentionPooling,
        ], "cat"],
    }
    single_dict = {"PR_layer": [[nn.Conv2d], "cat"]}

    if len(tri_tpacks) == 4:
        ultra_dict_shared = {
            **ultra_dict_general,
            "dim_i": [[d_x.shape[1:]], "cat"],
            "dim_f": [[(1, 1, 1)], "cat"],
            "kE_k": [[11, 13, 15], "cat"],
            "kE_cf_m": [[100, 150], "cat"],
            "kE_ck_base": [[None, 7, 9], "cat"],
        }
        position_dict = {"P_ck": [[10, 25, 50, 100], "cat"]}
        cpm_dict = {
            "PR_layer": [[None], "cat"],
            "P_ck": [[None], "cat"],
            "PR_ck_base": [[5, 7, 9], "cat"],
            "PR_pk_base": [[0, 2, 4], "cat"],
            "PR_pk_s": [[2, 3, 4], "cat"],
        }
        model_names = [
            "DeepSTARR",
            "Ultra_Att",
            "Ultra_Single",
            "Ultra_Recurrent",
            "Ultra_CPM",
            "Oyster",
        ]
        top_model_names = {"DeepSTARR", "Ultra_CPM", "Oyster"}
    else:
        ultra_dict_shared = {
            **ultra_dict_general,
            "late_comb": [[False, True], "cat"],
            "A_dim_i": [[d_x1.shape[1:]], "cat"],
            "B_dim_i": [[d_x2.shape[1:]], "cat"],
            "A_kE_k": [[11, 13, 15], "cat"],
            "A_kE_cf_m": [[100, 150, 200, 250], "cat"],
            "A_kE_ck_base": [[None, 7, 9], "cat"],
            "B_kE_k": [[5, 10, 15], "cat"],
            "B_kE_cf_m": [[10, 20, 30], "cat"],
            "B_kE_ck_base": [[None, 5, 7, 9], "cat"],
        }
        position_dict = {
            "A_P_ck": [[10, 25, 50, 100], "cat"],
            "B_P_ck": [[5, 10], "cat"],
        }
        cpm_dict = {
            "PR_layer": [[None], "cat"],
            "A_P_ck": [[None], "cat"],
            "A_PR_ck_base": [[5, 7, 9], "cat"],
            "A_PR_pk_base": [[0, 2, 4], "cat"],
            "A_PR_pk_s": [[2, 3, 4], "cat"],
            "B_P_ck": [[None], "cat"],
            "B_PR_ck_base": [[3, 4], "cat"],
            "B_PR_pk_base": [[0, 2], "cat"],
            "B_PR_pk_s": [[2, 3, 4], "cat"],
        }
        model_names = [
            "BiChrom",
            "Dual_Ultra_Att",
            "Dual_Ultra_Single",
            "Dual_Ultra_Recurrent",
            "Dual_Ultra_CPM",
            "Dual_Oyster",
        ]
        top_model_names = {"BiChrom", "Dual_Ultra_CPM", "Dual_Oyster"}

    for model_name in model_names:
        if len(tri_tpacks) == 4:
            if model_name == "DeepSTARR":
                modo = deepstarr.DeepSTARR
                init_mode = [deepstarr.Reset_DeepSTARR, {}]
            elif model_name == "Oyster":
                modo = oyster_models.Oyster
                init_mode = [oyster_models.Reset_Oyster, {}]
            else:
                modo = preass.Ultra
                init_mode = [preass.Reset_Ultra, {}]
        else:
            if model_name == "BiChrom":
                modo = othmods.BiChrom
                init_mode = [othmods.Reset_BiChrom, {}]
            elif model_name == "Dual_Oyster":
                modo = oyster_models.DualOyster
                init_mode = [oyster_models.Reset_DualOyster, {}]
            else:
                modo = preass.Dual_Ultra
                init_mode = [preass.Reset_Dual_Ultra, {}]

        print(f"Starting optimization for {model_name}.", flush=True)
        model_folder = se.NewFolder(optfolder + model_name)
        model_args = {"folder": model_folder, "iter": 0}
        model_args["pn"] = se.NewFolder(model_folder + "RO_0")
        se.PickleDump(model_args, model_folder + "RO_args")

        if model_name == "Oyster":
            opt_dict = single_oyster_search_space(
                opt_base_dict, tuple(d_x.shape[1:])
            )
        elif model_name == "Dual_Oyster":
            opt_dict = dual_oyster_search_space(
                opt_base_dict,
                tuple(d_x1.shape[1:]),
                tuple(d_x2.shape[1:]),
            )
        elif "CPM" in model_name:
            opt_dict = {**ultra_dict_shared, **cpm_dict}
        elif "Att" in model_name:
            opt_dict = {**ultra_dict_shared, **position_dict, **attention_dict}
        elif "Single" in model_name:
            opt_dict = {**ultra_dict_shared, **position_dict, **single_dict}
        elif "Recurrent" in model_name:
            opt_dict = {**ultra_dict_shared, **position_dict, **recurrent_dict}
        elif model_name == "DeepSTARR":
            opt_dict = {
                **opt_base_dict,
                "dim_i": [[d_x.shape[1:]], "cat"],
                "dim_o": [[(1, 1, 1)], "cat"],
                "ck_i": [[7, 12, 19], "cat"],
                "cf_i": [[64, 128, 256], "cat"],
                "num_convs": [[4, 5, 6], "cat"],
                "num_dense": [[1, 2, 3], "cat"],
                "activation_f": [[None, tu.AsinhAct(), tu.AsinhAct_L()], "cat"],
            }
        elif model_name == "BiChrom":
            opt_dict = {
                **opt_base_dict,
                "A_dim_i": [[d_x1.shape[1:]], "cat"],
                "B_dim_i": [[d_x2.shape[1:]], "cat"],
                "A_kE_k": [[24], "cat"],
                "A_cf": [[16, 32, 64, 128, 256], "cat"],
                "A_rec_f": [[32], "cat"],
                "A_dense_num": [[2, 3, 4, 5], "cat"],
                "A_dense_f": [[128, 256, 512], "cat"],
                "B_cf": [[15, 25, 30], "cat"],
                "B_rec_f": [[5, 25, 50], "cat"],
                "dropout": [[0.5, 0.25], "cat"],
                "activations": [[nn.ReLU(), nn.LeakyReLU()], "cat"],
                "activation_f": [[None, tu.AsinhAct(), tu.AsinhAct_L()], "cat"],
            }
        else:
            raise ValueError(f"No hyperparameter space defined for {model_name}.")

        se.PickleDump(opt_dict, model_args["pn"] + "opt_dict")
        trainer_args = {
            "inps": [d_x] if len(tri_tpacks) == 4 else [d_x1, d_x2],
            "out": d_y_ms,
            "out_bind": d_b,
            "bin_sizes": bin_sizes,
            "Split": split_transaug,
            "mod_init_mode": init_mode,
            "duds": None,
            **fit_args["TT_base"],
        }
        scorer_args = {"trainer_args": trainer_args, **fit_args["TCS_base"]}
        random_opt_args = {
            "algo": modo,
            "VarDict": opt_dict,
            "pathname": model_args["pn"],
            "CS_mode": [tu.TridentCanScorer, scorer_args],
            "data": None,
            "Splits": None,
            "budget": budget,
            "repeats": 1,
            "pickup": True,
            "savemodels": True,
            "parallel": False,
        }
        co.RandomOpt(**random_opt_args)

        if model_name not in top_model_names:
            print(f"Completed optimization for {model_name}.", flush=True)
            continue

        repeated_args = {
            **scorer_args,
            **fit_args["TCR_base"],
            "repeats": 10,
        }
        top_candidates = TopOfTop(model_args["pn"], repeated_args, modo)
        print(f"Top candidates for {model_name}: {top_candidates}", flush=True)

        top_folder = model_args["pn"] + "Top_1.0/"
        performance_means = []
        for candidate in top_candidates:
            metrics_path = top_folder + candidate + "/Mets"
            metrics = se.PickleLoad(metrics_path)
            metric_values = [value for value in metrics[0].values()]
            print(metric_values, flush=True)
            performance_means.append(np.mean(metric_values))

        best_candidate = top_candidates[int(np.argmin(performance_means))]
        best_config_index = int(best_candidate[2:])
        se.PickleDump(best_config_index, model_args["pn"] + "best_config_idx")
        candidate_dict, _ = se.PickleLoad(model_args["pn"] + "Out")
        best_config = candidate_dict[0][best_config_index]
        se.PickleDump(best_config, model_args["pn"] + "best_config")

        best_folder = se.NewFolder(top_folder + f"0_{best_config_index}")
        best_args = {**repeated_args, "pathname": best_folder}
        best_args["trainer_args"]["EUS_prop"] = 1.0
        best_args["repeats"] = 15
        tu.TridentCanRepeater(modo, best_config, data=None, **best_args)
        print(f"Completed top-model refinement for {model_name}.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the model families from bio_opt_all.py using the packs "
            "and splits for one experiment."
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
        "--budget",
        type=int,
        default=300,
        help="Random-search candidate budget per model (default: 300).",
    )
    args = parser.parse_args(argv)
    if args.budget <= 0:
        parser.error("--budget must be positive.")

    try:
        experiment = read_experiment(args.experiment_file)
        experiment_dir = (
            args.experiment_root.resolve() / experiment["experiment_name"]
        )
        if not experiment_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment output directory not found: {experiment_dir}. "
                "Run processing_exp_data.py first or pass --experiment-root."
            )
        tri_tpacks, bin_sizes, split_transaug = load_experiment_data(experiment_dir)
        (experiment_dir / "opt").mkdir(parents=True, exist_ok=True)
        print(
            f"Optimizing experiment {experiment['experiment_name']!r}; "
            f"results will be written to {experiment_dir / 'opt'}.",
            flush=True,
        )
        run_optimization(
            experiment_dir,
            tri_tpacks,
            bin_sizes,
            split_transaug,
            args.budget,
        )
        print(
            f"Completed model optimization for {experiment['experiment_name']!r}.",
            flush=True,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"model_opt: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
