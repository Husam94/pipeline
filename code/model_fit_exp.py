#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from scipy.stats import betaprime

import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex
import poseigen_binmeths.bin_utils as bu
from model_fit_experiments import bio_xps_functions as xpfus

#------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_FILE = PROJECT_ROOT / "data" / "experiments.json"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "experiments"
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model-fitting experiments for one experiment definition."
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
        "--data-path",
        type=Path,
        help="Override the experiment directory (supports existing custom layouts).",
    )
    parser.add_argument(
        "--targx",
        required=True,
        help="Target key used to load target-specific pickles, for example K562.",
    )
    parser.add_argument(
        "--oracle-rewrite",
        action="store_true",
        help="Recompute oracle predictions even if a cached file already exists.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of workers passed to each xp child script.",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="Forwarded to child scripts to skip fitting and only run downstream summaries.",
    )
    return parser.parse_args()


def run_xp_script(data_path, script="bio_xp2.py", n_workers=1, skip_fit=False, resource_comp=False):
    script_path = Path(__file__).resolve().parent / "model_fit_experiments" / script
    cmd = [
        sys.executable,
        str(script_path),
        "--data-path",
        data_path,
        "--n_workers",
        str(n_workers),
    ]

    if skip_fit:
        cmd.append("--skip_fit")
    if resource_comp:
        cmd.append("--resource_comp")

    env = os.environ.copy()
    env["BIO_XP_DATA_PATH"] = data_path

    subprocess.run(cmd, check=True, cwd=data_path, env=env)


def main():
    args = parse_args()

    with args.experiment_file.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    if not isinstance(experiment, dict):
        raise ValueError(
            f"{args.experiment_file} must contain one JSON experiment object."
        )
    experiment_name = experiment.get("experiment_name")
    if (
        not isinstance(experiment_name, str)
        or not EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name)
    ):
        raise ValueError("The experiment JSON contains an invalid experiment_name.")
    if args.n_workers <= 0:
        raise ValueError("--n-workers must be positive.")
    if args.data_path is None:
        data_path = str(
            (args.experiment_root.resolve() / experiment_name).resolve()
        )
    else:
        data_path = str(args.data_path.resolve())
    if not data_path.endswith(os.sep):
        data_path = data_path + os.sep

    targx = args.targx
    oracle_rewrite = args.oracle_rewrite

    os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)

    #====================================

    datafolder = se.NewFolder('data_ml')
    tri_tpacks, bin_sizes, split_transaug = xpfus.load_fit_data(data_path)
    if len(tri_tpacks) == 4:
        d_x, d_y_ms, d_s, d_b = tri_tpacks
    else:
        d_x1, d_y_ms, d_s, d_b, d_x2 = tri_tpacks

    #------------------------------------

    splix = split_transaug
    
    subj = 2
    split_eval = np.hstack([splix[subj], splix[subj] + np.sum([len(x) for x in splix])])

    d_y_full = np.vstack([d_y_ms, d_y_ms])
    d_y_eval = d_y_full[split_eval]

    d_b_full = np.vstack([d_b, d_b])
    d_b_eval = d_b_full[split_eval]

    d_w_eval = bu.BinWeights(d_b_eval, bin_sizes=bin_sizes, mode="recip")

    #------------------------------------

    xpsfolder = se.NewFolder("xps")
    optfolder = se.NewFolder("opt")

    if len(tri_tpacks) == 4: modx_names_formal = ['DeepSTARR', 'Ultra_CPM', 'Oyster']
    else: modx_names_formal = ['BiChrom', 'Dual_Ultra_CPM', 'Dual_Oyster']

    modx_configs = {}
    for modx in modx_names_formal:
        pn_modx = se.NewFolder(optfolder + modx)
        pn_modx_RO_args = pn_modx + "RO_args"
        modx_RO_args = se.PickleLoad(pn_modx_RO_args)
        best_config = se.PickleLoad(modx_RO_args["pn"] + "best_config")
        modx_configs[modx] = best_config
    se.PickleDump(modx_configs, xpsfolder + "modx_configs")

    print(list(modx_configs))


    #===================================
    modx_names_full = []
    for cof_name in modx_names_formal: 
        argoz = se.PickleLoad(f'{optfolder}{cof_name}/RO_args')
        RO_folo = argoz['pn']
        best_config_idx = se.PickleLoad(RO_folo + 'best_config_idx')
        match = re.search(r'RO_(\d+)$', Path(RO_folo).name)
        if match: iter = match.group(1)
        else: raise ValueError(f"Could not extract iteration number from RO_folo: {RO_folo}")
        modx_names_full.append(f'{cof_name}_{iter}_{best_config_idx}')
    #===================================

    """ORACLE"""

    d_s_full = np.vstack([d_s, d_s])
    d_s_eval = d_s_full[split_eval]

    pn_speclog_params = datafolder + f"{targx}_speclog_params.p"
    if os.path.exists(pn_speclog_params): 
        speclog_paramo = True
        speclog_params = se.PickleLoad(datafolder + f"{targx}_speclog_params")
    else: 
        vst_transform = se.PickleLoad(datafolder + f'{targx}_vst_transform')
        speclog_paramo = False

    targx_max = se.PickleLoad(datafolder + f"{targx}_max")
    targx_shift = se.PickleLoad(datafolder + f"{targx}_shift")

    pn_oracle_preds_shifted = xpsfolder + f"oracle_preds_shifted_{targx}.p"

    if (os.path.exists(pn_oracle_preds_shifted) is False) or oracle_rewrite:

        oracle_reps = 1
        oracle_samples = 15

        params = d_s_eval[:, :, 0, 0]
        alpha1 = params[:, 0]
        theta1 = params[:, 1]
        alpha2 = params[:, 2]
        theta2 = params[:, 3]
        scale = theta1 / theta2

        rng = np.random.default_rng(12345)

        draws = betaprime.rvs(
            a=alpha1[None, :, None],
            b=alpha2[None, :, None],
            size=(oracle_reps, params.shape[0], oracle_samples),
            random_state=rng,
        )

        scaled_draws = draws * scale[None, :, None]

        oracle_preds_novst_reps = mex.half_sample_mode(scaled_draws, axis=2)[..., None]

        if speclog_paramo: oracle_preds_reps = se.speclog(oracle_preds_novst_reps, *speclog_params)
        else: oracle_preds_reps = vst_transform(oracle_preds_novst_reps)

        oracle_preds_noshift = np.mean(oracle_preds_reps, axis=0)
        oracle_preds = (oracle_preds_noshift - targx_shift) / targx_max

        y = d_y_eval.reshape(-1).astype(float)
        x = oracle_preds.reshape(-1).astype(float)
        w = d_w_eval.reshape(-1).astype(float)

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x, y, w = x[mask], y[mask], w[mask]

        X = np.column_stack([x, np.ones_like(x)])
        sw = np.sqrt(w)

        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        m, b = beta

        oracle_preds_shifted = (m * oracle_preds.reshape(-1) + b).reshape(oracle_preds.shape)

        def weighted_rmse(y_true, y_pred, ww):
            r = y_true - y_pred
            return np.sqrt(np.sum(ww * r * r) / np.sum(ww))

        rmse_before = weighted_rmse(y, x, w)
        rmse_after = weighted_rmse(y, m * x + b, w)

        print(f"m = {m:.6f}, b = {b:.6f}")
        print(f"Weighted RMSE before: {rmse_before:.6f}")
        print(f"Weighted RMSE after : {rmse_after:.6f}")

        se.PickleDump(oracle_preds_shifted, pn_oracle_preds_shifted)

    oracle_preds_shifted = se.PickleLoad(pn_oracle_preds_shifted)

    #====================================
    """xp2: Loss Functions"""

    xp2_trial = 0
    pn_xp2 = se.NewFolder(xpsfolder + "xp2_" + str(xp2_trial))

    xp2_variables = ["Loss Function"]
    lossos = ["MSE", "MSE-C", "MSE-R"]

    xp2_combs = [[lo] for lo in lossos]
    xp2_combs = [[ic, c] for ic, c in enumerate(xp2_combs)]

    xp2_dict = {
        "trial": xp2_trial,
        "pn": pn_xp2,
        "variables": xp2_variables,
        "loss_funcs": lossos,
        "combs": xp2_combs,
        "EUS_prop": None,
        "EUS_mode": "recip",
        "modx_names": list(modx_configs),
        "modx_names_full": modx_names_full, 
        "TCR_repeats": 15,
    }

    se.PickleDump(xp2_dict, xpsfolder + "xp2_dict")

    #====================================
    """xp3: Undersamp Props"""

    xp3_trial = 0
    pn_xp3 = se.NewFolder(xpsfolder + "xp3_" + str(xp3_trial))

    xp3_variables = ["Prop"]
    us_props = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]

    xp3_combs = [[lo] for lo in us_props]
    xp3_combs = [[ic, c] for ic, c in enumerate(xp3_combs)]

    xp3_dict = {
        "trial": xp3_trial,
        "pn": pn_xp3,
        "variables": xp3_variables,
        "us_props": us_props,
        "combs": xp3_combs,
        "modx_names": list(modx_configs),
        "modx_names_full": modx_names_full, 
        "TCR_repeats": 15,
    }

    se.PickleDump(xp3_dict, xpsfolder + "xp3_dict")

    #====================================
    """xp3B: Sample with replacement"""

    xp3B_trial = 0
    pn_xp3B = se.NewFolder(xpsfolder + "xp3B_" + str(xp3B_trial))

    xp3B_variables = ["Prop"]
    us_props = [5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]

    xp3B_combs = [[lo] for lo in us_props]
    xp3B_combs = [[ic, c] for ic, c in enumerate(xp3B_combs)]

    xp3B_dict = {
        "trial": xp3B_trial,
        "pn": pn_xp3B,
        "variables": xp3B_variables,
        "us_props": us_props,
        "combs": xp3B_combs,
        "modx_names": list(modx_configs),
        "modx_names_full": modx_names_full, 
        "TCR_repeats": 15,
    }

    se.PickleDump(xp3B_dict, xpsfolder + "xp3B_dict")

    #====================================
    """xp4: Undersamp Props"""

    xp4_trial = 0
    pn_xp4 = se.NewFolder(xpsfolder + "xp4_" + str(xp4_trial))

    xp4_variables = ["Prop"]
    us_props = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]

    xp4_combs = [[lo] for lo in us_props]
    xp4_combs = [[ic, c] for ic, c in enumerate(xp4_combs)]

    xp4_dict = {
        "trial": xp4_trial,
        "pn": pn_xp4,
        "variables": xp4_variables,
        "us_props": us_props,
        "combs": xp4_combs,
        "modx_names": list(modx_configs),
        "modx_names_full": modx_names_full, 
        "TCR_repeats": 15,
    }

    se.PickleDump(xp4_dict, xpsfolder + "xp4_dict")

    #====================================

    for xp in ['2', '3', '3B']: #, "3", "3B"]:
        resource_comp = xp in ["3", "3B"]

        run_xp_script(
            data_path=data_path,
            script=f"bio_xp{xp}.py",
            n_workers=args.n_workers,
            skip_fit=args.skip_fit,
            resource_comp=resource_comp,
        )

if __name__ == "__main__":
    main()