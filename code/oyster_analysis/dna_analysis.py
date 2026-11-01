#!/usr/bin/env python3
"""Score DNA motifs against an Oyster ensemble's DNA branch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import poseigen_chisel as chis
import poseigen_oyster.analysis as interp
import poseigen_seaside.basics as se

from interp_ensemble import (
    DEFAULT_EXPERIMENT_FILE,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_LOSS,
    DEFAULT_XP_RUN,
    read_experiment,
    resolve_fit_folder,
)


def load_ensemble_manifest(ensemble_dir: Path) -> dict[str, Any]:
    path = ensemble_dir / "ensemble_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Weighted ensemble is missing: {path}. Run interp_ensemble.py first."
        )
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict) or manifest.get("model_name") not in {
        "Oyster",
        "Dual_Oyster",
    }:
        raise ValueError(
            f"Unsupported Oyster ensemble manifest: {path}"
        )
    paths = manifest.get("model_paths")
    weights = manifest.get("weights")
    if (
        not isinstance(paths, list)
        or not paths
        or not isinstance(weights, list)
        or len(paths) != len(weights)
    ):
        raise ValueError(f"Invalid weighted ensemble manifest: {path}")
    weights_array = np.asarray(weights, dtype=float)
    if (
        not np.all(np.isfinite(weights_array))
        or np.any(weights_array < 0)
        or not np.isclose(weights_array.sum(), 1.0)
    ):
        raise ValueError(f"Ensemble weights in {path} must be finite and sum to 1.")
    for model_path in paths:
        if not Path(f"{model_path}.pt").is_file():
            raise FileNotFoundError(f"Ensemble checkpoint not found: {model_path}.pt")
    return manifest


def _write_matrix_csv(path: Path, row_names: list[str], values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["motif_kmer", *range(1, values.shape[1] + 1)])
        writer.writerows(
            [name, *row.tolist()] for name, row in zip(row_names, values)
        )


def _score_motifs(
    model_paths: list[str],
    model_weights: np.ndarray,
    model_config: dict[str, Any],
    model_name: str,
    motif_file: Path,
    random_sequences: int,
    batch_size: int,
) -> dict[str, Any]:
    is_dual = model_name == "Dual_Oyster"
    sub_model = "OysterA" if is_dual else None
    config_prefix = "A_" if is_dual else ""
    kmer_length = int(model_config[f"{config_prefix}kE_k"])
    sequence_length = int(model_config[f"{config_prefix}dim_i"][1])
    kmer_positions = sequence_length - kmer_length + 1
    if kmer_length <= 0 or kmer_positions <= 0:
        raise ValueError(
            "Oyster model has invalid DNA dimensions: "
            f"sequence_length={sequence_length}, kE_k={kmer_length}."
        )

    motif_names_unfiltered, motif_matrices = chis.MEME2Motifs(
        str(motif_file),
        name_idx=1,
    )
    if not motif_matrices:
        raise ValueError(f"No motifs were parsed from {motif_file}.")
    forward_names = [
        name.split("_HUMAN")[0] for name in motif_names_unfiltered
    ]
    forward_matrices = [np.asarray(matrix, dtype=float) for matrix in motif_matrices]
    reverse_names = [f"{name}_R" for name in forward_names]
    reverse_matrices = [np.flip(matrix, axis=(0, 1)) for matrix in forward_matrices]
    names = [*forward_names, *reverse_names]
    matrices = [*forward_matrices, *reverse_matrices]
    random_motif = np.full((kmer_length, 4), 0.25)

    kmer_matrices: list[np.ndarray] = []
    kmer_names: list[str] = []
    motif_groups: list[np.ndarray] = []
    for motif_name, motif_matrix in zip(names, matrices):
        shifted_motifs, shifted_names = chis.kPPMFromPPM(
            motif_matrix,
            k=kmer_length,
            name=motif_name,
        )
        start = len(kmer_names)
        kmer_matrices.extend(np.asarray(motif) for motif in shifted_motifs)
        kmer_names.extend(shifted_names)
        motif_groups.append(np.arange(start, len(kmer_names)))

    contribution_args = {
        "sub_model": sub_model,
        "joint": False,
        "random_seqs": random_sequences,
        "unique": False,
        "scoreweight": False,
        "score_mode": "recipeuc",
        "batchsize": batch_size,
        "model_weights": model_weights,
        "dim_kE_length": kmer_positions,
    }
    score_rows: list[np.ndarray] = []
    for index, motif in enumerate(kmer_matrices, start=1):
        score = np.asarray(
            interp.kppm_contrib(
                model_paths,
                motif,
                **contribution_args,
            ),
            dtype=float,
        ).reshape(-1)
        score_rows.append(score)
        if index % 25 == 0 or index == len(kmer_matrices):
            print(f"Scored {index}/{len(kmer_matrices)} motif k-mers.", flush=True)
    if not score_rows:
        raise ValueError("No motif k-mers were generated.")
    if len({len(row) for row in score_rows}) != 1:
        raise ValueError("Motif contribution vectors have inconsistent lengths.")
    kmer_scores = np.stack(score_rows)
    baseline_scores = np.asarray(
        interp.kppm_contrib(
            model_paths,
            random_motif,
            **contribution_args,
        ),
        dtype=float,
    ).reshape(1, -1)
    return {
        "sequence_length": sequence_length,
        "kmer_length": kmer_length,
        "kmer_positions": kmer_positions,
        "sub_model": "A" if is_dual else None,
        "motif_names": names,
        "motif_matrices": matrices,
        "summary_motif_names": forward_names,
        "summary_motif_matrices": forward_matrices,
        "kmer_names": kmer_names,
        "motif_groups": motif_groups,
        "kmer_scores": kmer_scores,
        "baseline_scores": baseline_scores,
    }


def _summarize_motifs(
    scored: dict[str, Any], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scored["kmer_scores"])
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D motif scores, found shape {scores.shape}.")
    if scores.shape[1] == 1:
        expanded_scores = np.repeat(
            scores,
            scored["kmer_positions"],
            axis=1,
        )
    else:
        expanded_scores = interp.contrib_perpos(
            scores,
            config,
            submodel=scored["sub_model"],
        )
    if expanded_scores.shape[1] != scored["kmer_positions"]:
        raise ValueError(
            "Expanded motif score positions do not match the Oyster DNA input "
            f"length ({expanded_scores.shape[1]} vs {scored['kmer_positions']})."
        )

    motif_groups = scored["motif_groups"]
    motif_profiles = interp.sppm_contrib(expanded_scores, motif_groups)
    motif_profiles = interp.sppm_avgrevcomp(motif_profiles)
    motif_profiles_filled = interp.contrib_fill(
        motif_profiles,
        scored["sequence_length"],
        fill_center=False,
    )
    motif_names = scored["summary_motif_names"]
    return np.asarray(motif_profiles), np.asarray(motif_profiles_filled)


def write_motif_outputs(
    scored: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    top_n: int,
    rewrite: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figs"
    figures_dir.mkdir(exist_ok=True)
    motifs_dir = output_dir / "tables"
    motifs_dir.mkdir(exist_ok=True)

    motif_profiles, motif_profiles_filled = _summarize_motifs(scored, config)
    motif_names = scored["summary_motif_names"]
    max_abs = np.asarray(
        [
            profile[np.nanargmax(np.abs(profile))]
            if np.any(np.isfinite(profile))
            else np.nan
            for profile in motif_profiles
        ]
    )
    order = np.argsort(np.nan_to_num(np.abs(max_abs), nan=-np.inf))[::-1]
    profile_indices = order[:top_n]

    kmer_csv = motifs_dir / "kmer_motif_contributions.csv"
    if rewrite or not kmer_csv.is_file():
        _write_matrix_csv(kmer_csv, scored["kmer_names"], scored["kmer_scores"])
        print(f"Saved {kmer_csv}", flush=True)
    else:
        print(f"Reusing existing {kmer_csv}", flush=True)

    summary_csv = motifs_dir / "motif_contributions.csv"
    if rewrite or not summary_csv.is_file():
        with summary_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["motif", "max_abs_contribution"])
            writer.writerows(
                [motif_names[index], max_abs[index]]
                for index in order
            )
        print(f"Saved {summary_csv}", flush=True)
    else:
        print(f"Reusing existing {summary_csv}", flush=True)

    profile_csv = motifs_dir / "motif_profiles.csv"
    if rewrite or not profile_csv.is_file():
        _write_matrix_csv(profile_csv, motif_names, motif_profiles_filled)
        print(f"Saved {profile_csv}", flush=True)
    else:
        print(f"Reusing existing {profile_csv}", flush=True)

    profiles_path = output_dir / "motif_contributions.p"
    if rewrite or not profiles_path.is_file():
        se.PickleDump(
            {
                "motif_names": motif_names,
                "motif_matrices": scored["summary_motif_matrices"],
                "motif_profiles": motif_profiles,
                "motif_profiles_filled": motif_profiles_filled,
                "max_abs_contribution": max_abs,
                "baseline_scores": scored["baseline_scores"],
            },
            str(profiles_path),
        )
        print(f"Saved {profiles_path}", flush=True)
    else:
        print(f"Reusing existing {profiles_path}", flush=True)

    hist_path = figures_dir / "most_contributive_hist.png"
    if rewrite or not hist_path.is_file():
        finite_values = max_abs[np.isfinite(max_abs)]
        figure, axis = plt.subplots(figsize=(8, 5))
        if len(finite_values):
            axis.hist(finite_values, bins=50, color="black")
        axis.set(
            title="Most contributive motif scores",
            xlabel="Contribution at maximum absolute position",
            ylabel="Number of motifs",
        )
        figure.savefig(hist_path, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {hist_path}", flush=True)
    else:
        print(f"Reusing existing {hist_path}", flush=True)

    top_profiles_path = figures_dir / "top_motif_contributions.png"
    if rewrite or not top_profiles_path.is_file():
        figure, axis = plt.subplots(figsize=(12, 6))
        for index in profile_indices:
            axis.plot(
                np.arange(len(motif_profiles_filled[index])),
                motif_profiles_filled[index],
                label=motif_names[index],
            )
        axis.set(
            title=f"Top {len(profile_indices)} DNA motif contributions",
            xlabel="Position",
            ylabel="Contribution",
        )
        if len(profile_indices):
            axis.legend(fontsize="small", ncol=2)
        figure.savefig(top_profiles_path, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {top_profiles_path}", flush=True)
    else:
        print(f"Reusing existing {top_profiles_path}", flush=True)

    return [
        kmer_csv,
        summary_csv,
        profile_csv,
        profiles_path,
        hist_path,
        top_profiles_path,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute weighted DNA motif contributions for Oyster or the DNA "
            "branch of Dual_Oyster. Input-signal attribution is not included."
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
        "--motif-file",
        type=Path,
        required=True,
        help="MEME-format motif database to score, such as a HOCOMOCO collection.",
    )
    parser.add_argument("--xp-run", default=DEFAULT_XP_RUN, help="XP run folder (default: xp2_0)")
    parser.add_argument("--loss", default=DEFAULT_LOSS, help="Fitting loss folder (default: MSE-R)")
    parser.add_argument(
        "--random-sequences",
        type=int,
        default=16_384,
        help="Random sequences sampled per motif k-mer (default: 16384).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16_384,
        help="Scoring batch size (default: 16384).",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of motif profiles to plot.")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Recompute scores and overwrite every DNA interpretation output.",
    )
    args = parser.parse_args(argv)
    if args.random_sequences <= 0 or args.batch_size <= 0 or args.top <= 0:
        parser.error("--random-sequences, --batch-size, and --top must be positive.")

    try:
        experiment = read_experiment(args.experiment_file)
        print(
            f"Loaded experiment definition from {args.experiment_file}.",
            flush=True,
        )
        if not args.motif_file.is_file():
            raise FileNotFoundError(f"Motif database not found: {args.motif_file}")
        experiment_dir = (
            args.experiment_root.resolve() / experiment["experiment_name"]
        )
        fit_folder, _, _, _ = resolve_fit_folder(
            experiment_dir,
            "Dual_Oyster" if experiment["input_signals"] else "Oyster",
            args.xp_run,
            args.loss,
        )
        print(f"Using fitted Oyster model outputs from {fit_folder}.", flush=True)
        ensemble_dir = fit_folder / "ensemble_interp"
        manifest = load_ensemble_manifest(ensemble_dir)
        expected_model = "Dual_Oyster" if experiment["input_signals"] else "Oyster"
        if manifest["model_name"] != expected_model:
            raise ValueError(
                f"Ensemble is for {manifest['model_name']}, but experiment "
                f"metadata selects {expected_model}."
            )
        config_model_name = expected_model
        config_path = experiment_dir / "opt" / config_model_name / "RO_args.p"
        ro_args = se.PickleLoad(str(config_path))
        ro_path = Path(ro_args["pn"])
        if not ro_path.is_absolute():
            ro_path = experiment_dir / ro_path
        config = se.PickleLoad(str(ro_path / "best_config.p"))
        pack_path = experiment_dir / "packs" / "tri_tpacks.p"
        if not pack_path.is_file():
            raise FileNotFoundError(f"Experiment sequence packs not found: {pack_path}")
        packs = se.PickleLoad(str(pack_path))
        expected_pack_count = 5 if experiment["input_signals"] else 4
        if not isinstance(packs, list) or len(packs) != expected_pack_count:
            raise ValueError(
                f"{manifest['model_name']} requires {expected_pack_count} tri-packs "
                f"in {pack_path}."
            )
        seq_pack = np.asarray(packs[0])
        if seq_pack.ndim < 3:
            raise ValueError(
                f"Unexpected DNA sequence pack shape in {pack_path}: {seq_pack.shape}."
            )
        config_prefix = "A_" if manifest["model_name"] == "Dual_Oyster" else ""
        sequence_length = int(config[f"{config_prefix}dim_i"][1])
        if seq_pack.shape[2] != sequence_length:
            raise ValueError(
                "Oyster configuration and sequence pack lengths differ: "
                f"{sequence_length} vs {seq_pack.shape[2]}."
            )

        output_dir = ensemble_dir / "dna"
        cache_path = output_dir / "kmer_motif_scores.p"
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_outputs = [
            cache_path,
            output_dir / "motif_contributions.p",
            output_dir / "tables" / "kmer_motif_contributions.csv",
            output_dir / "tables" / "motif_contributions.csv",
            output_dir / "tables" / "motif_profiles.csv",
            output_dir / "figs" / "most_contributive_hist.png",
            output_dir / "figs" / "top_motif_contributions.png",
        ]
        if (
            not args.rewrite
            and all(path.is_file() for path in expected_outputs)
        ):
            scored = se.PickleLoad(str(cache_path))
            se.PickleLoad(str(output_dir / "motif_contributions.p"))
            print(
                f"All DNA interpretation outputs already exist; loaded cached "
                f"results from {output_dir}. Use --rewrite to regenerate them.",
                flush=True,
            )
        else:
            if cache_path.is_file() and not args.rewrite:
                scored = se.PickleLoad(str(cache_path))
                print(f"Loaded cached motif scores from {cache_path}.", flush=True)
            else:
                print(
                    f"Scoring DNA motifs for {experiment['experiment_name']!r} "
                    f"with {len(manifest['model_paths'])} weighted "
                    f"{manifest['model_name']} repeats.",
                    flush=True,
                )
                scored = _score_motifs(
                    manifest["model_paths"],
                    np.asarray(manifest["weights"], dtype=float),
                    config,
                    manifest["model_name"],
                    args.motif_file,
                    args.random_sequences,
                    args.batch_size,
                )
                se.PickleDump(scored, str(cache_path))
                print(f"Saved motif score cache to {cache_path}.", flush=True)
            print("Writing missing or rewritten motif summaries and figures.", flush=True)
            outputs = write_motif_outputs(
                scored,
                config,
                output_dir,
                args.top,
                rewrite=args.rewrite,
            )
            print(
                f"DNA interpretation outputs are ready in {output_dir}.",
                flush=True,
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"interp_dna: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
