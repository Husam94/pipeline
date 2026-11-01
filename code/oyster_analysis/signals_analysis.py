#!/usr/bin/env python3
"""Interpret DNA and input-signal contributions for a Dual_Oyster ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, KMeans

import poseigen_oyster.analysis as interp
import poseigen_seaside.basics as se
import poseigen_trident.utils as tu

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dna_analysis import load_ensemble_manifest
from ensemble_analysis import (
    DEFAULT_EXPERIMENT_FILE,
    DEFAULT_EXPERIMENT_ROOT,
    model_analysis_dir,
    read_experiment,
    resolve_fit_folder,
)
from data_preparation.experiment_context import (
    build_signal_pairs,
    read_metadata,
    resolve_signal_pair,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_FILE = PROJECT_ROOT / "data" / "ENCODE" / "meta_info.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _file_signature(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment data not found: {path}")
    stat = path.stat()
    return {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _load_model_config(ensemble_dir: Path) -> dict[str, Any]:
    config_path = ensemble_dir / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing Oyster model metadata: {config_path}. "
            "Run ensemble_analysis.py first."
        )
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected model configuration in {config_path}.")
    return config


def _signal_names(
    signal_ids: list[str],
    metadata_file: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    rows = read_metadata(metadata_file)
    pairs = build_signal_pairs(rows)
    targets: list[str] = []
    mapping: list[dict[str, str]] = []
    for signal_id in signal_ids:
        pair = resolve_signal_pair(signal_id, rows, pairs)
        target_names = {
            (row.get("Target") or "").strip()
            for row in rows
            if (row.get("Dataset") or "").strip() == pair.target_dataset
            and (row.get("Target") or "").strip()
        }
        if len(target_names) != 1:
            raise ValueError(
                f"Signal {signal_id!r} resolves to target dataset "
                f"{pair.target_dataset}, which has {len(target_names)} distinct "
                "Target names in meta_info.txt."
            )
        target = next(iter(target_names))
        targets.append(target)
        mapping.append(
            {
                "signal_id": signal_id,
                "target": target,
                "pair_id": pair.pair_id,
                "dataset": pair.target_dataset,
            }
        )

    counts: dict[str, int] = {}
    names: list[str] = []
    for target, signal_id in zip(targets, signal_ids):
        counts[target] = counts.get(target, 0) + 1
        names.append(
            f"{target} [{signal_id}]" if targets.count(target) > 1 else target
        )
    if len(set(names)) != len(names):
        raise ValueError("Input signal names are not unique after metadata resolution.")
    return names, mapping


def _prepare_analysis_data(
    packs: list[Any],
    split: Any,
    config: dict[str, Any],
    max_sites: int,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(packs, list) or len(packs) != 5:
        raise ValueError(
            "Dual_Oyster interpretation requires five tri-packs: DNA, output, "
            "scores, bins, and input signals."
        )
    if not isinstance(split, (list, tuple)) or len(split) < 4:
        raise ValueError("The experiment split file must contain a test split.")

    dna = np.asarray(packs[0], dtype=np.float32)
    target = np.asarray(packs[1], dtype=np.float32)
    bins = np.asarray(packs[3])
    signals = np.asarray(packs[4], dtype=np.float32)
    if dna.ndim != 4 or signals.ndim != 4:
        raise ValueError(
            "Expected DNA and input-signal packs in (sites, channels, length, "
            f"width) format; found {dna.shape} and {signals.shape}."
        )
    n_sites = len(dna)
    if len(signals) != n_sites or len(target) != n_sites or len(bins) != n_sites:
        raise ValueError("Experiment packs do not have the same number of sites.")

    expected_dna_shape = tuple(config.get("A_dim_i", ()))
    expected_signal_shape = tuple(config.get("B_dim_i", ()))
    if dna.shape[1:] != expected_dna_shape:
        raise ValueError(
            f"DNA pack shape {dna.shape[1:]} does not match model "
            f"A_dim_i={expected_dna_shape}."
        )
    if signals.shape[1:] != expected_signal_shape:
        raise ValueError(
            f"Input-signal pack shape {signals.shape[1:]} does not match model "
            f"B_dim_i={expected_signal_shape}."
        )
    if not np.all(np.isfinite(dna)) or not np.all(np.isfinite(signals)):
        raise ValueError("DNA and input-signal packs must contain finite values.")

    test_indices = np.asarray(split[-1], dtype=np.int64).reshape(-1)
    if np.any(test_indices < 0) or np.any(test_indices >= n_sites):
        raise ValueError("The test split contains an index outside the experiment packs.")
    eligible_sites = np.setdiff1d(np.arange(n_sites), np.unique(test_indices))
    if not len(eligible_sites):
        raise ValueError("No non-test sites are available for interpretation.")

    rng = np.random.default_rng(seed)
    if max_sites and len(eligible_sites) > max_sites:
        eligible_sites = np.sort(
            rng.choice(eligible_sites, size=max_sites, replace=False)
        )

    forward_dna = dna[eligible_sites]
    forward_signals = signals[eligible_sites]
    analysis_dna = np.concatenate(
        [forward_dna, np.flip(forward_dna, axis=(-2, -1))], axis=0
    )
    analysis_signals = np.concatenate(
        [forward_signals, np.flip(forward_signals, axis=(-2, -1))], axis=0
    )
    target_values = target.reshape(n_sites, -1)[:, 0][eligible_sites]
    bin_values = bins.reshape(n_sites, -1)[:, 0][eligible_sites]
    return {
        "dna": analysis_dna,
        "signals": analysis_signals,
        "source_indices": np.concatenate([eligible_sites, eligible_sites]),
        "orientations": np.asarray(
            ["forward"] * len(eligible_sites) + ["reverse-complement"] * len(eligible_sites)
        ),
        "target_values": np.concatenate([target_values, target_values]),
        "bin_values": np.concatenate([bin_values, bin_values]),
        "selected_sites": eligible_sites,
    }


def _load_models(model_paths: list[str]) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    for path in model_paths:
        checkpoint = Path(f"{path}.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Ensemble checkpoint not found: {checkpoint}")
        model = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Checkpoint does not contain a PyTorch model: {checkpoint}")
        models.append(model.to(DEVICE).eval())
    if not models:
        raise ValueError("The ensemble manifest does not contain any models.")
    return models


def _sequence_contributions(
    models: list[torch.nn.Module],
    model_weights: np.ndarray,
    dna: np.ndarray,
    signals: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    contributions: list[np.ndarray] = []
    for model_index, model in enumerate(models, start=1):
        branch = getattr(model, "OysterA", None)
        if branch is None:
            raise ValueError("An ensemble checkpoint has no OysterA DNA branch.")
        activations: list[np.ndarray] = []

        def capture(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("OysterA returned a non-tensor activation.")
            activations.append(output.detach().cpu().numpy())

        hook = branch.register_forward_hook(capture)
        try:
            with torch.no_grad():
                for start in range(0, len(dna), batch_size):
                    end = min(start + batch_size, len(dna))
                    dna_batch = torch.from_numpy(dna[start:end]).to(DEVICE)
                    signal_batch = torch.from_numpy(signals[start:end]).to(DEVICE)
                    model(dna_batch, signal_batch)
        finally:
            hook.remove()

        if not activations:
            raise RuntimeError("No sequence-branch activations were collected.")
        model_contribution = np.concatenate(activations, axis=0)
        if len(model_contribution) != len(dna):
            raise ValueError("DNA branch activation count does not match selected sites.")
        contributions.append(model_contribution)
        print(
            f"Extracted DNA-branch contributions from ensemble model "
            f"{model_index}/{len(models)}.",
            flush=True,
        )
    shapes = {item.shape for item in contributions}
    if len(shapes) != 1:
        raise ValueError("DNA branch contributions differ in shape across models.")
    return np.average(np.stack(contributions), axis=0, weights=model_weights)


def _signal_contributions(
    models: list[torch.nn.Module],
    model_weights: np.ndarray,
    signals: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Compute per-signal input-times-gradient contributions for OysterB."""
    model_contributions: list[np.ndarray] = []
    for model_index, model in enumerate(models, start=1):
        branch = getattr(model, "OysterB", None)
        if branch is None:
            raise ValueError("An ensemble checkpoint has no OysterB signal branch.")

        batches: list[np.ndarray] = []
        with torch.enable_grad():
            for start in range(0, len(signals), batch_size):
                signal_batch = torch.from_numpy(
                    signals[start : start + batch_size].copy()
                ).to(DEVICE)
                signal_batch.requires_grad_(True)
                branch_output = branch(signal_batch)
                if not isinstance(branch_output, torch.Tensor):
                    raise TypeError("OysterB returned a non-tensor activation.")
                gradients = torch.autograd.grad(
                    branch_output.sum(),
                    signal_batch,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                batch_contribution = (signal_batch * gradients).sum(dim=(-2, -1))
                batches.append(batch_contribution.detach().cpu().numpy())
        model_contribution = np.concatenate(batches, axis=0)
        model_contributions.append(model_contribution)
        print(
            f"Extracted input-signal contributions from ensemble model "
            f"{model_index}/{len(models)}.",
            flush=True,
        )
    shapes = {item.shape for item in model_contributions}
    if len(shapes) != 1:
        raise ValueError("Input-signal contributions differ in shape across models.")
    return np.average(np.stack(model_contributions), axis=0, weights=model_weights)


def _cluster_signal_kmers(
    signals: np.ndarray,
    kmer_length: int,
    requested_clusters: int,
    sample_limit: int,
    batch_sites: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    number_of_sites, number_of_signals, signal_length, _ = signals.shape
    if kmer_length <= 0 or kmer_length > signal_length:
        raise ValueError(
            f"Signal k-mer length {kmer_length} is invalid for input width "
            f"{signal_length}."
        )
    windows_per_site = signal_length - kmer_length + 1
    total_windows = number_of_sites * windows_per_site
    if total_windows == 0:
        raise ValueError("No signal k-mer windows are available to cluster.")
    sample_count = min(sample_limit, total_windows)
    number_of_clusters = min(requested_clusters, sample_count)
    labels = np.empty(
        (number_of_sites, windows_per_site, number_of_signals), dtype=np.int32
    )
    estimators: list[KMeans] = []

    for signal_index in range(number_of_signals):
        rng = np.random.default_rng(seed + signal_index)
        sample_ids = rng.choice(total_windows, size=sample_count, replace=False)
        site_ids = sample_ids // windows_per_site
        starts = sample_ids % windows_per_site
        sample_windows = signals[
            site_ids[:, None],
            signal_index,
            starts[:, None] + np.arange(kmer_length),
            0,
        ]
        estimator = KMeans(
            n_clusters=number_of_clusters,
            n_init=10,
            random_state=seed + signal_index,
        )
        estimator.fit(sample_windows)
        estimators.append(estimator)

        for first_site in range(0, number_of_sites, batch_sites):
            last_site = min(first_site + batch_sites, number_of_sites)
            window_batch = tu.TridentWindow(
                signals[first_site:last_site], kmer_length
            )
            feature_batch = window_batch[:, :, signal_index].reshape(
                -1, kmer_length
            )
            labels[first_site:last_site, :, signal_index] = estimator.predict(
                feature_batch
            ).reshape(last_site - first_site, windows_per_site)
        print(
            f"Clustered signal {signal_index + 1}/{number_of_signals} "
            f"into at most {number_of_clusters} initial groups.",
            flush=True,
        )

    centroids_by_signal: list[np.ndarray] = []
    refinement_info: list[dict[str, Any]] = []
    for signal_index, estimator in enumerate(estimators):
        original_labels = labels[:, :, signal_index].reshape(-1)
        counts = np.bincount(original_labels, minlength=number_of_clusters)
        sums = np.zeros((number_of_clusters, kmer_length), dtype=np.float64)
        for first_site in range(0, number_of_sites, batch_sites):
            last_site = min(first_site + batch_sites, number_of_sites)
            window_batch = tu.TridentWindow(
                signals[first_site:last_site], kmer_length
            )
            feature_batch = window_batch[:, :, signal_index].reshape(
                -1, kmer_length
            )
            batch_labels = labels[
                first_site:last_site, :, signal_index
            ].reshape(-1)
            for feature_index in range(kmer_length):
                sums[:, feature_index] += np.bincount(
                    batch_labels,
                    weights=feature_batch[:, feature_index],
                    minlength=number_of_clusters,
                )
        active_ids = np.flatnonzero(counts)
        if not len(active_ids):
            raise RuntimeError(f"No k-mer clusters were assigned to signal {signal_index}.")
        centers = sums[active_ids] / counts[active_ids, None]
        merged_labels = np.arange(len(active_ids), dtype=np.int32)
        knee_index: int | None = None
        if len(active_ids) >= 3:
            hierarchy = AgglomerativeClustering(
                distance_threshold=0,
                n_clusters=None,
                compute_full_tree=True,
            ).fit(centers)
            distances = hierarchy.distances_
            knee = KneeLocator(
                np.arange(len(distances)),
                distances,
                S=1,
                curve="convex",
                direction="increasing",
            ).knee
            if knee is not None and 0 <= int(knee) < len(distances):
                threshold = float(distances[int(knee)])
                if threshold > 0:
                    merged_labels = AgglomerativeClustering(
                        distance_threshold=threshold,
                        n_clusters=None,
                    ).fit_predict(centers)
                    knee_index = int(knee)
                else:
                    print(
                        f"Signal {signal_index + 1}: refinement knee has zero "
                        "distance; retaining initial groups.",
                        flush=True,
                    )
            else:
                print(
                    f"Signal {signal_index + 1}: no stable refinement knee; "
                    "retaining initial groups.",
                    flush=True,
                )

        merged_ids = np.unique(merged_labels)
        compact_labels = np.searchsorted(merged_ids, merged_labels).astype(np.int32)
        final_counts = np.bincount(compact_labels, weights=counts[active_ids])
        final_sums = np.zeros((len(merged_ids), kmer_length), dtype=np.float64)
        np.add.at(final_sums, compact_labels, centers * counts[active_ids, None])
        final_centers = final_sums / final_counts[:, None]

        old_to_final = np.full(number_of_clusters, -1, dtype=np.int32)
        old_to_final[active_ids] = compact_labels
        labels[:, :, signal_index] = old_to_final[
            labels[:, :, signal_index]
        ]
        centroids_by_signal.append(final_centers.astype(np.float32))
        refinement_info.append(
            {
                "initial_clusters_used": int(len(active_ids)),
                "refined_clusters": int(len(merged_ids)),
                "refinement_knee_index": knee_index,
                "cluster_counts": final_counts.astype(np.int64),
            }
        )
        print(
            f"Refined signal {signal_index + 1}: {len(active_ids)} initial "
            f"groups -> {len(merged_ids)} groups.",
            flush=True,
        )
    return labels, centroids_by_signal, refinement_info


def _cluster_contributions(
    models: list[torch.nn.Module],
    model_weights: np.ndarray,
    signals: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: list[np.ndarray],
    signal_names: list[str],
    kmer_length: int,
    batch_sites: int,
    batch_size: int,
    dim_kmer_length: int,
) -> list[dict[str, Any]]:
    number_of_sites, windows_per_site, number_of_signals = cluster_labels.shape
    if number_of_signals != len(signal_names):
        raise ValueError("Signal labels do not align with signal k-mer clusters.")
    if len(centroids) != number_of_signals:
        raise ValueError("Signal centroids do not align with signal k-mer clusters.")

    counts_by_signal = [
        np.bincount(
            cluster_labels[:, :, signal_index].reshape(-1),
            minlength=len(centroids[signal_index]),
        )
        for signal_index in range(number_of_signals)
    ]
    denominators = [
        np.zeros(len(centers), dtype=np.float64) for centers in centroids
    ]
    weighted_sums = [
        np.zeros((len(centers), dim_kmer_length), dtype=np.float64)
        for centers in centroids
    ]
    weighted_squares = [
        np.zeros((len(centers), dim_kmer_length), dtype=np.float64)
        for centers in centroids
    ]

    for first_site in range(0, number_of_sites, batch_sites):
        last_site = min(first_site + batch_sites, number_of_sites)
        window_batch = tu.TridentWindow(
            signals[first_site:last_site], kmer_length
        )
        contributions = interp.kmer_contrib_multi(
            models,
            np.array(window_batch, copy=True),
            sub_model="OysterB",
            joint=False,
            chunk=True,
            batchsize=batch_size,
            model_weights=model_weights,
            dim_kE_length=dim_kmer_length,
        )
        if contributions.shape[:3] != window_batch.shape[:3]:
            raise ValueError(
                "Signal k-mer contributions do not align with signal windows: "
                f"{contributions.shape} vs {window_batch.shape}."
            )
        if contributions.shape[-1] != dim_kmer_length:
            raise ValueError(
                "Signal attribution width does not match OysterB's output "
                f"positions ({contributions.shape[-1]} vs {dim_kmer_length})."
            )

        for signal_index, centers in enumerate(centroids):
            features = window_batch[:, :, signal_index].reshape(
                -1, kmer_length
            )
            labels = cluster_labels[
                first_site:last_site, :, signal_index
            ].reshape(-1)
            contribution = contributions[
                :, :, signal_index, :
            ].reshape(-1, dim_kmer_length)
            distances = np.linalg.norm(features - centers[labels], axis=1)
            scores = 1.0 / np.maximum(distances, 1e-12)
            np.add.at(denominators[signal_index], labels, scores)
            np.add.at(
                weighted_sums[signal_index],
                labels,
                contribution * scores[:, None],
            )
            np.add.at(
                weighted_squares[signal_index],
                labels,
                contribution**2 * scores[:, None],
            )
        print(
            f"Attributed signal k-mers for sites "
            f"{last_site:,}/{number_of_sites:,}.",
            flush=True,
        )

    cluster_results: list[dict[str, Any]] = []
    for signal_index, signal_name in enumerate(signal_names):
        means = weighted_sums[signal_index] / denominators[signal_index][:, None]
        variances = np.maximum(
            weighted_squares[signal_index] / denominators[signal_index][:, None]
            - means**2,
            0,
        )
        for cluster_id, (centroid, mean, variance) in enumerate(
            zip(centroids[signal_index], means, variances)
        ):
            cluster_results.append(
                {
                    "signal_index": signal_index,
                    "signal_name": signal_name,
                    "cluster_id": cluster_id,
                    "window_count": int(counts_by_signal[signal_index][cluster_id]),
                    "centroid": centroid,
                    "mean_contribution": mean,
                    "contribution_std": np.sqrt(variance),
                }
            )
    return cluster_results


def _normalized_influence(
    sequence_contributions: np.ndarray,
    signal_contributions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    sequence_values = sequence_contributions.reshape(
        len(sequence_contributions), -1
    )
    names = (
        ["DNA"]
        if sequence_values.shape[1] == 1
        else [f"DNA_{index + 1}" for index in range(sequence_values.shape[1])]
    )
    combined = np.concatenate([sequence_values, signal_contributions], axis=1)
    absolute_values = np.abs(combined)
    totals = absolute_values.sum(axis=1, keepdims=True)
    influence = np.divide(
        absolute_values,
        totals,
        out=np.zeros_like(absolute_values, dtype=np.float64),
        where=totals > 0,
    )
    return influence, names


def _cache_signature(
    experiment: dict[str, Any],
    experiment_dir: Path,
    pack_path: Path,
    split_path: Path,
    metadata_file: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    signal_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment["experiment_name"],
        "signal_ids": experiment["input_signals"],
        "signal_names": signal_names,
        "model_paths": manifest["model_paths"],
        "model_weights": manifest["weights"],
        "A_dim_i": list(config["A_dim_i"]),
        "B_dim_i": list(config["B_dim_i"]),
        "B_kE_k": int(config["B_kE_k"]),
        "packs": _file_signature(pack_path),
        "split": _file_signature(split_path),
        "metadata": _file_signature(metadata_file),
        "bin_sizes": _file_signature(
            experiment_dir / "packs" / "output_bin_sizes.p"
        ),
        "max_sites": args.max_sites,
        "clusters": args.clusters,
        "cluster_sample": args.cluster_sample,
        "batch_size": args.batch_size,
        "top_clusters": args.top_clusters,
        "seed": args.seed,
        "experiment_dir": str(experiment_dir.resolve()),
        "checkpoints": [
            _file_signature(Path(f"{path}.pt")) for path in manifest["model_paths"]
        ],
    }


def _write_csv_if_needed(
    path: Path,
    rewrite: bool,
    writer_function: Any,
) -> None:
    if path.is_file() and not rewrite:
        print(f"Reusing existing {path}.", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer_function(path)
    print(f"Saved {path}.", flush=True)


def _write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    figures_dir: Path,
    tables_dir: Path,
    bin_sizes: Any,
    rewrite: bool,
    top_clusters: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    signal_names = result["signal_names"]
    signal_mapping = result["signal_mapping"]
    feature_names = result["feature_names"]
    source_indices = result["source_indices"]
    orientations = result["orientations"]
    target_values = result["target_values"]
    bin_values = result["bin_values"].astype(int)
    feature_values = np.concatenate(
        [
            result["sequence_contributions"].reshape(
                len(source_indices), -1
            ),
            result["signal_contributions"],
        ],
        axis=1,
    )
    influence_values = np.asarray(result["influence"])
    outputs: list[Path] = []

    names_path = output_dir / "signal_names.json"

    def write_names(path: Path) -> None:
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(signal_mapping, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")

    _write_csv_if_needed(names_path, rewrite, write_names)
    outputs.append(names_path)

    sequence_path = output_dir / "sequence_contributions.p"
    if rewrite or not sequence_path.is_file():
        se.PickleDump(
            result["sequence_contributions"],
            str(sequence_path),
        )
        print(f"Saved {sequence_path}.", flush=True)
    else:
        print(f"Reusing existing {sequence_path}.", flush=True)
    outputs.append(sequence_path)

    signal_path = output_dir / "signal_contributions.p"
    if rewrite or not signal_path.is_file():
        se.PickleDump(
            {
                "signal_names": signal_names,
                "signal_mapping": signal_mapping,
                "contributions": result["signal_contributions"],
            },
            str(signal_path),
        )
        print(f"Saved {signal_path}.", flush=True)
    else:
        print(f"Reusing existing {signal_path}.", flush=True)
    outputs.append(signal_path)

    feature_csv = tables_dir / "feature_contributions.csv"

    def write_features(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "source_site_index",
                    "orientation",
                    "output_bin",
                    "output_value",
                    *feature_names,
                ]
            )
            for index, values in enumerate(feature_values):
                writer.writerow(
                    [
                        int(source_indices[index]),
                        orientations[index],
                        int(bin_values[index]),
                        float(target_values[index]),
                        *values.tolist(),
                    ]
                )

    _write_csv_if_needed(feature_csv, rewrite, write_features)
    outputs.append(feature_csv)

    bin_sizes_array = np.asarray(bin_sizes, dtype=float).reshape(-1)
    if (
        not len(bin_sizes_array)
        or np.any(~np.isfinite(bin_sizes_array))
        or np.any(bin_sizes_array < 0)
        or bin_sizes_array.sum() <= 0
    ):
        raise ValueError("Output bin sizes must be finite non-negative values.")
    if np.any(bin_values < 0) or np.any(bin_values >= len(bin_sizes_array)):
        raise ValueError("Output bin IDs do not align with output_bin_sizes.p.")
    edges01 = np.concatenate([[0.0], np.cumsum(bin_sizes_array)])
    edges01 /= edges01[-1]
    target_min = float(np.min(target_values))
    target_max = float(np.max(target_values))
    edges = target_min + (target_max - target_min) * edges01
    medians = np.full((len(bin_sizes_array), influence_values.shape[1]), np.nan)
    for bin_index in range(len(bin_sizes_array)):
        selected = bin_values == bin_index
        if np.any(selected):
            medians[bin_index] = np.median(influence_values[selected], axis=0)

    influence_csv = tables_dir / "signal_influence_by_bin.csv"

    def write_influence(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "bin_index",
                    "target_minimum",
                    "target_maximum",
                    "feature",
                    "median_absolute_influence",
                ]
            )
            for bin_index, row in enumerate(medians):
                for feature_name, value in zip(feature_names, row):
                    writer.writerow(
                        [
                            bin_index,
                            edges[bin_index],
                            edges[bin_index + 1],
                            feature_name,
                            value,
                        ]
                    )

    _write_csv_if_needed(influence_csv, rewrite, write_influence)
    outputs.append(influence_csv)

    influence_figure = figures_dir / "signal_influence_by_target_bin.png"
    if rewrite or not influence_figure.is_file():
        figure, axis = plt.subplots(figsize=(11, 6))
        for feature_index, feature_name in enumerate(feature_names):
            axis.stairs(
                medians[:, feature_index],
                edges,
                label=feature_name,
                linewidth=1.8,
            )
        axis.set(
            xlabel="Output signal value",
            ylabel="Median absolute contribution influence",
            title="DNA and input-signal influence by output bin",
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize="small", frameon=False)
        figure.savefig(influence_figure, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {influence_figure}.", flush=True)
    else:
        print(f"Reusing existing {influence_figure}.", flush=True)
    outputs.append(influence_figure)

    cluster_path = output_dir / "signal_kmer_clusters.p"
    cluster_data = {
        "signal_names": signal_names,
        "signal_mapping": signal_mapping,
        "labels": result["cluster_labels"],
        "centroids": result["cluster_centroids"],
        "refinement": result["cluster_refinement"],
        "contributions": result["cluster_contributions"],
    }
    if rewrite or not cluster_path.is_file():
        se.PickleDump(cluster_data, str(cluster_path))
        print(f"Saved {cluster_path}.", flush=True)
    else:
        print(f"Reusing existing {cluster_path}.", flush=True)
    outputs.append(cluster_path)

    cluster_csv = tables_dir / "signal_kmer_cluster_contributions.csv"

    def write_clusters(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                [
                    "signal",
                    "signal_id",
                    "cluster_id",
                    "window_count",
                    "centroid_values",
                    "mean_contribution_by_position",
                    "contribution_std_by_position",
                ]
            )
            for row in result["cluster_contributions"]:
                writer.writerow(
                    [
                        row["signal_name"],
                        signal_mapping[row["signal_index"]]["signal_id"],
                        row["cluster_id"],
                        row["window_count"],
                        json.dumps(np.asarray(row["centroid"]).tolist()),
                        json.dumps(
                            np.asarray(row["mean_contribution"]).tolist()
                        ),
                        json.dumps(
                            np.asarray(row["contribution_std"]).tolist()
                        ),
                    ]
                )

    _write_csv_if_needed(cluster_csv, rewrite, write_clusters)
    outputs.append(cluster_csv)

    cluster_figure = figures_dir / "signal_kmer_cluster_contributions.png"
    if rewrite or not cluster_figure.is_file():
        figure, axes = plt.subplots(
            len(signal_names),
            1,
            figsize=(11, max(3, 2.5 * len(signal_names))),
            squeeze=False,
        )
        for signal_index, (signal_name, axis) in enumerate(
            zip(signal_names, axes[:, 0])
        ):
            rows = [
                item
                for item in result["cluster_contributions"]
                if item["signal_index"] == signal_index
            ]
            rows.sort(
                key=lambda item: abs(
                    float(np.mean(item["mean_contribution"]))
                ),
                reverse=True,
            )
            rows = rows[:top_clusters]
            cluster_ids = [item["cluster_id"] for item in rows]
            means = [
                float(np.mean(item["mean_contribution"])) for item in rows
            ]
            colors = ["firebrick" if value < 0 else "steelblue" for value in means]
            axis.bar(cluster_ids, means, color=colors)
            axis.set(
                title=signal_name,
                xlabel="Refined k-mer cluster",
                ylabel="Mean contribution",
            )
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Input-signal k-mer cluster contributions")
        figure.tight_layout()
        figure.savefig(cluster_figure, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {cluster_figure}.", flush=True)
    else:
        print(f"Reusing existing {cluster_figure}.", flush=True)
    outputs.append(cluster_figure)
    return outputs


def run_analysis(args: argparse.Namespace) -> Path:
    experiment = read_experiment(args.experiment_file)
    if not experiment["input_signals"]:
        raise ValueError(
            "signals_analysis.py requires at least one input signal in the "
            "experiment JSON."
        )
    print(
        f"Loaded experiment {experiment['experiment_name']!r} with "
        f"{len(experiment['input_signals'])} input signal(s).",
        flush=True,
    )

    experiment_dir = args.experiment_root.resolve() / experiment["experiment_name"]
    _, _, _, config_name = resolve_fit_folder(
        experiment_dir,
        "Dual_Oyster",
        args.xp_run,
        args.loss,
    )
    ensemble_dir = model_analysis_dir(experiment_dir, config_name)
    manifest = load_ensemble_manifest(ensemble_dir)
    if manifest["model_name"] != "Dual_Oyster":
        raise ValueError(
            f"Expected a Dual_Oyster ensemble, found {manifest['model_name']}."
        )
    print(f"Using weighted Dual_Oyster ensemble in {ensemble_dir}.", flush=True)

    config = _load_model_config(ensemble_dir)
    input_signal_shape = tuple(config.get("B_dim_i", ()))
    kmer_length = int(config.get("B_kE_k", 0))
    if len(input_signal_shape) != 3 or kmer_length <= 0:
        raise ValueError("Best configuration has invalid OysterB dimensions.")

    pack_path = experiment_dir / "packs" / "tri_tpacks.p"
    split_path = experiment_dir / "split" / "split_transaug.p"
    packs = se.PickleLoad(str(pack_path))
    split = se.PickleLoad(str(split_path))
    metadata_file = args.metadata_file.resolve()
    signal_names, signal_mapping = _signal_names(
        experiment["input_signals"],
        metadata_file,
    )
    if len(signal_names) != input_signal_shape[0]:
        raise ValueError(
            f"Experiment lists {len(signal_names)} input signals but the "
            f"OysterB model expects {input_signal_shape[0]}."
        )
    print("Input signal labels resolved from ENCODE metadata:", flush=True)
    for name, mapping in zip(signal_names, signal_mapping):
        print(f"  {name}: {mapping['signal_id']} ({mapping['pair_id']})", flush=True)

    signature = _cache_signature(
        experiment,
        experiment_dir,
        pack_path,
        split_path,
        metadata_file,
        manifest,
        config,
        signal_names,
        args,
    )
    output_dir = ensemble_dir / "analysis" / "signals"
    figures_dir = (
        experiment_dir
        / "output"
        / "figures"
        / "oyster_models"
        / config_name
        / "signals"
    )
    tables_dir = (
        experiment_dir
        / "output"
        / "tables"
        / "oyster_models"
        / config_name
        / "signals"
    )
    cache_path = output_dir / "signals_analysis.p"
    if cache_path.is_file() and not args.rewrite:
        result = se.PickleLoad(str(cache_path))
        if not isinstance(result, dict) or result.get("signature") != signature:
            raise ValueError(
                f"Cached signal interpretation does not match the current "
                f"experiment or options: {cache_path}. Use --rewrite to recompute."
            )
        print(f"Loaded cached signal analysis from {cache_path}.", flush=True)
    else:
        analysis_data = _prepare_analysis_data(
            packs,
            split,
            config,
            args.max_sites,
            args.seed,
        )
        if analysis_data["signals"].shape[1] != len(signal_names):
            raise ValueError(
                "Input-signal pack channels do not match experiment input_signals."
            )
        print(
            f"Analyzing {len(analysis_data['selected_sites']):,} non-test sites "
            f"({len(analysis_data['source_indices']):,} orientations).",
            flush=True,
        )

        models = _load_models(manifest["model_paths"])
        model_weights = np.asarray(manifest["weights"], dtype=float)
        sequence_contributions = _sequence_contributions(
            models,
            model_weights,
            analysis_data["dna"],
            analysis_data["signals"],
            args.batch_size,
        )
        signal_contributions = _signal_contributions(
            models,
            model_weights,
            analysis_data["signals"],
            args.batch_size,
        )
        if signal_contributions.shape[1] != len(signal_names):
            raise ValueError(
                "Signal attribution channels do not match metadata signal names."
            )
        influence, dna_feature_names = _normalized_influence(
            sequence_contributions,
            signal_contributions,
        )

        print(
            f"Creating {kmer_length}-value k-mer windows for the input signals.",
            flush=True,
        )
        cluster_labels, cluster_centroids, cluster_refinement = (
            _cluster_signal_kmers(
                analysis_data["signals"],
                kmer_length,
                args.clusters,
                args.cluster_sample,
                batch_sites=512,
                seed=args.seed,
            )
        )
        dimension_kmer_length = input_signal_shape[1] - kmer_length + 1
        cluster_contributions = _cluster_contributions(
            models,
            model_weights,
            analysis_data["signals"],
            cluster_labels,
            cluster_centroids,
            signal_names,
            kmer_length,
            batch_sites=128,
            batch_size=args.batch_size,
            dim_kmer_length=dimension_kmer_length,
        )
        result = {
            "signature": signature,
            "signal_names": signal_names,
            "signal_mapping": signal_mapping,
            "feature_names": [*dna_feature_names, *signal_names],
            "source_indices": analysis_data["source_indices"],
            "orientations": analysis_data["orientations"],
            "target_values": analysis_data["target_values"],
            "bin_values": analysis_data["bin_values"],
            "sequence_contributions": sequence_contributions,
            "signal_contributions": signal_contributions,
            "influence": influence,
            "cluster_labels": cluster_labels,
            "cluster_centroids": cluster_centroids,
            "cluster_refinement": cluster_refinement,
            "cluster_contributions": cluster_contributions,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        se.PickleDump(result, str(cache_path))
        print(f"Saved signal analysis cache to {cache_path}.", flush=True)
        del models

    outputs = _write_outputs(
        result,
        output_dir,
        figures_dir,
        tables_dir,
        se.PickleLoad(str(experiment_dir / "packs" / "output_bin_sizes.p")),
        rewrite=args.rewrite,
        top_clusters=args.top_clusters,
    )
    print(
        f"Signal and sequence contribution figures and tables are ready under "
        f"{experiment_dir / 'output'} ({len(outputs)} files).",
        flush=True,
    )
    return experiment_dir / "output"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interpret DNA and input-signal contributions, signal k-mers, and "
            "k-mer clusters for one Dual_Oyster experiment."
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
        help=f"Parent directory of experiment outputs (default: {DEFAULT_EXPERIMENT_ROOT})",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help="Tab-delimited ENCODE metadata used to name input signals.",
    )
    parser.add_argument(
        "--xp-run",
        help="Override the selected fitting run; omit both overrides to use model_selection.json.",
    )
    parser.add_argument(
        "--loss",
        help="Override the selected fit condition; omit both overrides to use model_selection.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Maximum model attribution batch size (default: 2048).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=100,
        help="Initial k-mer clusters per input signal (default: 100).",
    )
    parser.add_argument(
        "--cluster-sample",
        type=int,
        default=1_000_000,
        help="Maximum k-mers used to fit each clustering model (default: 1000000).",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=60_000,
        help="Maximum non-test sites to analyze; 0 means all (default: 60000).",
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=20,
        help="Maximum clusters shown per signal in the contribution plot (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Recompute and overwrite all signal-interpretation outputs.",
    )
    args = parser.parse_args(argv)
    if min(args.batch_size, args.clusters, args.cluster_sample, args.top_clusters) <= 0:
        parser.error(
            "--batch-size, --clusters, --cluster-sample, and --top-clusters "
            "must be positive."
        )
    if args.max_sites < 0:
        parser.error("--max-sites must be non-negative.")

    try:
        output_dir = run_analysis(args)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"signals_analysis: error: {error}", file=sys.stderr)
        return 1
    print(f"Completed signals_analysis.py: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
