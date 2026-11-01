#!/usr/bin/env python3
"""Generate and select filtered experiment flag sites."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import poseigen_binmeths as bm
import poseigen_chisel.poseigen_chisel as chis
import poseigen_seaside.basics as se

if __package__:
    from .experiment_context import (
        CT_RESOLUTION,
        DEFAULT_EXPERIMENT_FILE,
        DEFAULT_OUTPUT_ROOT,
        ENCODE_ROOT,
        EXPERIMENT_NAME_PATTERN,
        GENOME_NAME_PATTERN,
        PARAMETER_NAMES,
        PROJECT_ROOT,
        build_signal_pairs,
        read_experiment_file,
        read_metadata,
        resolve_parameters,
        resolve_signal_pair,
        write_experiment_file,
    )
else:
    from experiment_context import (
        CT_RESOLUTION,
        DEFAULT_EXPERIMENT_FILE,
        DEFAULT_OUTPUT_ROOT,
        ENCODE_ROOT,
        EXPERIMENT_NAME_PATTERN,
        GENOME_NAME_PATTERN,
        PARAMETER_NAMES,
        PROJECT_ROOT,
        build_signal_pairs,
        read_experiment_file,
        read_metadata,
        resolve_parameters,
        resolve_signal_pair,
        write_experiment_file,
    )


FLAG_WINDOW = 2_000
FLAG_ZERO_ENDS = 5_000
DEFAULT_TARGET_PERCENTILE = 1
DEFAULT_CONTROL_PERCENTILE = 5


class SignalPairLike(Protocol):
    pair_id: str
    control_replicate_id: str


def _progress(message: str) -> None:
    print(f"[site_selection] {message}", flush=True)


def _pickle_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required processed data not found: {path}")
    return se.PickleLoad(str(path))


def _pickle_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    se.PickleDump(value, str(path))


def _load_current(path: Path) -> list[np.ndarray]:
    current = _pickle_load(path)
    if not isinstance(current, list) or not current:
        raise ValueError(f"Expected non-empty chromosome currents in {path}.")
    arrays = []
    for chromosome in current:
        array = np.asarray(chromosome)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f"Invalid current array dimensions in {path}.")
        arrays.append(array)
    return arrays


def _pair_file(root: Path, pair_id: str, suffix: str) -> Path:
    return root / f"{pair_id}_{suffix}.p"


def _load_pair_mode(pair: SignalPairLike) -> list[np.ndarray]:
    return _load_current(
        _pair_file(ENCODE_ROOT / "betaprime", pair.pair_id, "gbp_mode")
    )


def _load_control_mean(pair: SignalPairLike) -> list[np.ndarray]:
    return _load_current(
        _pair_file(ENCODE_ROOT / "merged", pair.control_replicate_id, "mean")
    )


def _load_vst(pair: SignalPairLike) -> Any:
    return _pickle_load(
        _pair_file(
            PROJECT_ROOT / "data" / "ENCODE" / "vst",
            pair.pair_id,
            "vst_transform",
        )
    )


def _percentile_bounds(
    values: np.ndarray,
    percentile: float,
) -> tuple[float, float]:
    valid = values[np.isfinite(values) & (values > 0)]
    if not len(valid):
        raise ValueError("Signal has no positive finite values for thresholding.")
    low, high = np.percentile(valid, [percentile, 100])
    return float(low), float(high)


def resolve_site_percentiles(
    experiment: dict[str, Any],
) -> tuple[float, float]:
    values = {
        "target_percentile": experiment.get(
            "target_percentile",
            DEFAULT_TARGET_PERCENTILE,
        ),
        "control_percentile": experiment.get(
            "control_percentile",
            DEFAULT_CONTROL_PERCENTILE,
        ),
    }
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 100
        ):
            raise ValueError(f"{name} must be a number from 0 to 100.")
    experiment.update(values)
    return float(values["target_percentile"]), float(values["control_percentile"])


def _validate_current_shapes(currents: list[list[np.ndarray]]) -> None:
    if not currents:
        return
    reference = currents[0]
    for current in currents[1:]:
        if len(current) != len(reference) or any(
            array.shape != reference_array.shape
            for array, reference_array in zip(current, reference)
        ):
            raise ValueError("Signal and control currents have mismatched shapes.")


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _filter_markers_with_stage_qc(
    markers: Any,
    bad_stage_masks: list[list[np.ndarray]],
    autosome_ids: list[str],
    window_size: int,
) -> tuple[Any, list[int]]:
    chromosome_count = len(autosome_ids)
    if any(len(stage_masks) != chromosome_count for stage_masks in bad_stage_masks):
        raise ValueError("Site-filter masks do not align with autosome chromosomes.")

    bad_current = [
        np.column_stack(
            [
                np.asarray(stage_masks[index]).reshape(-1)
                for stage_masks in bad_stage_masks
            ]
        ).astype(np.int8)
        for index in range(chromosome_count)
    ]
    marker_rows = markers[
        markers.iloc[:, 0].isin(autosome_ids)
    ].reset_index(drop=True)
    stage_passes = np.ones(
        (len(marker_rows), len(bad_stage_masks)),
        dtype=bool,
    )
    for row_index in marker_rows.index:
        window = chis.SigCurrent2Slice(
            bad_current,
            marker_rows.iloc[row_index],
            autosome_ids,
            select_BS_ids=autosome_ids,
            Vsize=window_size,
            reso=CT_RESOLUTION,
            newreso=None,
            resomode=np.mean,
            exact=False,
            Vpad=2,
        )
        stage_passes[row_index] = np.sum(window, axis=0) == 0

    if stage_passes.shape[1]:
        cumulative_passes = np.logical_and.accumulate(stage_passes, axis=1)
        stage_counts = [
            len(markers),
            *(
                int(np.count_nonzero(cumulative_passes[:, index]))
                for index in range(cumulative_passes.shape[1])
            ),
        ]
        final_passes = cumulative_passes[:, -1]
    else:
        stage_counts = [len(markers)]
        final_passes = np.ones(len(marker_rows), dtype=bool)

    filtered_markers = marker_rows.iloc[np.flatnonzero(final_passes)]
    return filtered_markers.reset_index(drop=True), stage_counts


def _save_site_selection_qc(
    figures_dir: Path,
    tables_dir: Path,
    output_pair_id: str,
    stage_counts: list[int],
    selected_count: int,
    window_size: int,
    target_percentile: float,
    control_percentile: float,
    output_minimum: float,
    output_maximum: float,
    control_thresholds: dict[str, dict[str, float]],
) -> None:
    stage_labels = [
        "Flag candidates",
        "After blacklist window",
        f"After target p{target_percentile:g} window",
        f"After control p{control_percentile:g} windows",
    ]
    labels = [*stage_labels, "Harpoon-selected"]
    counts = [*stage_counts, selected_count]
    figure, axis = plt.subplots(figsize=(10, 5))
    y_positions = np.arange(len(labels))
    axis.barh(y_positions, counts, color=[
        "slategray",
        "steelblue",
        "steelblue",
        "steelblue",
        "darkorange",
    ])
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set(
        title=f"{output_pair_id}: site-filter attrition",
        xlabel="Candidate sites retained",
    )
    axis.grid(axis="x", alpha=0.25)
    maximum_count = max(counts, default=0)
    axis.set_xlim(0, maximum_count + max(maximum_count * 0.15, 1))
    for position, count in zip(y_positions, counts):
        axis.text(
            count + max(maximum_count * 0.01, 0.5),
            position,
            f"{count:,}",
            va="center",
        )
    control_minima = ", ".join(
        f"{replicate_id}: {threshold['minimum']:.4g}"
        for replicate_id, threshold in control_thresholds.items()
    )
    figure.text(
        0.01,
        0.01,
        f"Complete {window_size:,}-bp windows; target p{target_percentile:g} "
        f"range = [{output_minimum:.4g}, {output_maximum:.4g}]; control "
        f"p{control_percentile:g} lower bounds = {control_minima or 'none'}.",
        ha="left",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    _save_figure(figure, figures_dir / "site_filter_attrition.png")
    report = {
        "pair_id": output_pair_id,
        "filter_window_bp": window_size,
        "target_thresholds": {
            "percentile": target_percentile,
            "minimum": output_minimum,
            "maximum": output_maximum,
        },
        "control_percentile": control_percentile,
        "control_thresholds": control_thresholds,
        "site_counts": {
            label: count for label, count in zip(stage_labels, stage_counts)
        },
        "harpoon_selected": selected_count,
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / f"{output_pair_id}_site_selection_qc.json").open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)
        report_file.write("\n")


def _load_selection_genome_data(
    genome: str,
) -> tuple[list[str], list[np.ndarray]]:
    if not GENOME_NAME_PATTERN.fullmatch(genome):
        raise ValueError(
            "Genome name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'."
        )
    genome_root = PROJECT_ROOT / "data" / "genomes" / genome
    basic_ids, _ = _pickle_load(genome_root / f"{genome}_basic_id_sizes.p")
    blacklist = _pickle_load(genome_root / f"{genome}_basic_BLC_LR50.p")
    if len(basic_ids) < 4:
        raise ValueError(f"{genome}: basic chromosome list is too short.")
    autosome_ids = basic_ids[:-3]
    autosome_blacklist = blacklist[: len(autosome_ids)]
    if len(autosome_blacklist) != len(autosome_ids):
        raise ValueError(f"{genome}: blacklist is not aligned to basic chromosomes.")
    return autosome_ids, autosome_blacklist


def select_sites(
    experiment_dir: Path,
    output_pair: SignalPairLike,
    input_pairs: list[SignalPairLike],
    parameters: dict[str, int],
    autosome_ids: list[str],
    blacklist: list[np.ndarray],
    target_percentile: float = DEFAULT_TARGET_PERCENTILE,
    control_percentile: float = DEFAULT_CONTROL_PERCENTILE,
) -> Any:
    unique_controls = {
        pair.control_replicate_id for pair in [output_pair, *input_pairs]
    }
    _progress(
        f"{output_pair.pair_id}: loading output signal and "
        f"{len(unique_controls)} unique control current(s)."
    )
    output_mode = _load_pair_mode(output_pair)
    all_pairs = [output_pair, *input_pairs]
    controls_by_id: dict[str, list[np.ndarray]] = {}
    for pair in all_pairs:
        if pair.control_replicate_id not in controls_by_id:
            controls_by_id[pair.control_replicate_id] = _load_control_mean(pair)
    controls = list(controls_by_id.values())
    _validate_current_shapes([output_mode, *controls, blacklist])

    output_values = np.vstack(output_mode).reshape(-1)
    output_min, output_max = _percentile_bounds(
        output_values,
        target_percentile,
    )
    control_masks: list[list[np.ndarray]] = []
    control_thresholds: dict[str, dict[str, float]] = {}
    for replicate_id, current in controls_by_id.items():
        low, high = _percentile_bounds(
            np.vstack(current).reshape(-1),
            control_percentile,
        )
        control_thresholds[replicate_id] = {"minimum": low, "maximum": high}
        control_masks.append(
            [
                np.isfinite(chromosome)
                & (chromosome >= low)
                & (chromosome <= high)
                for chromosome in current
            ]
        )

    blacklist_good = [np.asarray(chromosome) > 0 for chromosome in blacklist]
    target_good = [
        np.isfinite(signal)
        & (signal >= output_min)
        & (signal <= output_max)
        for signal in output_mode
    ]
    controls_good = [
        np.ones_like(mask, dtype=bool) for mask in blacklist_good
    ]
    for masks in control_masks:
        controls_good = [
            good & mask for good, mask in zip(controls_good, masks)
        ]
    combined_good = [
        blacklist_mask & target_mask & control_mask
        for blacklist_mask, target_mask, control_mask in zip(
            blacklist_good,
            target_good,
            controls_good,
        )
    ]
    bad_mask = chis.BinCurrentInverter(
        [mask.astype(np.int8) for mask in combined_good]
    )

    flag_current = chis.CurrentsZeroEnds(
        [chromosome + 1 for chromosome in output_mode],
        reso=CT_RESOLUTION,
        zeroends=FLAG_ZERO_ENDS,
    )
    _progress(
        f"{output_pair.pair_id}: flagging signal across "
        f"{len(autosome_ids)} autosomes."
    )
    flags = chis.CurrentFlagger(
        flag_current,
        reso=CT_RESOLUTION,
        window=FLAG_WINDOW,
        stride=1,
        flagmode=np.argmax,
        parallel=True,
    )
    _progress(f"{output_pair.pair_id}: converting flags to candidate sites.")
    markers = chis.SigCurrent2Markers(
        flags,
        select_BS_ids=autosome_ids,
        reso=CT_RESOLUTION,
        select=None,
        select_mode=None,
    )
    markers["val"] = markers["val"] - 1
    _progress(
        f"{output_pair.pair_id}: filtering {len(markers):,} candidate sites "
        "against controls and the blacklist."
    )
    filter_window_size = (
        parameters["output_size"]
        + parameters["translate_bp"] * (parameters["num_trans_wins"] - 1)
        + CT_RESOLUTION
    )
    bad_stage_masks = [
        [~mask for mask in stage_masks]
        for stage_masks in (blacklist_good, target_good, controls_good)
    ]
    _progress(
        f"{output_pair.pair_id}: applying whole-window filters to "
        f"{len(markers):,} candidate sites."
    )
    filtered_markers, stage_counts = _filter_markers_with_stage_qc(
        markers,
        bad_stage_masks,
        autosome_ids,
        filter_window_size,
    )
    _progress(
        f"{output_pair.pair_id}: {len(filtered_markers):,} candidates remain "
        "after blacklist, target, and control windows."
    )
    flag_dir = experiment_dir / "flag"
    figs_dir = (
        experiment_dir
        / "output"
        / "figures"
        / "data_preparation"
        / "site_selection"
    )
    tables_dir = (
        experiment_dir
        / "output"
        / "tables"
        / "data_preparation"
        / "site_selection"
    )
    flag_dir.mkdir(parents=True, exist_ok=True)
    _pickle_dump(flags, flag_dir / "flags.p")
    _pickle_dump(markers, flag_dir / "flag_markers.p")
    _pickle_dump(filtered_markers, flag_dir / "filtered_flag_markers.p")
    _pickle_dump(bad_mask, flag_dir / "bad_site_mask.p")
    with (flag_dir / "control_thresholds.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "output_signal_minimum": output_min,
                "output_signal_maximum": output_max,
                "target_percentile": target_percentile,
                "control_percentile": control_percentile,
                "control_thresholds": control_thresholds,
                "control_replicate_ids": list(controls_by_id),
            },
            file,
            indent=2,
        )
        file.write("\n")
    if filtered_markers.empty:
        _save_site_selection_qc(
            figs_dir,
            tables_dir,
            output_pair.pair_id,
            stage_counts,
            0,
            filter_window_size,
            target_percentile,
            control_percentile,
            output_min,
            output_max,
            control_thresholds,
        )
        raise ValueError(
            f"{output_pair.pair_id}: no flag sites remain after control and "
            "blacklist filtering."
        )

    output_transform = _load_vst(output_pair)
    _progress(
        f"{output_pair.pair_id}: selecting "
        f"{parameters['unique_sites']:,} sites from "
        f"{len(filtered_markers):,} filtered candidates."
    )
    transformed_values = np.asarray(
        output_transform(np.asarray(filtered_markers["val"])),
        dtype=np.float64,
    ).reshape(-1)
    if not np.all(np.isfinite(transformed_values)):
        raise ValueError(f"{output_pair.pair_id}: VST produced non-finite site values.")
    if parameters["unique_sites"] > len(filtered_markers):
        raise ValueError(
            f"unique_sites={parameters['unique_sites']} exceeds the "
            f"{len(filtered_markers)} filtered flag sites."
        )
    value_bins = bm.HistBinning(transformed_values, bins=1_000)
    selected_indices = bm.Harpoon(
        value_bins,
        select=parameters["unique_sites"],
    )
    if len(selected_indices) != parameters["unique_sites"]:
        raise ValueError(
            f"{output_pair.pair_id}: site selection returned "
            f"{len(selected_indices)} sites; expected {parameters['unique_sites']}."
        )
    selected_sites = chis.SortMarkers(
        filtered_markers.iloc[selected_indices].copy(),
        autosome_ids,
    )

    _pickle_dump(selected_sites, flag_dir / "selected_sites.p")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(transformed_values, bins=100, color="steelblue")
    axis.set(title="Filtered output-signal values", xlabel="VST value", ylabel="Sites")
    _save_figure(figure, figs_dir / "output_signal_distribution.png")
    _save_site_selection_qc(
        figs_dir,
        tables_dir,
        output_pair.pair_id,
        stage_counts,
        len(selected_sites),
        filter_window_size,
        target_percentile,
        control_percentile,
        output_min,
        output_max,
        control_thresholds,
    )

    counts = selected_sites["subseq"].value_counts().reindex(autosome_ids, fill_value=0)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(counts.index.astype(str), counts.to_numpy())
    axis.set(title="Selected sites by chromosome", xlabel="Chromosome", ylabel="Sites")
    axis.tick_params(axis="x", rotation=45)
    _save_figure(figure, figs_dir / "selected_sites_by_chromosome.png")
    _progress(
        f"{output_pair.pair_id}: selected {len(selected_sites):,} sites "
        "after control/blacklist filtering; flag outputs and figures saved."
    )
    return selected_sites


def main(argv: list[str] | None = None) -> int:
    if __package__:
        from . import experiment_context as exp_data
    else:
        import experiment_context as exp_data

    parser = argparse.ArgumentParser(
        description="Generate, filter, and select flag sites for one experiment."
    )
    parser.add_argument(
        "experiment_file",
        nargs="?",
        type=Path,
        default=exp_data.DEFAULT_EXPERIMENT_FILE,
        help=f"Experiment JSON file (default: {exp_data.DEFAULT_EXPERIMENT_FILE})",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=ENCODE_ROOT / "meta_info.txt",
        help="ENCODE metadata file used to resolve signal IDs and controls",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Parent directory for experiment outputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    for parameter in exp_data.PARAMETER_NAMES:
        parser.add_argument(
            f"--{parameter.replace('_', '-')}",
            type=int,
            help="Override this experiment parameter for site selection",
        )
    args = parser.parse_args(argv)

    try:
        experiment_file = args.experiment_file.resolve()
        experiment = exp_data.read_experiment_file(experiment_file)
        name = experiment.get("experiment_name")
        if isinstance(name, str):
            name = name.strip()
            experiment["experiment_name"] = name
        genome = experiment.get("genome")
        output_signal = experiment.get("output_signal")
        input_signals = experiment.get("input_signals", [])
        if not isinstance(name, str) or not exp_data.EXPERIMENT_NAME_PATTERN.fullmatch(name):
            raise ValueError("experiment_name must be a valid directory name.")
        if not isinstance(genome, str):
            raise ValueError(f"{name}: genome must be a non-empty string.")
        if not isinstance(output_signal, str) or not output_signal.strip():
            raise ValueError(f"{name}: output_signal must be a non-empty signal ID.")
        if input_signals is None:
            input_signals = []
        if not isinstance(input_signals, list) or any(
            not isinstance(signal, str) or not signal.strip()
            for signal in input_signals
        ):
            raise ValueError(f"{name}: input_signals must be a list of signal IDs.")

        _progress(f"{name}: resolving signal pairs and selection parameters.")
        metadata_rows = exp_data.read_metadata(args.metadata_file)
        cli_values = {
            parameter: getattr(args, parameter)
            for parameter in exp_data.PARAMETER_NAMES
        }
        parameters = exp_data.resolve_parameters(
            experiment,
            cli_values,
            has_input_signals=bool(input_signals),
        )
        target_percentile, control_percentile = resolve_site_percentiles(experiment)
        exp_data.write_experiment_file(experiment_file, experiment)
        pairs = exp_data.build_signal_pairs(metadata_rows)
        output_pair = exp_data.resolve_signal_pair(
            output_signal,
            metadata_rows,
            pairs,
        )
        input_pairs = [
            exp_data.resolve_signal_pair(signal, metadata_rows, pairs)
            for signal in input_signals
        ]
        autosome_ids, blacklist = _load_selection_genome_data(genome)
        experiment_dir = args.output_root / name
        select_sites(
            experiment_dir,
            output_pair,
            input_pairs,
            parameters,
            autosome_ids,
            blacklist,
            target_percentile,
            control_percentile,
        )
        _progress(f"{name}: site selection completed.")
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        print(f"site_selection: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
