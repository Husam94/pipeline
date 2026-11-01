#!/usr/bin/env python3
"""Build sequence and signal packs from previously selected experiment sites."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any

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
        SignalPair,
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
        SignalPair,
        build_signal_pairs,
        read_experiment_file,
        read_metadata,
        resolve_parameters,
        resolve_signal_pair,
        write_experiment_file,
    )


class _Tee:
    def __init__(self, terminal, log_file) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _create_log_path(output_root: Path, experiment_name: str) -> Path:
    if not EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name):
        raise ValueError("Cannot create a log for an invalid experiment name.")
    logs_dir = output_root / experiment_name / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return logs_dir / f"pack_builder_{file_stamp}.log"


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


def _load_pair_mode(pair: SignalPair) -> list[np.ndarray]:
    return _load_current(
        _pair_file(ENCODE_ROOT / "betaprime", pair.pair_id, "gbp_mode")
    )


def _load_vst(pair: SignalPair):
    return _pickle_load(
        _pair_file(PROJECT_ROOT / "data" / "ENCODE" / "vst", pair.pair_id, "vst_transform")
    )


def load_genome_data(
    genome: str,
) -> tuple[list[str], list[str], list[np.ndarray], list[str]]:
    if not GENOME_NAME_PATTERN.fullmatch(genome):
        raise ValueError(
            "Genome name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'."
        )
    genome_root = PROJECT_ROOT / "data" / "genomes" / genome
    basic_ids, _ = _pickle_load(genome_root / f"{genome}_basic_id_sizes.p")
    sequences = _pickle_load(genome_root / f"{genome}_basic.p")
    blacklist = _pickle_load(genome_root / f"{genome}_basic_BLC_LR50.p")
    if len(basic_ids) < 4:
        raise ValueError(f"{genome}: basic chromosome list is too short.")
    autosome_ids = basic_ids[:-3]
    autosome_blacklist = blacklist[: len(autosome_ids)]
    if len(autosome_blacklist) != len(autosome_ids):
        raise ValueError(f"{genome}: blacklist is not aligned to basic chromosomes.")
    return basic_ids, autosome_ids, autosome_blacklist, sequences


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _progress(message: str) -> None:
    print(f"[pack_builder] {message}", flush=True)


def _normalized_scores(
    signal_pair: SignalPair,
    markers: Any,
    basic_ids: list[str],
    autosome_ids: list[str],
    size: int,
    center_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    signal_pack = chis.SigCurrent2Pack(
        _load_pair_mode(signal_pair),
        markers,
        basic_ids,
        select_BS_ids=autosome_ids,
        Vsize=size,
        reso=CT_RESOLUTION,
    )
    transformed = np.asarray(_load_vst(signal_pair)(signal_pack))
    if not np.all(np.isfinite(transformed)):
        raise ValueError(f"{signal_pair.pair_id}: VST pack has non-finite values.")
    center = (
        float(np.median(transformed))
        if center_mode == "median"
        else float(np.min(transformed))
    )
    scale = float(np.max(np.abs(transformed - center)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"{signal_pair.pair_id}: cannot normalize a constant or invalid signal pack."
        )
    return (transformed - center) / scale, transformed


def _tri_pack_arrays(packs: dict[str, np.ndarray]) -> list[np.ndarray]:
    names = list(packs)
    if not names or names[0] not in {"pack_seq_ohe", "tpack_seq_ohe"}:
        raise ValueError("Sequence pack must be first when constructing tri-packs.")
    tri = [np.expand_dims(packs[names[0]], axis=1)]
    tri.extend(
        np.expand_dims(np.swapaxes(packs[name], -1, -2), axis=-1)
        for name in names[1:]
    )
    return tri


def make_packs(
    experiment_dir: Path,
    output_pair: SignalPair,
    input_pairs: list[SignalPair],
    markers: Any,
    basic_ids: list[str],
    autosome_ids: list[str],
    sequences: list[str],
    parameters: dict[str, int],
) -> None:
    translate_width = parameters["translate_bp"] * (
        parameters["num_trans_wins"] - 1
    )
    output_width = parameters["output_size"] + translate_width
    sequence_width = parameters["seq_size"] + translate_width

    _progress(
        f"{output_pair.pair_id}: creating sequence pack "
        f"for {len(markers):,} selected sites."
    )
    packs: dict[str, np.ndarray] = {
        "pack_seq_ohe": np.asarray(
            chis.SeqCurrent2Pack(
                sequences,
                markers,
                basic_ids,
                select_BS_ids=None,
                Vsize=sequence_width,
                seqmode=[chis.Seqs2OHE, {}],
                stranded=False,
            ),
            dtype=np.float32,
        )
    }
    if len(packs["pack_seq_ohe"]) != len(markers):
        raise ValueError(
            "Genome sequence packing dropped selected sites; reduce seq_size or "
            "translate_bp, or inspect chromosome boundaries."
        )

    output_scores, output_transformed = _normalized_scores(
        output_pair,
        markers,
        basic_ids,
        autosome_ids,
        output_width,
        "median",
    )
    packs["pack_output_scores"] = output_scores
    _progress(f"{output_pair.pair_id}: packing output signal and parameters.")
    output_exp = _pickle_load(
        _pair_file(ENCODE_ROOT / "betaprime", output_pair.pair_id, "gbp_exp")
    )
    packs["pack_output_exp"] = np.asarray(
        chis.SigCurrent2Pack(
            output_exp,
            markers,
            basic_ids,
            select_BS_ids=autosome_ids,
            Vsize=output_width,
            reso=CT_RESOLUTION,
        )
    )
    if len(packs["pack_output_exp"]) != len(markers):
        raise ValueError("Output beta-prime parameter pack does not align with sites.")

    if input_pairs:
        input_width = parameters["input_size"] + translate_width
        input_score_packs = []
        input_transformed = []
        for index, pair in enumerate(input_pairs, start=1):
            _progress(
                f"{output_pair.pair_id}: packing input signal "
                f"{index}/{len(input_pairs)} ({pair.pair_id})."
            )
            packed_scores, transformed = _normalized_scores(
                pair,
                markers,
                basic_ids,
                autosome_ids,
                input_width,
                "minimum",
            )
            input_score_packs.append(packed_scores)
            input_transformed.append(transformed.reshape(-1))
        packs["pack_input_scores"] = np.concatenate(input_score_packs, axis=-1)
        if len(packs["pack_input_scores"]) != len(markers):
            raise ValueError("Input signal packs do not align with selected sites.")
    else:
        input_transformed = []

    transform_mode = [se.tanh, {}]
    reverse_transform_mode = [se.revtanh, {}]
    bin_options = {
        "mode": transform_mode,
        "geometric": False,
        "obs_min": 50,
        "bins_min": 10,
        "bins_max": 100,
        "bins_increm": 2,
        "scalefax_min": 1,
        "scalefax_increm": 0.01,
        "scalefax_max": 3,
    }
    _progress(f"{output_pair.pair_id}: optimizing output-signal bins.")
    bin_optimization = bm.BinOpt(packs["pack_output_scores"], **bin_options)
    if bin_optimization is None:
        raise ValueError(f"{output_pair.pair_id}: could not optimize output bins.")
    number_of_bins, scale_factor = bin_optimization
    bin_ranges = bm.ObjUni(
        packs["pack_output_scores"],
        scalefactor=scale_factor,
        mode=transform_mode,
        mode_rev=reverse_transform_mode,
        bins=number_of_bins,
    )
    bin_sizes = bm.UniProp(bin_ranges)
    packs["pack_output_bind"] = bm.Binner(
        packs["pack_output_scores"],
        uni=bin_ranges,
    )
    packs = {
        "pack_seq_ohe": packs["pack_seq_ohe"],
        "pack_output_scores": packs["pack_output_scores"],
        "pack_output_exp": packs["pack_output_exp"],
        "pack_output_bind": packs["pack_output_bind"],
        **(
            {"pack_input_scores": packs["pack_input_scores"]}
            if "pack_input_scores" in packs
            else {}
        ),
    }
    for name, pack in packs.items():
        if len(pack) != len(markers):
            raise ValueError(
                f"{name} contains {len(pack)} rows for {len(markers)} selected sites."
            )
    _pickle_dump(bin_ranges, experiment_dir / "packs" / "output_bin_ranges.p")
    _pickle_dump(bin_sizes, experiment_dir / "packs" / "output_bin_sizes.p")

    pack_dir = experiment_dir / "packs"
    pack_figs = (
        experiment_dir
        / "output"
        / "figures"
        / "data_preparation"
        / "pack_builder"
    )
    pack_dir.mkdir(parents=True, exist_ok=True)
    for name, pack in packs.items():
        _pickle_dump(pack, pack_dir / f"{name}.p")
    _write_json(
        {
            "output_pair_id": output_pair.pair_id,
            "input_pair_ids": [pair.pair_id for pair in input_pairs],
            "pack_names": list(packs),
        },
        pack_dir / "pack_manifest.json",
    )

    pack_sizes = {
        "pack_seq_ohe": parameters["seq_size"],
        "pack_output_scores": parameters["output_size"],
        "pack_output_exp": parameters["output_size"],
        "pack_output_bind": parameters["output_size"],
    }
    pack_resolutions = {
        name: (1 if name == "pack_seq_ohe" else CT_RESOLUTION)
        for name in packs
    }
    if input_pairs:
        pack_sizes["pack_input_scores"] = parameters["input_size"]
    translated_packs: dict[str, np.ndarray] = {}
    for index, (name, pack) in enumerate(packs.items(), start=1):
        _progress(
            f"{output_pair.pair_id}: translation-augmenting pack "
            f"{index}/{len(packs)} ({name})."
        )
        transposed = np.swapaxes(pack, -1, -2)
        augmented = chis.MakeTransAugs(
            [transposed],
            resos=[pack_resolutions[name]],
            slice_size=[pack_sizes[name]],
            slice_mode=None,
            num_trans=parameters["num_trans_wins"],
        )[0]
        translated = np.swapaxes(augmented, -1, -2)
        if len(translated) != len(markers) * parameters["num_trans_wins"]:
            raise ValueError(
                f"{name}: translation augmentation changed the expected site count."
            )
        translated_packs[f"t{name}"] = translated
        _pickle_dump(translated, pack_dir / f"t{name}.p")

    _pickle_dump(_tri_pack_arrays(packs), pack_dir / "tri_packs.p")
    _pickle_dump(
        _tri_pack_arrays(translated_packs),
        pack_dir / "tri_tpacks.p",
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(output_transformed.reshape(-1), bins=50, color="darkorange")
    axis.set(title="Output signal pack values", xlabel="VST value", ylabel="Values")
    _save_figure(figure, pack_figs / "output_signal_pack.png")
    if input_transformed:
        figure, axis = plt.subplots(figsize=(8, 5))
        for pair, values in zip(input_pairs, input_transformed):
            axis.hist(values, bins=50, alpha=0.5, label=pair.pair_id)
        axis.set(title="Input signal pack values", xlabel="VST value", ylabel="Values")
        axis.legend()
        _save_figure(figure, pack_figs / "input_signal_packs.png")
    _progress(
        f"{output_pair.pair_id}: saved {len(packs)} signal/sequence packs "
        "and associated figures."
    )


def process_experiment(
    experiment: dict[str, Any],
    experiment_file: Path,
    metadata_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    name = experiment.get("experiment_name")
    if isinstance(name, str):
        name = name.strip()
        experiment["experiment_name"] = name
    genome = experiment.get("genome")
    output_signal = experiment.get("output_signal")
    input_signals = experiment.get("input_signals", [])
    if not isinstance(name, str) or not EXPERIMENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "experiment_name must start with a letter or digit and contain only "
            "letters, digits, spaces, '.', '_' or '-'."
        )
    if not isinstance(genome, str) or not genome.strip():
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

    _progress(f"Starting experiment {name!r} using genome {genome}.")
    cli_values = {
        parameter: getattr(args, parameter)
        for parameter in PARAMETER_NAMES
    }
    try:
        parameters = resolve_parameters(
            experiment,
            cli_values,
            has_input_signals=bool(input_signals),
        )
    except ValueError:
        write_experiment_file(experiment_file, experiment)
        raise
    write_experiment_file(experiment_file, experiment)
    _progress(f"{name}: processing parameters resolved and saved to {experiment_file}.")

    pairs = build_signal_pairs(metadata_rows)
    output_pair = resolve_signal_pair(output_signal, metadata_rows, pairs)
    input_pairs = [
        resolve_signal_pair(signal, metadata_rows, pairs)
        for signal in input_signals
    ]
    _progress(
        f"{name}: resolved output {output_signal} to Pair ID "
        f"{output_pair.pair_id}; {len(input_pairs)} input signal(s)."
    )
    if not input_pairs:
        _progress(f"{name}: no input signals specified; continuing without input packs.")
    _progress(f"{name}: loading prepared genome resources for {genome}.")
    basic_ids, autosome_ids, _, sequences = load_genome_data(genome)
    experiment_dir = args.output_root / name
    selected_sites_path = experiment_dir / "flag" / "selected_sites.p"
    if not selected_sites_path.is_file():
        raise FileNotFoundError(
            f"{name}: selected sites not found at {selected_sites_path}; "
            "run site_selection.py first."
        )
    selected_sites = _pickle_load(selected_sites_path)
    if not hasattr(selected_sites, "columns") or selected_sites.empty:
        raise ValueError(f"{name}: selected sites are missing or empty.")
    make_packs(
        experiment_dir,
        output_pair,
        input_pairs,
        selected_sites,
        basic_ids,
        autosome_ids,
        sequences,
        parameters,
    )
    _progress(f"Completed pack generation for {name!r}; outputs are in {experiment_dir}.")


def run_experiment_with_log(
    experiment: dict[str, Any],
    experiment_file: Path,
    metadata_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> Path:
    name = experiment.get("experiment_name")
    if not isinstance(name, str):
        raise ValueError("Experiment must have a valid experiment_name.")
    name = name.strip()
    log_path = _create_log_path(args.output_root, name)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Started: {_timestamp()}\n")
        log_file.flush()
        stdout_tee = _Tee(sys.stdout, log_file)
        stderr_tee = _Tee(sys.stderr, log_file)
        try:
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                process_experiment(
                    experiment,
                    experiment_file,
                    metadata_rows,
                    args,
                )
                _progress(f"Run log saved to {log_path}.")
        except (OSError, ValueError, csv.Error, RuntimeError) as error:
            log_file.write(f"[{_timestamp()}] ERROR: {error}\n")
            log_file.flush()
            raise
        log_file.write(f"Finished: {_timestamp()}\n")
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build sequence and signal packs from selected sites for one "
            "experiment in a JSON file."
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
    for parameter in PARAMETER_NAMES:
        parser.add_argument(
            f"--{parameter.replace('_', '-')}",
            type=int,
            help=(
                "Override this value for the experiment; positive "
                "integers are required except translate-bp, which may be zero"
            ),
        )
    args = parser.parse_args(argv)

    try:
        _progress(f"Reading experiment definitions from {args.experiment_file}.")
        experiment = read_experiment_file(args.experiment_file)
        _progress(
            f"Loaded experiment {experiment.get('experiment_name', '<unnamed>')!r}."
        )
        metadata_rows = read_metadata(args.metadata_file)
        run_experiment_with_log(
            experiment,
            args.experiment_file,
            metadata_rows,
            args,
        )
        _progress("Experiment processing completed.")
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        print(f"pack_builder: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
