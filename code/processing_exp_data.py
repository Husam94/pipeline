#!/usr/bin/env python3
"""Select experiment sites, create sequence/signal packs, and generate splits."""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import json
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import poseigen_binmeths as bm
import poseigen_chisel.poseigen_chisel as chis
import poseigen_seaside.basics as se


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODE_ROOT = PROJECT_ROOT / "data" / "ENCODE"
DEFAULT_EXPERIMENT_FILE = PROJECT_ROOT / "data" / "experiments.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments"
CT_RESOLUTION = 50
FLAG_WINDOW = 2_000
FLAG_ZERO_ENDS = 5_000
NUM_TRANSLATION_WINDOWS = 3
SPLIT_PROPORTIONS = [0.55, 0.15, 0.15, 0.15]
SPLIT_SCORE_EPSILON = 1e-12
PARAMETER_NAMES = (
    "output_size",
    "seq_size",
    "input_size",
    "translate_bp",
    "unique_sites",
)
METADATA_COLUMNS = (
    "File name",
    "Replicate ID",
    "Pair ID",
    "Dataset",
    "Target",
    "Controls",
)
PAIR_ACCESSION_PATTERN = re.compile(r"ENCSR[A-Z0-9]+")
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")
GENOME_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class SignalPair:
    pair_id: str
    target_dataset: str
    control_dataset: str
    control_replicate_id: str


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
    return logs_dir / f"processing_exp_data_{file_stamp}.log"


def read_experiment_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Experiment JSON file not found: {path}")
    with path.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    if not isinstance(experiment, dict):
        raise ValueError(f"{path} must contain one JSON experiment object.")
    return experiment


def write_experiment_file(
    path: Path, experiment: dict[str, Any]
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(experiment, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def resolve_parameters(
    experiment: dict[str, Any],
    cli_values: dict[str, int | None],
    has_input_signals: bool,
) -> dict[str, int]:
    missing: list[str] = []
    values: dict[str, int] = {}
    for name in PARAMETER_NAMES:
        command_value = cli_values[name]
        value = command_value if command_value is not None else experiment.get(name)
        experiment[name] = value
        if value is None and name == "input_size" and not has_input_signals:
            continue
        if value is None:
            missing.append(name)
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        minimum = 0 if name == "translate_bp" else 1
        if value < minimum:
            requirement = "non-negative" if name == "translate_bp" else "positive"
            raise ValueError(f"{name} must be a {requirement} integer.")
        values[name] = value
    if missing:
        raise ValueError(
            "Missing processing parameter(s): "
            + ", ".join(missing)
            + ". Set them in the experiment JSON or provide the matching CLI options."
        )
    return values


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"ENCODE metadata file not found: {path}")
    with path.open(encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter="\t")
        missing = set(METADATA_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        return [dict(row) for row in reader]


def _split_ids(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _control_accessions(value: str | None) -> list[str]:
    return list(dict.fromkeys(PAIR_ACCESSION_PATTERN.findall(value or "")))


def _single_replicate_id(
    dataset: str, rows_by_dataset: dict[str, list[dict[str, str]]]
) -> str:
    identifiers = {
        identifier
        for row in rows_by_dataset.get(dataset, [])
        for identifier in _split_ids(row.get("Replicate ID"))
    }
    if len(identifiers) != 1:
        raise ValueError(
            f"Dataset {dataset} must have exactly one Replicate ID in meta_info.txt."
        )
    return next(iter(identifiers))


def build_signal_pairs(rows: list[dict[str, str]]) -> dict[str, SignalPair]:
    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dataset = (row.get("Dataset") or "").strip()
        if dataset:
            rows_by_dataset.setdefault(dataset, []).append(row)

    pairs: dict[str, SignalPair] = {}
    for row in rows:
        target_dataset = (row.get("Dataset") or "").strip()
        if not target_dataset or not (row.get("Target") or "").strip():
            continue
        pair_ids = _split_ids(row.get("Pair ID"))
        if not pair_ids:
            continue
        controls = _control_accessions(row.get("Controls"))
        if len(pair_ids) != len(controls):
            raise ValueError(
                f"Pair IDs for {row.get('File name')} do not align with Controls."
            )
        for pair_id, control_dataset in zip(pair_ids, controls):
            if control_dataset not in rows_by_dataset:
                raise ValueError(
                    f"{pair_id}: control dataset {control_dataset} is missing "
                    "from meta_info.txt."
                )
            control_replicate_id = _single_replicate_id(
                control_dataset, rows_by_dataset
            )
            pair = SignalPair(
                pair_id=pair_id,
                target_dataset=target_dataset,
                control_dataset=control_dataset,
                control_replicate_id=control_replicate_id,
            )
            existing = pairs.get(pair_id)
            if existing is not None and existing != pair:
                raise ValueError(
                    f"Pair ID {pair_id} refers to inconsistent metadata rows."
                )
            pairs[pair_id] = pair
    return pairs


def resolve_signal_pair(
    signal_id: str,
    rows: list[dict[str, str]],
    pairs: dict[str, SignalPair],
) -> SignalPair:
    signal_id = signal_id.strip()
    if signal_id in pairs:
        return pairs[signal_id]

    datasets = {
        (row.get("Dataset") or "").strip()
        for row in rows
        if (row.get("File name") or "").strip() == signal_id
        or signal_id in _split_ids(row.get("Replicate ID"))
    }
    datasets.discard("")
    matching_pairs = {
        pair.pair_id: pair
        for pair in pairs.values()
        if pair.target_dataset in datasets
    }
    if len(matching_pairs) == 1:
        return next(iter(matching_pairs.values()))
    if len(matching_pairs) > 1:
        raise ValueError(
            f"Signal ID {signal_id!r} maps to multiple target/control pairs; "
            "use a Pair ID from meta_info.txt."
        )
    if not datasets:
        raise ValueError(
            f"Signal ID {signal_id!r} was not found as a File name, Replicate ID, "
            "or Pair ID in meta_info.txt."
        )
    raise ValueError(
        f"Signal ID {signal_id!r} does not resolve to a processed target/control "
        "pair. Run processing_replicates.py first and use a target File name, "
        "Replicate ID, or Pair ID."
    )


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


def _load_control_mean(pair: SignalPair) -> list[np.ndarray]:
    return _load_current(
        _pair_file(ENCODE_ROOT / "merged", pair.control_replicate_id, "mean")
    )


def _load_vst(pair: SignalPair):
    return _pickle_load(
        _pair_file(PROJECT_ROOT / "data" / "ENCODE" / "vst", pair.pair_id, "vst_transform")
    )


def _percentile_bounds(values: np.ndarray, percentile: float) -> tuple[float, float]:
    valid = values[np.isfinite(values) & (values > 0)]
    if not len(valid):
        raise ValueError("Signal has no positive finite values for thresholding.")
    low, high = np.percentile(valid, [percentile, 100])
    return float(low), float(high)


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


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _progress(message: str) -> None:
    print(f"[processing_exp_data] {message}", flush=True)


def make_flags(
    experiment_dir: Path,
    output_pair: SignalPair,
    input_pairs: list[SignalPair],
    parameters: dict[str, int],
    autosome_ids: list[str],
    blacklist: list[np.ndarray],
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
    output_min, output_max = _percentile_bounds(output_values, 0)
    control_masks: list[list[np.ndarray]] = []
    control_thresholds: dict[str, dict[str, float]] = {}
    for replicate_id, current in controls_by_id.items():
        low, high = _percentile_bounds(np.vstack(current).reshape(-1), 5)
        control_thresholds[replicate_id] = {"minimum": low, "maximum": high}
        control_masks.append(
            [
                np.isfinite(chromosome)
                & (chromosome >= low)
                & (chromosome <= high)
                for chromosome in current
            ]
        )

    combined_good = [
        np.asarray(chromosome) > 0
        for chromosome in blacklist
    ]
    combined_good = [
        good
        & np.isfinite(signal)
        & (signal >= output_min)
        & (signal <= output_max)
        for good, signal in zip(combined_good, output_mode)
    ]
    for masks in control_masks:
        combined_good = [
            good & mask for good, mask in zip(combined_good, masks)
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
    filtered_markers = chis.MarkersFilter(
        markers,
        bad_mask,
        autosome_ids,
        select_BS_ids=None,
        reso=CT_RESOLUTION,
        exact=False,
        Msizes=[
            parameters["output_size"]
            + parameters["translate_bp"] * (NUM_TRANSLATION_WINDOWS - 1)
            + CT_RESOLUTION
        ],
        threshold=0,
        threshmode=np.sum,
        threshsign=np.equal,
    )
    if filtered_markers.empty:
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

    flag_dir = experiment_dir / "flag"
    figs_dir = flag_dir / "figs"
    flag_dir.mkdir(parents=True, exist_ok=True)
    _pickle_dump(flags, flag_dir / "flags.p")
    _pickle_dump(markers, flag_dir / "flag_markers.p")
    _pickle_dump(filtered_markers, flag_dir / "filtered_flag_markers.p")
    _pickle_dump(selected_sites, flag_dir / "selected_sites.p")
    _pickle_dump(bad_mask, flag_dir / "bad_site_mask.p")
    with (flag_dir / "control_thresholds.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "output_signal_minimum": output_min,
                "output_signal_maximum": output_max,
                "control_thresholds": control_thresholds,
                "control_replicate_ids": list(controls_by_id),
            },
            file,
            indent=2,
        )
        file.write("\n")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(transformed_values, bins=100, color="steelblue")
    axis.set(title="Filtered output-signal values", xlabel="VST value", ylabel="Sites")
    _save_figure(figure, figs_dir / "output_signal_distribution.png")

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
    translate_width = parameters["translate_bp"] * (NUM_TRANSLATION_WINDOWS - 1)
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
    pack_figs = pack_dir / "figs"
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
            num_trans=NUM_TRANSLATION_WINDOWS,
        )[0]
        translated = np.swapaxes(augmented, -1, -2)
        if len(translated) != len(markers) * NUM_TRANSLATION_WINDOWS:
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


def make_splits(
    experiment_dir: Path,
    markers: Any,
    bin_sizes: np.ndarray,
    split_candidates: int,
    split_cycles: int,
    seed: int | None,
) -> None:
    if "group" not in markers:
        raise ValueError(
            "Selected sites have no 'group' column required for grouped splits."
        )
    if not (experiment_dir / "packs" / "tpack_output_scores.p").is_file():
        raise FileNotFoundError("Missing translation-augmented output score pack.")
    translated_scores = _pickle_load(
        experiment_dir / "packs" / "tpack_output_scores.p"
    )
    translated_bind = _pickle_load(
        experiment_dir / "packs" / "tpack_output_bind.p"
    )
    horizontal_scores = np.asarray(
        chis.StackTransAugsHorizontal(
            translated_scores,
            num_shifts=NUM_TRANSLATION_WINDOWS,
        )
    )
    horizontal_bind = np.asarray(
        chis.StackTransAugsHorizontal(
            translated_bind,
            num_shifts=NUM_TRANSLATION_WINDOWS,
        )
    )
    groups = np.asarray(markers["group"])
    if len(groups) != len(markers) or len(horizontal_scores) != len(groups):
        raise ValueError("Grouped split inputs do not align with selected sites.")

    categories = np.unique(horizontal_bind)
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
        f"{experiment_dir.name}: searching {split_candidates:,} candidate "
        f"splits across {split_cycles} cycle(s)."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cycle in range(split_cycles):
            _progress(
                f"{experiment_dir.name}: starting split-search cycle "
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
                        f"{experiment_dir.name}: split cycle "
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
    expanded_split = bm.SplitExpander(
        1,
        NUM_TRANSLATION_WINDOWS,
        best_overall_split,
        ma=None,
    )
    split_dir = experiment_dir / "split"
    split_figs = split_dir / "figs"
    split_dir.mkdir(parents=True, exist_ok=True)
    _pickle_dump(best_overall_split, split_dir / "split.p")
    _pickle_dump(expanded_split, split_dir / "split_transaug.p")
    _write_json(
        {
            "best_score": best_overall_score,
            "cycle_scores": cycle_scores,
            "proportions": SPLIT_PROPORTIONS,
            "split_candidates_per_cycle": split_candidates,
            "split_cycles": split_cycles,
            "seed": seed,
        },
        split_dir / "split_metadata.json",
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        ["Train", "Stop", "Eval", "Test"],
        [len(indices) for indices in best_overall_split],
        color="slateblue",
    )
    axis.set(title="Selected sites by split", ylabel="Sites")
    _save_figure(figure, split_figs / "split_sizes.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(1, len(cycle_scores) + 1), cycle_scores, marker="o")
    axis.set(
        title="Best split score by cycle",
        xlabel="Cycle",
        ylabel="Weighted relative bin error",
    )
    _save_figure(figure, split_figs / "split_scores.png")
    _progress(f"{experiment_dir.name}: split files and figures saved to {split_dir}.")


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
    basic_ids, autosome_ids, blacklist, sequences = load_genome_data(genome)
    experiment_dir = args.output_root / name
    for folder in ("flag", "packs", "split"):
        (experiment_dir / folder / "figs").mkdir(parents=True, exist_ok=True)

    selected_sites = make_flags(
        experiment_dir,
        output_pair,
        input_pairs,
        parameters,
        autosome_ids,
        blacklist,
    )
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
    bin_sizes = _pickle_load(experiment_dir / "packs" / "output_bin_sizes.p")
    make_splits(
        experiment_dir,
        selected_sites,
        np.asarray(bin_sizes),
        args.split_candidates,
        args.split_cycles,
        args.seed,
    )
    _progress(f"Completed experiment {name!r}; outputs are in {experiment_dir}.")


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
            "Create filtered flag sites, sequence/signal packs, and grouped "
            "train/stop/eval/test splits for one experiment in a JSON file."
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
    parser.add_argument(
        "--split-candidates",
        type=int,
        default=10_000,
        help="Random split candidates per cycle (default: 10000, as in the notebook)",
    )
    parser.add_argument(
        "--split-cycles",
        type=int,
        default=10,
        help="Number of split-search cycles (default: 10, as in the notebook)",
    )
    parser.add_argument("--seed", type=int, help="Seed for reproducible split search")
    args = parser.parse_args(argv)

    if args.split_candidates <= 0 or args.split_cycles <= 0:
        parser.error("--split-candidates and --split-cycles must be positive.")

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
        print(f"processing_exp_data: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
