"""Shared experiment configuration, metadata, and signal-pair helpers."""

from __future__ import annotations

import csv
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODE_ROOT = PROJECT_ROOT / "data" / "ENCODE"
DEFAULT_EXPERIMENT_FILE = PROJECT_ROOT / "data" / "experiments.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "experiments"
CT_RESOLUTION = 50
DEFAULT_NUM_TRANSLATION_WINDOWS = 3
PARAMETER_NAMES = (
    "output_size",
    "seq_size",
    "input_size",
    "translate_bp",
    "num_trans_wins",
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
        if value is None and name == "num_trans_wins":
            value = DEFAULT_NUM_TRANSLATION_WINDOWS
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
        "pair. Run merge_replicates.py first and use a target File name, "
        "Replicate ID, or Pair ID."
    )
