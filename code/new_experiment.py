#!/usr/bin/env python3
"""Record experiment inputs for downstream processing."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "experiments.json"
SIGNAL_ID_COLUMNS = ("File name", "Replicate ID", "Pair ID")
PARAMETER_NAMES = (
    "output_size",
    "seq_size",
    "input_size",
    "translate_bp",
    "unique_sites",
)
EXPERIMENT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def _validate_field(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} cannot be empty.")
    if any(character in value for character in "\t\r\n"):
        raise ValueError(f"{name} cannot contain tabs or newlines.")
    return value


def load_signal_ids(metadata_path: Path) -> set[str]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ENCODE metadata file not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter="\t")
        missing = set(SIGNAL_ID_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{metadata_path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        signal_ids = {
            identifier.strip()
            for row in reader
            for column in SIGNAL_ID_COLUMNS
            for identifier in (row.get(column) or "").split(",")
            if identifier.strip()
        }
    return signal_ids


def _validate_signal_ids(
    input_signals: list[str],
    output_signal: str,
    valid_signal_ids: set[str],
) -> None:
    supplied_ids = [
        _validate_field(identifier, "Signal ID")
        for value in [*input_signals, output_signal]
        for identifier in value.split(",")
    ]
    unknown = list(dict.fromkeys(
        identifier for identifier in supplied_ids if identifier not in valid_signal_ids
    ))
    if unknown:
        raise ValueError(
            "Unknown signal ID(s): "
            + ", ".join(unknown)
            + ". Use a File name, Replicate ID, or Pair ID from meta_info.txt."
        )


def save_experiment(
    output_path: Path,
    experiment_name: str,
    genome: str,
    input_signals: list[str],
    output_signal: str,
    valid_signal_ids: set[str] | None = None,
    parameters: dict[str, int | None] | None = None,
) -> None:
    experiment_name = _validate_field(experiment_name, "Experiment name")
    if not EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name):
        raise ValueError(
            "Experiment name must start with a letter or digit and contain only "
            "letters, digits, spaces, '.', '_' or '-'."
        )
    genome = _validate_field(genome, "Genome")
    output_signal = _validate_field(output_signal, "Output signal")
    signals = [
        _validate_field(signal, "Input signal") for signal in input_signals
    ]
    if valid_signal_ids is not None:
        _validate_signal_ids(signals, output_signal, valid_signal_ids)
    experiment = {
        "experiment_name": experiment_name,
        "genome": genome,
        "input_signals": signals,
        "output_signal": output_signal,
        **{
            name: (parameters or {}).get(name)
            for name in PARAMETER_NAMES
        },
    }
    for name, value in (parameters or {}).items():
        if name not in PARAMETER_NAMES:
            raise ValueError(f"Unknown experiment parameter: {name}")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "translate_bp" else 1)
        ):
            requirement = "non-negative" if name == "translate_bp" else "positive"
            raise ValueError(f"{name} must be a {requirement} integer.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size:
        with output_path.open(encoding="utf-8") as existing_file:
            existing_data = json.load(existing_file)
        if isinstance(existing_data, dict) and existing_data.get(
            "experiment_name"
        ) == experiment_name:
            raise ValueError(
                f"Experiment {experiment_name!r} already exists in {output_path}."
            )
        raise ValueError(
            f"{output_path} already contains an experiment. Use --output-file "
            "to save this experiment to a separate JSON file."
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(experiment, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Save an experiment definition for downstream processing."
    )
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--genome", required=True, help="Genome name")
    parser.add_argument(
        "--input-signals",
        nargs="*",
        default=[],
        metavar="DATA_ID",
        help=(
            "Optional input signal IDs from meta_info.txt: file names, "
            "Replicate IDs, or Pair IDs"
        ),
    )
    parser.add_argument(
        "--output-signal",
        required=True,
        help=(
            "Output signal ID from meta_info.txt: a file name, Replicate ID, "
            "or Pair ID"
        ),
    )
    for name in PARAMETER_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=int,
            help=(
                "Processing parameter; sizes and unique-sites must be positive, "
                "translate-bp may be zero"
            ),
        )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "ENCODE" / "meta_info.txt",
        help="ENCODE metadata file used to validate signal IDs",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"JSON experiment file (default: {DEFAULT_OUTPUT_FILE})",
    )
    args = parser.parse_args(argv)

    try:
        valid_signal_ids = load_signal_ids(args.metadata_file)
        save_experiment(
            args.output_file,
            args.name,
            args.genome,
            args.input_signals,
            args.output_signal,
            valid_signal_ids,
            {
                name: getattr(args, name)
                for name in PARAMETER_NAMES
                if getattr(args, name) is not None
            },
        )
    except (OSError, ValueError, csv.Error) as error:
        print(f"new_experiment: error: {error}", file=sys.stderr)
        return 1
    print(f"Saved experiment to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
