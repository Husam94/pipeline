#!/usr/bin/env python3
"""Prepare genome, ENCODE, replicate, and experiment data for model fitting."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from data_preparation.download_encode import (
    FILE_ACCESSION_PATTERN,
    ensure_metadata_columns,
)
from data_preparation.experiment_context import (
    DEFAULT_OUTPUT_ROOT,
    GENOME_NAME_PATTERN,
    build_signal_pairs,
    read_experiment_file,
    read_metadata,
    resolve_signal_pair,
    write_experiment_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_SCRIPT_DIR = Path(__file__).resolve().parent / "data_preparation"
ENCODE_ROOT = PROJECT_ROOT / "data" / "ENCODE"
HG38_FASTA_URL = (
    "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/"
    "@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
)
REQUIRED_METADATA_COLUMNS = {
    "File format",
    "Mapped run type",
    "predictd",
}
CONTROL_DATASET_PATTERN = re.compile(r"ENCSR[A-Z0-9]+")


def _experiment_inputs(experiment: dict[str, Any]) -> tuple[str, list[str], str]:
    genome = experiment.get("genome")
    output_signal = experiment.get("output_signal")
    input_signals = experiment.get("input_signals", [])
    if not isinstance(genome, str) or not GENOME_NAME_PATTERN.fullmatch(genome):
        raise ValueError(
            "The experiment genome must start with a letter or digit and contain "
            "only letters, digits, '.', '_' or '-'."
        )
    if not isinstance(output_signal, str) or not output_signal.strip():
        raise ValueError("The experiment must contain a non-empty output_signal.")
    if input_signals is None:
        input_signals = []
    if not isinstance(input_signals, list) or any(
        not isinstance(signal, str) or not signal.strip()
        for signal in input_signals
    ):
        raise ValueError("Experiment input_signals must be a list of signal IDs.")
    return genome, input_signals, output_signal


def _experiment_genome(experiment: dict[str, Any]) -> str:
    genome = experiment.get("genome")
    if not isinstance(genome, str) or not GENOME_NAME_PATTERN.fullmatch(genome):
        raise ValueError(
            "The experiment genome must start with a letter or digit and contain "
            "only letters, digits, '.', '_' or '-'."
        )
    return genome


def _is_file_accession(value: Any) -> bool:
    return isinstance(value, str) and FILE_ACCESSION_PATTERN.fullmatch(value) is not None


def _uses_raw_accessions(experiment: dict[str, Any]) -> bool:
    if any(
        name in experiment
        for name in ("input_accessions", "output_accessions", "control_accessions")
    ):
        return True
    output_signal = experiment.get("output_signal")
    if isinstance(output_signal, list):
        return True
    input_signals = experiment.get("input_signals", [])
    return isinstance(input_signals, list) and bool(input_signals) and all(
        _is_file_accession(signal) for signal in input_signals
    )


def _accession_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not _is_file_accession(accession) for accession in value
    ):
        raise ValueError(
            f"{name} must be a list of ENCODE file accessions "
            "(ENCFF followed by letters or digits)."
        )
    return list(dict.fromkeys(value))


def _raw_accession_lists(
    experiment: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    input_accessions = experiment.get("input_accessions")
    if input_accessions is None:
        input_signals = experiment.get("input_signals", [])
        if input_signals is None:
            input_signals = []
        if not isinstance(input_signals, list) or any(
            not _is_file_accession(accession) for accession in input_signals
        ):
            raise ValueError(
                "Use input_accessions for raw ENCODE IDs; input_signals is reserved "
                "for generated signal IDs."
            )
        input_accessions = input_signals

    output_accessions = experiment.get("output_accessions")
    if output_accessions is None:
        output_signal = experiment.get("output_signal")
        if isinstance(output_signal, list):
            output_accessions = output_signal
        elif _is_file_accession(output_signal):
            output_accessions = [output_signal]

    input_accessions = _accession_list(input_accessions, "input_accessions")
    output_accessions = _accession_list(output_accessions, "output_accessions")
    control_accessions = _accession_list(
        experiment.get("control_accessions"), "control_accessions"
    )
    if not output_accessions:
        raise ValueError("Provide at least one accession in output_accessions.")
    return input_accessions, output_accessions, control_accessions


def _resolve_accessions(
    signals: list[str], metadata_rows: list[dict[str, str]]
) -> list[str]:
    if not metadata_rows:
        raise ValueError("ENCODE metadata contains no file rows.")
    missing_columns = REQUIRED_METADATA_COLUMNS.difference(metadata_rows[0])
    if missing_columns:
        raise ValueError(
            "ENCODE metadata is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    signal_pairs = build_signal_pairs(metadata_rows)
    datasets: set[str] = set()
    for signal in dict.fromkeys(signals):
        pair = resolve_signal_pair(signal, metadata_rows, signal_pairs)
        datasets.update((pair.target_dataset, pair.control_dataset))

    while True:
        related_controls = {
            pair.control_dataset
            for pair in signal_pairs.values()
            if pair.target_dataset in datasets
        }
        additions = related_controls.difference(datasets)
        if not additions:
            break
        datasets.update(additions)

    accessions = [
        (row.get("File name") or "").strip()
        for row in metadata_rows
        if (row.get("Dataset") or "").strip() in datasets
        and (row.get("File name") or "").strip()
    ]
    accessions = list(dict.fromkeys(accessions))
    if not accessions:
        raise ValueError("No ENCODE files found for the experiment signals.")
    return accessions


def _read_metadata_if_present(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        return []
    return read_metadata(metadata_path)


def _raw_file_path(accession: str, row: dict[str, str]) -> Path | None:
    file_format = (row.get("File format") or "").strip()
    normalized_format = file_format.casefold()
    if normalized_format not in {"bam", "bigwig"}:
        raise ValueError(
            f"{accession}: unsupported ENCODE file format {file_format!r}; "
            "expected BAM or bigWig."
        )
    matches = [
        path
        for path in (ENCODE_ROOT / "raw").glob(f"{accession}.*")
        if path.is_file()
        and path.name[len(accession) + 1 :].casefold() == normalized_format
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{accession}: multiple raw files found: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def _accessions_to_download(
    accessions: list[str], metadata_rows: list[dict[str, str]]
) -> list[str]:
    rows_by_accession = {
        (row.get("File name") or "").strip(): row
        for row in metadata_rows
        if (row.get("File name") or "").strip()
    }
    return [
        accession
        for accession in accessions
        if accession not in rows_by_accession
        or _raw_file_path(accession, rows_by_accession[accession]) is None
    ]


def _missing_raw_accessions(
    accessions: list[str], metadata_rows: list[dict[str, str]]
) -> list[str]:
    rows_by_accession = {
        (row.get("File name") or "").strip(): row
        for row in metadata_rows
        if (row.get("File name") or "").strip()
    }
    missing: list[str] = []
    for accession in accessions:
        row = rows_by_accession.get(accession)
        if row is None:
            raise ValueError(
                f"{accession} is missing from {ENCODE_ROOT / 'meta_info.txt'}."
            )
        if _raw_file_path(accession, row) is None:
            missing.append(accession)
    return missing


def _validate_raw_accessions(
    input_accessions: list[str],
    output_accessions: list[str],
    control_accessions: list[str],
    metadata_rows: list[dict[str, str]],
) -> None:
    rows_by_accession = {
        (row.get("File name") or "").strip(): row
        for row in metadata_rows
        if (row.get("File name") or "").strip()
    }
    raw_accessions = set(input_accessions + output_accessions + control_accessions)
    target_accessions = list(dict.fromkeys(input_accessions + output_accessions))
    target_datasets: dict[str, set[str]] = {}
    for accession in target_accessions:
        row = rows_by_accession.get(accession)
        if row is None:
            raise ValueError(f"No ENCODE metadata was found for {accession}.")
        dataset = (row.get("Dataset") or "").strip()
        if not dataset or not (row.get("Target") or "").strip():
            raise ValueError(
                f"{accession} must identify a target dataset, not a control file."
            )
        controls = set(CONTROL_DATASET_PATTERN.findall(row.get("Controls") or ""))
        if not controls:
            raise ValueError(
                f"{accession} has no control dataset in ENCODE metadata."
            )
        target_datasets.setdefault(dataset, set()).update(controls)

    output_datasets = {
        (rows_by_accession[accession].get("Dataset") or "").strip()
        for accession in output_accessions
    }
    if len(output_datasets) != 1:
        raise ValueError(
            "All output_accessions must belong to one target dataset; "
            "the experiment processor supports one output pair."
        )
    output_dataset = next(iter(output_datasets))
    if len(target_datasets[output_dataset]) != 1:
        raise ValueError(
            f"Output dataset {output_dataset} must have exactly one control dataset."
        )

    expected_controls = set().union(*target_datasets.values())
    listed_control_datasets: set[str] = set()
    for accession in control_accessions:
        row = rows_by_accession.get(accession)
        if row is None:
            raise ValueError(f"No ENCODE metadata was found for {accession}.")
        if (row.get("Target") or "").strip():
            raise ValueError(
                f"{accession} is a target file; control_accessions must contain "
                "control files."
            )
        dataset = (row.get("Dataset") or "").strip()
        if not dataset:
            raise ValueError(f"{accession} has no dataset in ENCODE metadata.")
        listed_control_datasets.add(dataset)

    missing_controls = expected_controls.difference(listed_control_datasets)
    extra_controls = listed_control_datasets.difference(expected_controls)
    if missing_controls:
        raise ValueError(
            "List the matching ENCODE control file accessions in "
            "control_accessions. Missing control dataset(s): "
            + ", ".join(sorted(missing_controls))
        )
    if extra_controls:
        raise ValueError(
            "control_accessions contains dataset(s) not referenced by the listed "
            "target accessions: "
            + ", ".join(sorted(extra_controls))
        )

    dataset_accessions: dict[str, set[str]] = {}
    for accession in raw_accessions:
        row = rows_by_accession.get(accession)
        if row is None:
            raise ValueError(f"No ENCODE metadata was found for {accession}.")
        dataset = (row.get("Dataset") or "").strip()
        dataset_accessions.setdefault(dataset, set()).add(accession)

    for dataset in set(target_datasets).union(expected_controls):
        listed = dataset_accessions.get(dataset, set())
        known_files = {
            (row.get("File name") or "").strip()
            for row in metadata_rows
            if (row.get("Dataset") or "").strip() == dataset
            and (row.get("File name") or "").strip()
        }
        unlisted = known_files.difference(raw_accessions)
        if unlisted:
            raise ValueError(
                f"Dataset {dataset} has ENCODE file(s) not listed in the "
                "experiment's accession fields: "
                + ", ".join(sorted(unlisted))
            )
        if len(listed) < 2:
            raise ValueError(
                f"Dataset {dataset} needs at least two listed replicate files "
                "for replicate processing."
            )


def _generated_signal_ids(
    input_accessions: list[str],
    output_accessions: list[str],
    metadata_rows: list[dict[str, str]],
) -> tuple[list[str], str]:
    pairs = build_signal_pairs(metadata_rows)
    rows_by_accession = {
        (row.get("File name") or "").strip(): row
        for row in metadata_rows
        if (row.get("File name") or "").strip()
    }
    output_pair_ids = {
        resolve_signal_pair(accession, metadata_rows, pairs).pair_id
        for accession in output_accessions
    }
    if len(output_pair_ids) != 1:
        raise ValueError(
            "output_accessions must resolve to exactly one target/control pair."
        )

    input_datasets = list(
        dict.fromkeys(
            (rows_by_accession[accession].get("Dataset") or "").strip()
            for accession in input_accessions
        )
    )
    input_pair_ids: list[str] = []
    for dataset in input_datasets:
        dataset_pair_ids = [
            pair.pair_id for pair in pairs.values() if pair.target_dataset == dataset
        ]
        if not dataset_pair_ids:
            raise ValueError(
                f"Input dataset {dataset} did not resolve to a target/control pair."
            )
        input_pair_ids.extend(dataset_pair_ids)
    return list(dict.fromkeys(input_pair_ids)), next(iter(output_pair_ids))


def _missing_predictd_accessions(
    accessions: list[str], metadata_rows: list[dict[str, str]]
) -> list[str]:
    rows_by_accession = {
        (row.get("File name") or "").strip(): row
        for row in metadata_rows
        if (row.get("File name") or "").strip()
    }
    return [
        accession
        for accession in accessions
        if accession in rows_by_accession
        and (rows_by_accession[accession].get("File format") or "")
        .strip()
        .casefold()
        == "bam"
        and (rows_by_accession[accession].get("Mapped run type") or "")
        .strip()
        .casefold()
        == "single-ended"
        and not (rows_by_accession[accession].get("predictd") or "").strip()
    ]


def _genome_resources(genome: str) -> tuple[Path, ...]:
    genome_root = PROJECT_ROOT / "data" / "genomes" / genome
    return (
        genome_root / f"{genome}_id_sizes.p",
        genome_root / f"{genome}_basic.p",
        genome_root / f"{genome}_basic_id_sizes.p",
        genome_root / f"{genome}_basic_BLC_LR50.p",
    )


def _run_script(script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(PREPARATION_SCRIPT_DIR / script), *arguments]
    print(f"Starting {script}...", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{script} exited with status {result.returncode}."
        )
    print(f"Finished {script}.", flush=True)


def _write_accessions(path: Path, accessions: list[str]) -> None:
    path.write_text("\n".join(accessions) + "\n", encoding="utf-8")


def _publish_replicate_qc(
    experiment: dict[str, Any],
    experiment_dir: Path,
    metadata_rows: list[dict[str, str]],
) -> None:
    _, input_signals, output_signal = _experiment_inputs(experiment)
    pairs = build_signal_pairs(metadata_rows)
    pair_ids = {
        resolve_signal_pair(signal_id, metadata_rows, pairs).pair_id
        for signal_id in [*input_signals, output_signal]
    }
    figure_dir = (
        experiment_dir
        / "output"
        / "figures"
        / "data_preparation"
        / "replicate_qc"
    )
    table_dir = (
        experiment_dir
        / "output"
        / "tables"
        / "data_preparation"
        / "replicate_qc"
    )
    for pair_id in sorted(pair_ids):
        outputs = (
            (
                ENCODE_ROOT
                / "replicate_qc"
                / "figs"
                / f"{pair_id}_target_replicate_concordance.png",
                figure_dir,
            ),
            (
                ENCODE_ROOT
                / "vst"
                / "figs"
                / f"{pair_id}_vst_transform.png",
                figure_dir,
            ),
            (
                ENCODE_ROOT
                / "vst"
                / "figs"
                / f"{pair_id}_vst_scale.png",
                figure_dir,
            ),
            (
                ENCODE_ROOT
                / "vst"
                / "figs"
                / f"{pair_id}_vst_spread.png",
                figure_dir,
            ),
            (
                ENCODE_ROOT
                / "replicate_qc"
                / f"{pair_id}_target_replicate_concordance.json",
                table_dir,
            ),
        )
        for source, destination_dir in outputs:
            if not source.is_file():
                raise FileNotFoundError(
                    f"Expected replicate QC output is missing: {source}"
                )
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_dir / source.name)


def prepare_data(
    experiment_file: Path,
    genome_url: str | None,
    predictd_file: Path | None,
    redo_splits: bool = False,
    split_candidates: int = 10_000,
    split_cycles: int = 10,
    seed: int | None = None,
) -> None:
    print(f"Reading experiment definition from {experiment_file}.", flush=True)
    experiment = read_experiment_file(experiment_file)
    genome = _experiment_genome(experiment)
    experiment_name = experiment.get("experiment_name", "<unnamed>")
    print(
        f"Preparing experiment {experiment_name!r} with genome {genome!r}.",
        flush=True,
    )
    metadata_path = ENCODE_ROOT / "meta_info.txt"
    ensure_metadata_columns(metadata_path)
    raw_accessions = _uses_raw_accessions(experiment)
    input_accessions: list[str] = []
    output_accessions: list[str] = []
    control_accessions: list[str] = []

    if raw_accessions:
        input_accessions, output_accessions, control_accessions = (
            _raw_accession_lists(experiment)
        )
        accessions = list(
            dict.fromkeys(
                input_accessions + output_accessions + control_accessions
            )
        )
        print(
            f"Checking metadata and downloads for {len(accessions)} listed "
            "ENCODE accession(s).",
            flush=True,
        )
        metadata_rows = _read_metadata_if_present(metadata_path)
        missing_accessions = _accessions_to_download(accessions, metadata_rows)
        if missing_accessions:
            print(
                f"Downloading or retrieving metadata for "
                f"{len(missing_accessions)} listed accession(s).",
                flush=True,
            )
            with tempfile.TemporaryDirectory(
                prefix="pipeline-data-prep-"
            ) as temp_dir:
                accession_file = Path(temp_dir) / "listed_accessions.txt"
                _write_accessions(accession_file, missing_accessions)
                _run_script(
                    "download_encode.py",
                    ["--input-file", str(accession_file)],
                )
        metadata_rows = read_metadata(metadata_path)
        rows_by_accession = {
            (row.get("File name") or "").strip(): row
            for row in metadata_rows
            if (row.get("File name") or "").strip()
        }
        missing_metadata = [
            accession for accession in accessions
            if accession not in rows_by_accession
        ]
        if missing_metadata:
            raise RuntimeError(
                "ENCODE metadata was not retrieved for: "
                + ", ".join(missing_metadata)
            )
        missing_raw = _missing_raw_accessions(accessions, metadata_rows)
        if missing_raw:
            raise RuntimeError(
                "ENCODE files are still missing after download: "
                + ", ".join(missing_raw)
            )
        _validate_raw_accessions(
            input_accessions,
            output_accessions,
            control_accessions,
            metadata_rows,
        )
        print(
            f"Validated {len(input_accessions)} input, "
            f"{len(output_accessions)} output, and "
            f"{len(control_accessions)} control accession(s).",
            flush=True,
        )
    else:
        input_signals, output_signal = _experiment_inputs(experiment)
        print(f"Resolving experiment signals using {metadata_path}.", flush=True)
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"ENCODE metadata not found: {metadata_path}. It is needed to "
                "resolve the experiment's signal IDs before downloading data."
            )
        metadata_rows = read_metadata(metadata_path)
        accessions = _resolve_accessions(
            [*input_signals, output_signal],
            metadata_rows,
        )
        missing_accessions = _missing_raw_accessions(accessions, metadata_rows)
        if missing_accessions:
            print(
                f"Downloading {len(missing_accessions)} missing ENCODE file(s).",
                flush=True,
            )
            with tempfile.TemporaryDirectory(
                prefix="pipeline-data-prep-"
            ) as temp_dir:
                accession_file = Path(temp_dir) / "missing_accessions.txt"
                _write_accessions(accession_file, missing_accessions)
                _run_script(
                    "download_encode.py",
                    ["--input-file", str(accession_file)],
                )
            metadata_rows = read_metadata(metadata_path)
            missing_raw = _missing_raw_accessions(accessions, metadata_rows)
            if missing_raw:
                raise RuntimeError(
                    "ENCODE files are still missing after download: "
                    + ", ".join(missing_raw)
                )
        else:
            print(
                f"Resolved experiment signals to {len(accessions)} ENCODE "
                "file(s); all are downloaded.",
                flush=True,
            )

    print(f"Checking prepared resources for genome {genome!r}.", flush=True)
    resources = _genome_resources(genome)
    missing_resources = [path for path in resources if not path.is_file()]
    if missing_resources:
        if not genome_url and genome == "hg38":
            genome_url = HG38_FASTA_URL
            print("Using the default hg38 FASTA URL.", flush=True)
        if not genome_url:
            raise ValueError(
                f"Genome resources for {genome!r} are incomplete. Provide "
                "--genome-url to prepare them. Missing: "
                + ", ".join(str(path) for path in missing_resources)
            )
        print(
            f"Preparing missing genome resources for {genome!r}.",
            flush=True,
        )
        genome_dir = resources[0].parent
        genome_arguments = [genome_url, genome]
        if genome_dir.is_dir() and any(genome_dir.iterdir()):
            genome_arguments.append("--rewrite")
        _run_script("prepare_genome.py", genome_arguments)
        missing_resources = [path for path in resources if not path.is_file()]
        if missing_resources:
            raise RuntimeError(
                f"Genome preparation did not produce required {genome} resources: "
                + ", ".join(str(path) for path in missing_resources)
            )
        print(f"Genome resources are ready for {genome}.", flush=True)
    else:
        print(f"Genome resources already present for {genome}.", flush=True)

    print("Checking required ENCODE files.", flush=True)
    missing_accessions = _missing_raw_accessions(accessions, metadata_rows)
    if missing_accessions:
        print(
            f"Downloading {len(missing_accessions)} missing ENCODE file(s).",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="pipeline-data-prep-") as temp_dir:
            accession_file = Path(temp_dir) / "missing_accessions.txt"
            _write_accessions(accession_file, missing_accessions)
            _run_script("download_encode.py", ["--input-file", str(accession_file)])
    else:
        print("All required ENCODE files are already downloaded.", flush=True)

    print("Checking single-ended BAM fragment lengths.", flush=True)
    missing_predictd = _missing_predictd_accessions(accessions, metadata_rows)
    if missing_predictd:
        print(
            f"Predicting fragment lengths for {len(missing_predictd)} file(s).",
            flush=True,
        )
        if predictd_file is not None:
            _run_script(
                "predictd.py",
                ["--precomputed-file", str(predictd_file)],
            )
        else:
            with tempfile.TemporaryDirectory(prefix="pipeline-data-prep-") as temp_dir:
                accession_file = Path(temp_dir) / "predictd_accessions.txt"
                _write_accessions(accession_file, missing_predictd)
                _run_script(
                    "predictd.py",
                    ["--input-file", str(accession_file)],
                )
        updated_metadata = read_metadata(metadata_path)
        missing_predictd = _missing_predictd_accessions(accessions, updated_metadata)
        if missing_predictd:
            raise RuntimeError(
                "Missing predictd values after prediction for: "
                + ", ".join(missing_predictd)
            )
        print("Required fragment lengths are ready.", flush=True)
    else:
        print("No predictd calculation is needed.", flush=True)

    with tempfile.TemporaryDirectory(prefix="pipeline-data-prep-") as temp_dir:
        accession_file = Path(temp_dir) / "experiment_accessions.txt"
        _write_accessions(accession_file, accessions)
        print(
            f"Processing currents for {len(accessions)} ENCODE file(s).",
            flush=True,
        )
        _run_script(
            "generate_encode_currents.py",
            ["--genome", genome, "--input-file", str(accession_file)],
        )
        print("Processing replicate and target/control data.", flush=True)
        _run_script(
            "merge_replicates.py",
            ["--genome", genome, "--input-file", str(accession_file)],
        )

    if raw_accessions:
        metadata_rows = read_metadata(metadata_path)
        input_signals, output_signal = _generated_signal_ids(
            input_accessions,
            output_accessions,
            metadata_rows,
        )
        experiment["input_accessions"] = input_accessions
        experiment["output_accessions"] = output_accessions
        experiment["control_accessions"] = control_accessions
        experiment["input_signals"] = input_signals
        experiment["output_signal"] = output_signal
        write_experiment_file(experiment_file, experiment)
        print(
            "Stored generated signal IDs in the experiment file: "
            f"{len(input_signals)} input pair(s), output pair {output_signal}.",
            flush=True,
        )

    print(f"Selecting experiment sites for {experiment_name!r}.", flush=True)
    _run_script("site_selection.py", [str(experiment_file)])

    print(f"Building experiment packs for {experiment_name!r}.", flush=True)
    _run_script("pack_builder.py", [str(experiment_file)])

    print(f"Generating or reusing splits for {experiment_name!r}.", flush=True)
    split_arguments = [
        str(experiment_file),
        "--split-candidates",
        str(split_candidates),
        "--split-cycles",
        str(split_cycles),
    ]
    if seed is not None:
        split_arguments.extend(["--seed", str(seed)])
    if redo_splits:
        split_arguments.append("--redo-splits")
    _run_script("data_split.py", split_arguments)

    experiment_dir = DEFAULT_OUTPUT_ROOT / experiment_name
    write_experiment_file(experiment_dir / "output" / "experiment.json", experiment)
    _publish_replicate_qc(experiment, experiment_dir, metadata_rows)

    print(f"Data preparation complete for {experiment_name!r}.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all genome, ENCODE, signal, replicate, and experiment data "
            "required by one experiment."
        )
    )
    parser.add_argument(
        "experiment_file",
        type=Path,
        help="JSON experiment definition",
    )
    parser.add_argument(
        "--genome-url",
        help="FASTA URL for genome preparation (hg38 has a built-in default)",
    )
    parser.add_argument(
        "--predictd-file",
        type=Path,
        help="Optional precomputed ENCODE accession/fragment-length file",
    )
    parser.add_argument(
        "--redo-splits",
        action="store_true",
        help="Regenerate splits even when existing split files are present",
    )
    parser.add_argument(
        "--split-candidates",
        type=int,
        default=10_000,
        help="Random split candidates per cycle when generating splits",
    )
    parser.add_argument(
        "--split-cycles",
        type=int,
        default=10,
        help="Split-search cycles when generating splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for reproducible split generation",
    )
    args = parser.parse_args(argv)

    if args.split_candidates <= 0 or args.split_cycles <= 0:
        parser.error("--split-candidates and --split-cycles must be positive.")

    try:
        prepare_data(
            args.experiment_file.resolve(),
            args.genome_url,
            args.predictd_file.resolve() if args.predictd_file else None,
            args.redo_splits,
            args.split_candidates,
            args.split_cycles,
            args.seed,
        )
    except (OSError, ValueError, csv.Error, RuntimeError) as error:
        print(f"data_prep: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
