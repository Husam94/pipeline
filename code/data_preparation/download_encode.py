#!/usr/bin/env python3
"""Download ENCODE files and append file/experiment metadata to a TSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ENCODE_BASE_URL = "https://www.encodeproject.org/"
METADATA_COLUMNS = (
    "File name",
    "Replicate ID",
    "Pair ID",
    "Dataset",
    "File format",
    "Mapped run type",
    "Mapped read length",
    "Assembly",
    "Date added",
    "Assay",
    "Target",
    "Species",
    "Cell",
    "Condition",
    "Biological replicate",
    "Technical replicate",
    "Description",
    "Size range",
    "Lab",
    "Controls",
    "predictd",
)
FILE_ACCESSION_PATTERN = re.compile(r"ENCFF[A-Z0-9]+")
EXPERIMENT_ACCESSION_PATTERN = re.compile(r"ENCSR[A-Z0-9]+")


def _request(url: str, accept: str = "application/json") -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"Accept": accept, "User-Agent": "download_encode"},
    )


def _fetch_json(url: str, frame: str = "object") -> dict[str, Any]:
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}format=json&frame={frame}"
    with urllib.request.urlopen(_request(request_url), timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected JSON response from ENCODE: {url}")
    return result


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(filter(None, (_display(item) for item in value)))
    if isinstance(value, dict):
        for key in ("label", "title", "accession", "name", "@id"):
            if value.get(key):
                return _display(value[key])
        return ""
    return " ".join(str(value).split())


def _resolve_label(value: Any, label_keys: tuple[str, ...]) -> str:
    if isinstance(value, dict):
        for key in label_keys:
            if value.get(key):
                return _display(value[key])
        reference = value.get("@id")
    else:
        reference = value

    if isinstance(reference, str) and (
        reference.startswith("/") or reference.startswith(ENCODE_BASE_URL)
    ):
        details = _fetch_json(urllib.parse.urljoin(ENCODE_BASE_URL, reference))
        for key in label_keys:
            if details.get(key):
                return _display(details[key])
        return _display(details.get("accession") or details.get("title"))
    return _display(value)


def _accession_from_reference(value: Any) -> str:
    if isinstance(value, dict):
        accession = value.get("accession")
        if isinstance(accession, str):
            return accession
        value = value.get("@id", "")
    if isinstance(value, str):
        match = EXPERIMENT_ACCESSION_PATTERN.search(value)
        if match:
            return match.group(0)
    return _display(value)


def _dataset_accession(value: Any) -> str:
    accession = _accession_from_reference(value)
    if not EXPERIMENT_ACCESSION_PATTERN.fullmatch(accession):
        raise ValueError(f"File metadata has an invalid experiment dataset: {value!r}")
    return accession


def _file_extension(file_data: dict[str, Any], accession: str) -> str:
    href = file_data.get("href")
    if isinstance(href, str):
        basename = Path(
            urllib.parse.unquote(urllib.parse.urlparse(href).path)
        ).name
        prefix = f"{accession}."
        if basename.startswith(prefix):
            extension = basename[len(prefix) :]
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", extension):
                return extension

    fallback = file_data.get("file_format") or file_data.get("file_type")
    extension = _display(fallback)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", extension):
        raise ValueError(f"Could not determine a safe file extension for {accession}")
    return extension


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{value:.1f} TiB"


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def _file_replicates(
    file_data: dict[str, Any], experiment_data: dict[str, Any]
) -> list[dict[str, Any]]:
    replicates = experiment_data.get("replicates") or []
    if not isinstance(replicates, list):
        return []
    replicates = [replicate for replicate in replicates if isinstance(replicate, dict)]

    biological_replicates = {
        _display(value) for value in _as_values(file_data.get("biological_replicates"))
    }
    technical_replicates = {
        _display(value) for value in _as_values(file_data.get("technical_replicates"))
    }
    if not biological_replicates and not technical_replicates:
        return replicates

    matching = []
    for replicate in replicates:
        biological = _display(replicate.get("biological_replicate_number"))
        technical = _display(replicate.get("technical_replicate_number"))
        combined_technical = f"{biological}_{technical}" if biological and technical else ""
        if biological_replicates and biological not in biological_replicates:
            continue
        if technical_replicates and not technical_replicates.intersection(
            {technical, combined_technical}
        ):
            continue
        matching.append(replicate)
    return matching or replicates


def _sample_summary(sample: dict[str, Any]) -> str:
    for key in ("summary", "biosample_summary", "simple_biosample_summary"):
        value = _display(sample.get(key))
        if value:
            return value
    return ""


def _summary_species_and_cell(summary: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][a-z]+ [a-z]+)\s+(.+)$", summary.strip())
    if not match:
        return "", ""
    cell = re.sub(r"\s+(?:cell line|primary cell)$", "", match.group(2).strip(), flags=re.I)
    return match.group(1), cell


def _treatment_label(treatment: Any) -> str:
    if isinstance(treatment, str) and (
        treatment.startswith("/") or treatment.startswith(ENCODE_BASE_URL)
    ):
        treatment = _fetch_json(
            urllib.parse.urljoin(ENCODE_BASE_URL, treatment), frame="embedded"
        )
    if not isinstance(treatment, dict):
        return _display(treatment)

    name = _display(
        treatment.get("treatment_term_name")
        or treatment.get("term_name")
        or treatment.get("name")
        or treatment.get("treatment_term_id")
    )
    if not name:
        return ""
    details = [
        value
        for value in (
            _display(treatment.get("concentration")),
            _display(treatment.get("duration")),
        )
        if value
    ]
    return f"{name} ({', '.join(details)})" if details else name


def _condition_from_description(description: str) -> str:
    heat_shock = re.search(r"\bheat[\s-]*shock(?:ed)?\b", description, re.IGNORECASE)
    if heat_shock:
        return "heat shock"
    treated = re.search(
        r"\b(?:treated|treatment|exposed)\s+(?:with|to)\s+([^.;,\n]+)",
        description,
        re.IGNORECASE,
    )
    return treated.group(1).strip() if treated else ""


def _sample_metadata(
    file_data: dict[str, Any], experiment_data: dict[str, Any]
) -> tuple[str, str, str]:
    samples: list[dict[str, Any]] = []
    for replicate in _file_replicates(file_data, experiment_data):
        library = replicate.get("library")
        sample = library.get("biosample") if isinstance(library, dict) else None
        if isinstance(sample, dict):
            samples.append(sample)
        elif isinstance(sample, str) and (
            sample.startswith("/") or sample.startswith(ENCODE_BASE_URL)
        ):
            samples.append(
                _fetch_json(
                    urllib.parse.urljoin(ENCODE_BASE_URL, sample), frame="embedded"
                )
            )

    if not samples:
        samples = [experiment_data]

    species_values: list[str] = []
    cell_values: list[str] = []
    condition_values: list[str] = []
    for sample in samples:
        organism = sample.get("organism")
        species = _resolve_label(
            organism, ("scientific_name", "organism", "name", "title")
        )
        if species:
            species_values.append(species)

        cell = _resolve_label(
            sample.get("biosample_term_name"), ("term_name", "title", "name")
        )
        if not cell:
            cell = _resolve_label(
                sample.get("biosample_ontology"),
                ("term_name", "cell_line_name", "title", "name"),
            )
        if cell:
            cell_values.append(cell)

        for treatment in _as_values(sample.get("treatments")):
            label = _treatment_label(treatment)
            if label:
                condition_values.append(label)
        direct_treatment = _treatment_label(
            {
                "treatment_term_name": sample.get("treatment_term_name"),
                "concentration": sample.get("treatment_concentration"),
                "duration": sample.get("treatment_duration"),
            }
        )
        if direct_treatment:
            condition_values.append(direct_treatment)

        summary_species, summary_cell = _summary_species_and_cell(
            _sample_summary(sample)
        )
        if summary_species:
            species_values.append(summary_species)
        if summary_cell:
            cell_values.append(summary_cell)

    if not species_values or not cell_values:
        summary_species, summary_cell = _summary_species_and_cell(
            _sample_summary(experiment_data)
        )
        if summary_species:
            species_values.append(summary_species)
        if summary_cell:
            cell_values.append(summary_cell)

    if not condition_values:
        description = _display(
            experiment_data.get("description") or file_data.get("description")
        )
        fallback_condition = _condition_from_description(description)
        if fallback_condition:
            condition_values.append(fallback_condition)

    species = ", ".join(dict.fromkeys(value for value in species_values if value))
    cell = ", ".join(dict.fromkeys(value for value in cell_values if value))
    condition = ", ".join(dict.fromkeys(value for value in condition_values if value))
    return species, cell, condition


def _progress_message(accession: str, downloaded: int, total: int | None) -> str:
    if total is None:
        return f"Downloading {accession}: {_format_size(downloaded)}"
    percentage = min(100, int(downloaded * 100 / total)) if total else 100
    return (
        f"Downloading {accession}: {percentage}% "
        f"({_format_size(downloaded)} / {_format_size(total)})"
    )


def _write_progress(
    accession: str, downloaded: int, total: int | None, is_terminal: bool
) -> None:
    message = _progress_message(accession, downloaded, total)
    if is_terminal and total:
        percentage = min(100, int(downloaded * 100 / total))
        bar_width = 24
        filled = int(bar_width * percentage / 100)
        message = (
            f"Downloading {accession}: "
            f"[{'#' * filled}{'-' * (bar_width - filled)}] "
            f"{message.split(': ', 1)[1]}"
        )
    sys.stderr.write(f"\r{message}" if is_terminal else f"{message}\n")
    sys.stderr.flush()


def _download_file(href: str, output_path: Path, accession: str) -> None:
    download_url = urllib.parse.urljoin(ENCODE_BASE_URL, href)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".download",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    is_terminal = sys.stderr.isatty()
    progress_started = False
    try:
        with urllib.request.urlopen(
            _request(download_url, accept="*/*"), timeout=60
        ) as response:
            with temporary_path.open("wb") as output:
                header_value = response.headers.get("Content-Length")
                total_bytes = int(header_value) if header_value else None
                if total_bytes is not None and total_bytes < 0:
                    raise ValueError(f"Invalid Content-Length for {download_url}")

                downloaded = 0
                last_update = time.monotonic()
                last_percentage: int | None = 0 if total_bytes else None
                progress_started = True
                _write_progress(accession, 0, total_bytes, is_terminal)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    percentage = (
                        int(downloaded * 100 / total_bytes) if total_bytes else None
                    )
                    if (
                        is_terminal
                        and (
                            (percentage is not None and percentage != last_percentage)
                            or now - last_update >= 0.5
                        )
                    ) or (not is_terminal and now - last_update >= 2):
                        _write_progress(
                            accession, downloaded, total_bytes, is_terminal
                        )
                        last_update = now
                        last_percentage = percentage
                if total_bytes is not None and downloaded != total_bytes:
                    raise ValueError(
                        f"Incomplete download for {output_path.name}: "
                        f"received {downloaded} of {total_bytes} bytes"
                    )
                _write_progress(
                    accession, downloaded, total_bytes, is_terminal
                )
        if temporary_path.stat().st_size == 0:
            raise ValueError(f"Downloaded ENCODE file is empty: {download_url}")
        temporary_path.replace(output_path)
    finally:
        if progress_started and is_terminal:
            sys.stderr.write("\n")
            sys.stderr.flush()
        temporary_path.unlink(missing_ok=True)


def _metadata_row(
    accession: str, file_data: dict[str, Any], dataset_data: dict[str, Any]
) -> dict[str, str]:
    controls = dataset_data.get("controls") or dataset_data.get("possible_controls") or []
    if not isinstance(controls, list):
        controls = [controls]
    species, cell, condition = _sample_metadata(file_data, dataset_data)

    return {
        "File name": accession,
        "Replicate ID": "",
        "Pair ID": "",
        "Dataset": _dataset_accession(file_data.get("dataset")),
        "File format": _display(file_data.get("file_format")),
        "Mapped run type": _display(file_data.get("mapped_run_type")),
        "Mapped read length": _display(file_data.get("mapped_read_length")),
        "Assembly": _display(file_data.get("assembly")),
        "Date added": _display(file_data.get("date_created")),
        "Assay": _display(
            dataset_data.get("assay_title") or dataset_data.get("assay_term_name")
        ),
        "Target": _resolve_label(dataset_data.get("target"), ("label", "title")),
        "Species": species,
        "Cell": cell,
        "Condition": condition,
        "Biological replicate": _display(file_data.get("biological_replicates")),
        "Technical replicate": _display(file_data.get("technical_replicates")),
        "Description": _display(dataset_data.get("description")),
        "Size range": _display(dataset_data.get("size_range")),
        "Lab": _resolve_label(dataset_data.get("lab"), ("title", "name")),
        "Controls": ", ".join(
            filter(None, (_accession_from_reference(control) for control in controls))
        ),
    }


def _read_existing_accessions(metadata_path: Path) -> set[str]:
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return set()
    with metadata_path.open(encoding="utf-8", newline="") as metadata:
        reader = csv.DictReader(metadata, delimiter="\t")
        if reader.fieldnames is None or "File name" not in reader.fieldnames:
            raise ValueError(f"Metadata file has no 'File name' column: {metadata_path}")
        return {
            row["File name"]
            for row in reader
            if row.get("File name")
        }


def ensure_metadata_columns(metadata_path: Path) -> None:
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return

    with metadata_path.open(encoding="utf-8", newline="") as metadata:
        reader = csv.DictReader(metadata, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None or "File name" not in fieldnames:
            raise ValueError(f"Metadata file has no 'File name' column: {metadata_path}")
        if "Target" not in fieldnames:
            raise ValueError(f"Metadata file has no 'Target' column: {metadata_path}")
        rows = list(reader)

    updated_fieldnames = list(METADATA_COLUMNS) + [
        name for name in fieldnames if name not in METADATA_COLUMNS
    ]
    if updated_fieldnames == fieldnames:
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=updated_fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        assert temporary_path is not None
        temporary_path.replace(metadata_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def backfill_sample_metadata(metadata_path: Path) -> None:
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return

    with metadata_path.open(encoding="utf-8", newline="") as metadata:
        reader = csv.DictReader(metadata, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Metadata file has no header: {metadata_path}")
        rows = list(reader)

    incomplete_rows = [
        row for row in rows
        if not row.get("Species", "").strip() or not row.get("Cell", "").strip()
    ]
    if not incomplete_rows:
        return

    experiments: dict[str, dict[str, Any]] = {}
    for row in incomplete_rows:
        accession = (row.get("File name") or "").strip()
        dataset = (row.get("Dataset") or "").strip()
        if not accession or not dataset:
            raise ValueError(
                f"Cannot backfill sample metadata for row with missing file or "
                f"dataset accession: {row!r}"
            )
        if dataset not in experiments:
            experiments[dataset] = _fetch_json(
                urllib.parse.urljoin(
                    ENCODE_BASE_URL, f"experiments/{dataset}/"
                ),
                frame="embedded",
            )

        file_data = {
            "biological_replicates": _as_values(row.get("Biological replicate")),
            "technical_replicates": _as_values(row.get("Technical replicate")),
        }
        species, cell, condition = _sample_metadata(file_data, experiments[dataset])
        row["Species"] = species
        row["Cell"] = cell
        row["Condition"] = condition

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=metadata_path.parent,
            prefix=f".{metadata_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.DictWriter(
                temporary,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(metadata_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _append_metadata(metadata_path: Path, row: dict[str, str]) -> None:
    needs_header = not metadata_path.exists() or metadata_path.stat().st_size == 0
    if needs_header:
        fieldnames = list(METADATA_COLUMNS)
    else:
        with metadata_path.open(encoding="utf-8", newline="") as metadata:
            fieldnames = csv.DictReader(metadata, delimiter="\t").fieldnames
        if fieldnames is None:
            raise ValueError(f"Metadata file has no header: {metadata_path}")

    with metadata_path.open("a", encoding="utf-8", newline="") as metadata:
        writer = csv.DictWriter(
            metadata,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        if needs_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def process_file(
    accession: str,
    output_dir: Path,
    metadata_path: Path,
    known_accessions: set[str],
) -> None:
    file_data = _fetch_json(
        urllib.parse.urljoin(ENCODE_BASE_URL, f"files/{accession}/")
    )
    metadata_row = None
    if accession not in known_accessions:
        dataset_accession = _dataset_accession(file_data.get("dataset"))
        dataset_data = _fetch_json(
            urllib.parse.urljoin(ENCODE_BASE_URL, f"experiments/{dataset_accession}/"),
            frame="embedded",
        )
        metadata_row = _metadata_row(accession, file_data, dataset_data)

    href = file_data.get("href")
    if not isinstance(href, str):
        raise ValueError(f"File metadata has no download URL for {accession}")
    output_path = output_dir / f"{accession}.{_file_extension(file_data, accession)}"
    if output_path.exists():
        print(f"Already downloaded: {output_path.name}")
    else:
        _download_file(href, output_path, accession)
        print(f"Downloaded {accession} to {output_path}")

    if metadata_row is not None:
        _append_metadata(metadata_path, metadata_row)
        known_accessions.add(accession)
        print(f"Added metadata for {accession}")
    else:
        print(f"Metadata already present for {accession}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download ENCODE files and maintain a tab-delimited metadata list."
    )
    parser.add_argument(
        "file_names",
        nargs="*",
        help="One or more ENCODE file accessions, for example ENCFF000ABC",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Text file containing accessions separated by tabs, commas, or newlines",
    )
    args = parser.parse_args(argv)

    if args.input_file and args.file_names:
        parser.error("provide file accessions directly or with --input-file, not both")
    if args.input_file:
        try:
            file_names = [
                name.strip()
                for name in re.split(
                    r"[\t,\r\n]+",
                    args.input_file.read_text(encoding="utf-8-sig"),
                )
                if name.strip()
            ]
        except (OSError, UnicodeError) as error:
            parser.error(f"cannot read input file {args.input_file}: {error}")
    else:
        file_names = args.file_names

    if not file_names:
        parser.error("provide file accessions or an --input-file")

    invalid = [
        name for name in file_names if not FILE_ACCESSION_PATTERN.fullmatch(name)
    ]
    if invalid:
        parser.error(
            "file names must be ENCODE file accessions (ENCFF followed by letters "
            f"or digits): {', '.join(invalid)}"
        )

    project_root = Path(__file__).resolve().parents[1]
    encode_dir = project_root / "data" / "ENCODE"
    raw_dir = encode_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = encode_dir / "meta_info.txt"

    try:
        ensure_metadata_columns(metadata_path)
        backfill_sample_metadata(metadata_path)
        known_accessions = _read_existing_accessions(metadata_path)
        failures = 0
        for accession in dict.fromkeys(file_names):
            try:
                process_file(accession, raw_dir, metadata_path, known_accessions)
            except (
                OSError,
                urllib.error.URLError,
                json.JSONDecodeError,
                ValueError,
            ) as error:
                failures += 1
                print(f"download_encode: error processing {accession}: {error}", file=sys.stderr)
        return 1 if failures else 0
    except (OSError, csv.Error, ValueError) as error:
        print(f"download_encode: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
