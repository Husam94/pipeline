#!/usr/bin/env python3
"""Run MACS3 predictd for selected single-end ENCODE files."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from download_encode import FILE_ACCESSION_PATTERN, ensure_metadata_columns


PREDICTED_LENGTH_PATTERN = re.compile(
    r"predicted fragment length is\s+(\d+)\s+bps", re.IGNORECASE
)
SUPPORTED_FORMATS = {
    "BAM",
    "SAM",
    "BED",
    "ELAND",
    "ELANDMULTI",
    "ELANDEXPORT",
    "BOWTIE",
}


def _read_metadata(
    metadata_path: Path,
) -> tuple[list[str], list[dict[str, str | None]]]:
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        raise ValueError(f"Metadata file does not exist or is empty: {metadata_path}")

    ensure_metadata_columns(metadata_path)
    with metadata_path.open(encoding="utf-8", newline="") as metadata:
        reader = csv.DictReader(metadata, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None or not {
            "File name",
            "File format",
            "Mapped run type",
            "predictd",
        }.issubset(fieldnames):
            raise ValueError(f"Metadata file is missing required columns: {metadata_path}")
        rows: list[dict[str, str | None]] = list(reader)
        return fieldnames, rows


def _read_precomputed_values(precomputed_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with precomputed_path.open(encoding="utf-8-sig", newline="") as source:
        for line_number, row in enumerate(csv.reader(source, delimiter="\t"), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if (
                line_number == 1
                and len(row) == 2
                and row[0].strip().casefold() in {"id", "file name", "accession"}
                and row[1].strip().casefold()
                in {"predictd", "fragment length", "predicted fragment length"}
            ):
                continue
            if len(row) != 2:
                raise ValueError(
                    f"Expected ID and fragment length on line {line_number} "
                    f"of {precomputed_path}"
                )

            accession, raw_length = (value.strip() for value in row)
            if not FILE_ACCESSION_PATTERN.fullmatch(accession):
                raise ValueError(
                    f"Invalid ENCODE file accession on line {line_number}: {accession!r}"
                )
            try:
                length = int(raw_length)
            except ValueError as error:
                raise ValueError(
                    f"Invalid fragment length on line {line_number}: {raw_length!r}"
                ) from error
            if length <= 0:
                raise ValueError(
                    f"Fragment length must be positive on line {line_number}: {length}"
                )
            if accession in values:
                raise ValueError(
                    f"Duplicate accession {accession} in {precomputed_path}"
                )
            values[accession] = str(length)

    if not values:
        raise ValueError(f"No precomputed fragment lengths found in {precomputed_path}")
    return values


def _apply_precomputed_values(
    values: dict[str, str],
    rows: list[dict[str, str | None]],
    fieldnames: list[str],
    metadata_path: Path,
    rewrite: bool,
) -> None:
    updated = 0
    for accession, predicted_length in values.items():
        matching_rows = [row for row in rows if row.get("File name") == accession]
        if not matching_rows:
            print(f"Skipping {accession}: not present in {metadata_path}")
            continue
        if len(matching_rows) > 1:
            raise ValueError(f"Multiple metadata rows found for {accession}")
        row = matching_rows[0]

        if (row.get("Mapped run type") or "").strip().casefold() != "single-ended":
            print(
                f"Skipping {accession}: mapped run type is "
                f"{(row.get('Mapped run type') or '').strip() or 'not specified'}"
            )
            continue
        existing_prediction = (row.get("predictd") or "").strip()
        if existing_prediction and not rewrite:
            print(f"Already predicted for {accession}: {existing_prediction}")
            continue

        row["predictd"] = predicted_length
        updated += 1

    if updated:
        _write_metadata(metadata_path, fieldnames, rows)
    print(f"Imported precomputed predictd values for {updated} file(s)")


def _find_raw_file(accession: str, file_format: str, raw_dir: Path) -> Path:
    macs_format = file_format.upper()
    if macs_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"MACS3 predictd does not support file format {file_format!r} "
            f"for {accession}"
        )

    matching_files = [
        path
        for path in raw_dir.glob(f"{accession}.*")
        if path.is_file()
        and path.name[len(accession) + 1 :].split(".", maxsplit=1)[0].casefold()
        == file_format.casefold()
    ]
    if len(matching_files) != 1:
        raise ValueError(
            f"Expected one {file_format} file for {accession} in {raw_dir}, "
            f"found {len(matching_files)}"
        )
    return matching_files[0]


def _write_metadata(
    metadata_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str | None]],
) -> None:
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
        assert temporary_path is not None
        temporary_path.replace(metadata_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _run_predictd(
    accession: str,
    input_path: Path,
    file_format: str,
    predictd_dir: Path,
    macs3_executable: str,
) -> str:
    output_dir = predictd_dir / accession
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "predictd.log"
    command = [
        macs3_executable,
        "predictd",
        "-i",
        str(input_path),
        "-f",
        file_format.upper(),
        "--outdir",
        str(output_dir),
        "--verbose",
        "2",
    ]

    print(f"Running MACS3 predictd for {accession}; log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture MACS3 output")
        for line in process.stdout:
            log.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"MACS3 predictd exited with status {return_code}; see {log_path}"
        )
    log_contents = log_path.read_text(encoding="utf-8")
    match = PREDICTED_LENGTH_PATTERN.search(log_contents)
    if match is None:
        raise ValueError(
            f"MACS3 did not report a predicted fragment length for {accession}; "
            f"see {log_path}"
        )
    return match.group(1)


def _process_accession(
    accession: str,
    rows: list[dict[str, str | None]],
    fieldnames: list[str],
    metadata_path: Path,
    raw_dir: Path,
    predictd_dir: Path,
    macs3_executable: str,
    rewrite: bool,
) -> None:
    matching_rows = [row for row in rows if row.get("File name") == accession]
    if not matching_rows:
        raise ValueError(f"{accession} is not present in {metadata_path}")
    if len(matching_rows) > 1:
        raise ValueError(f"Multiple metadata rows found for {accession}")
    row = matching_rows[0]

    if (row.get("Mapped run type") or "").strip().casefold() != "single-ended":
        print(
            f"Skipping {accession}: mapped run type is "
            f"{(row.get('Mapped run type') or '').strip() or 'not specified'}"
        )
        return
    existing_prediction = (row.get("predictd") or "").strip()
    if existing_prediction and not rewrite:
        print(f"Already predicted for {accession}: {existing_prediction}")
        return
    if existing_prediction:
        print(f"Recomputing predictd for {accession} (was {existing_prediction})")

    file_format = (row.get("File format") or "").strip()
    input_path = _find_raw_file(accession, file_format, raw_dir)
    predicted_length = _run_predictd(
        accession,
        input_path,
        file_format,
        predictd_dir,
        macs3_executable,
    )
    row["predictd"] = predicted_length
    _write_metadata(metadata_path, fieldnames, rows)
    print(f"Saved predictd value {predicted_length} for {accession}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MACS3 predictd for single-end ENCODE files."
    )
    parser.add_argument(
        "file_names",
        nargs="*",
        help="One or more ENCODE file accessions, for example ENCFF721LSL",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Text file containing accessions separated by tabs, commas, or newlines",
    )
    parser.add_argument(
        "--precomputed-file",
        type=Path,
        help="Import tab-delimited ENCODE ID and fragment-length pairs instead of running MACS3",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Replace predictd values that are already present in meta_info.txt",
    )
    args = parser.parse_args(argv)

    precomputed_values = None
    if args.precomputed_file:
        if args.input_file or args.file_names:
            parser.error(
                "provide either --precomputed-file or accessions/--input-file, not both"
            )
        try:
            precomputed_values = _read_precomputed_values(args.precomputed_file)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            parser.error(f"cannot read precomputed file {args.precomputed_file}: {error}")
        file_names = list(precomputed_values)
    elif args.input_file:
        if args.file_names:
            parser.error(
                "provide file accessions directly or with --input-file, not both"
            )
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
    metadata_path = encode_dir / "meta_info.txt"
    raw_dir = encode_dir / "raw"
    predictd_dir = encode_dir / "predictd"

    try:
        fieldnames, rows = _read_metadata(metadata_path)
        if precomputed_values is not None:
            _apply_precomputed_values(
                precomputed_values,
                rows,
                fieldnames,
                metadata_path,
                args.rewrite,
            )
            return 0

        macs3_executable = shutil.which("macs3")
        if macs3_executable is None:
            raise RuntimeError(
                "MACS3 was not found on PATH; activate the environment that provides "
                "the 'macs3' command."
            )

        failures = 0
        for accession in dict.fromkeys(file_names):
            try:
                _process_accession(
                    accession,
                    rows,
                    fieldnames,
                    metadata_path,
                    raw_dir,
                    predictd_dir,
                    macs3_executable,
                    args.rewrite,
                )
            except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
                failures += 1
                print(f"predictd: error processing {accession}: {error}", file=sys.stderr)
        return 1 if failures else 0
    except (OSError, csv.Error, ValueError, RuntimeError) as error:
        print(f"predictd: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
