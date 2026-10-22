#!/usr/bin/env python3
"""Download and prepare a genome FASTA and any matching Boyle-Lab blacklist."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO


BLACKLIST_LIST_URL = (
    "https://api.github.com/repos/Boyle-Lab/Blacklist/contents/lists?ref=master"
)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "prepare_genome",
        },
    )


def _write_fasta_stream(source: BinaryIO, output_path: Path) -> None:
    magic = source.read(2)
    source.seek(0)
    if magic == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=source) as uncompressed:
            with output_path.open("wb") as output:
                shutil.copyfileobj(uncompressed, output)
    else:
        with output_path.open("wb") as output:
            shutil.copyfileobj(source, output)


def download_genome(genome_url: str, genome_fasta: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=genome_fasta.parent, delete=False) as raw_file:
        raw_path = Path(raw_file.name)
    fasta_path = raw_path.with_suffix(".fasta")
    try:
        with urllib.request.urlopen(_request(genome_url), timeout=60) as response:
            with raw_path.open("wb") as download:
                shutil.copyfileobj(response, download)
        if raw_path.stat().st_size == 0:
            raise ValueError(f"Downloaded genome is empty: {genome_url}")
        with raw_path.open("rb") as source:
            _write_fasta_stream(source, fasta_path)
        fasta_path.replace(genome_fasta)
    finally:
        raw_path.unlink(missing_ok=True)
        fasta_path.unlink(missing_ok=True)


def parse_fasta(fasta_path: Path) -> tuple[list[str], list[str]]:
    sequences: list[str] = []
    identifiers: list[str] = []
    current_sequence: list[str] | None = None

    with fasta_path.open(encoding="ascii") as fasta:
        for line_number, line in enumerate(fasta, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_sequence is not None:
                    sequences.append("".join(current_sequence))
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header on line {line_number}")
                identifiers.append(header.split()[0])
                current_sequence = []
            else:
                if current_sequence is None:
                    raise ValueError(
                        f"Sequence data before first FASTA header on line {line_number}"
                    )
                current_sequence.append("".join(line.split()))

    if current_sequence is not None:
        sequences.append("".join(current_sequence))
    if not sequences:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return sequences, identifiers


def _dump_pickle(value: object, output_path: Path) -> None:
    with output_path.open("wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)


def save_genome_data(
    genome_name: str, output_dir: Path, sequences: list[str], identifiers: list[str]
) -> tuple[list[int], list[str]]:
    sizes = [len(sequence) for sequence in sequences]
    _dump_pickle(sequences, output_dir / f"{genome_name}.p")
    _dump_pickle(
        [identifiers, sizes], output_dir / f"{genome_name}_id_sizes.p"
    )

    for suffix, count in (("basic", 25), ("mini", 3)):
        selected_sequences = sequences[:count]
        selected_ids = identifiers[:count]
        selected_sizes = sizes[:count]
        _dump_pickle(
            selected_sequences, output_dir / f"{genome_name}_{suffix}.p"
        )
        _dump_pickle(
            [selected_ids, selected_sizes],
            output_dir / f"{genome_name}_{suffix}_id_sizes.p",
        )
    return sizes, identifiers


def find_blacklist(genome_name: str) -> dict[str, str] | None:
    with urllib.request.urlopen(_request(BLACKLIST_LIST_URL), timeout=30) as response:
        entries = json.load(response)

    matches = [
        entry
        for entry in entries
        if entry.get("type") == "file"
        and genome_name.casefold() in entry.get("name", "").casefold()
        and entry.get("name", "").casefold().endswith((".bed", ".bed.gz"))
    ]
    if not matches:
        return None
    if len(matches) > 1:
        filenames = ", ".join(sorted(entry["name"] for entry in matches))
        raise ValueError(
            f"Multiple blacklist BED files match {genome_name!r}: {filenames}"
        )
    entry = matches[0]
    download_url = entry.get("download_url")
    if not isinstance(download_url, str):
        raise ValueError(f"Blacklist entry has no download URL: {entry.get('name')}")
    return {"name": entry["name"], "download_url": download_url}


def download_blacklist(blacklist: dict[str, str], output_dir: Path) -> Path:
    filename = Path(blacklist["name"]).name
    compressed_path = output_dir / filename
    with urllib.request.urlopen(
        _request(blacklist["download_url"]), timeout=60
    ) as response:
        with compressed_path.open("wb") as output:
            shutil.copyfileobj(response, output)

    if filename.endswith(".gz"):
        bed_path = output_dir / filename[:-3]
        try:
            with gzip.open(compressed_path, "rb") as source:
                with bed_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
        except (OSError, EOFError):
            bed_path.unlink(missing_ok=True)
            raise
        return bed_path
    return compressed_path


def process_blacklist(
    genome_name: str,
    bed_path: Path,
    output_dir: Path,
    sizes: list[int],
    identifiers: list[str],
) -> None:
    try:
        import numpy as np
        import poseigen_chisel as chis
    except ImportError as error:
        raise RuntimeError(
            "Blacklist processing requires numpy and poseigen_chisel."
        ) from error

    markers = chis.Bed2Markers(str(bed_path))
    basic_ids = identifiers[:25]
    basic_blacklist_current = chis.Markers2BinCurrent(
        markers,
        sizes,
        identifiers,
        select_BS_ids=basic_ids,
        Msizes=markers.iloc[:, 3],
        reso=1,
        inverse=True,
        dtype=np.int8,
    )
    _dump_pickle(
        basic_blacklist_current,
        output_dir / f"{genome_name}_basic_BLC.p",
    )

    for resolution in (10, 20, 50):
        lowered_current = chis.LowerResCurrent(
            basic_blacklist_current,
            1,
            resolution,
            resomode=np.min,
            dtype=np.int8,
        )
        _dump_pickle(
            lowered_current,
            output_dir / f"{genome_name}_basic_BLC_LR{resolution}.p",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare a genome FASTA and matching blacklist."
    )
    parser.add_argument("genome_url", help="URL of the genome FASTA (plain or gzip)")
    parser.add_argument("genome_name", help="Genome label used for output file names")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Regenerate outputs when the genome data directory is not empty",
    )
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.genome_name):
        parser.error("genome_name must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "genomes" / args.genome_name
    if not args.rewrite and output_dir.is_dir() and any(output_dir.iterdir()):
        print("Already present, use --rewrite to make new")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        genome_fasta = output_dir / f"{args.genome_name}.fasta"
        download_genome(args.genome_url, genome_fasta)
        sequences, identifiers = parse_fasta(genome_fasta)
        sizes, identifiers = save_genome_data(
            args.genome_name, output_dir, sequences, identifiers
        )
        print(f"Saved genome files to {output_dir}")

        blacklist = find_blacklist(args.genome_name)
        if blacklist is None:
            print(f"No matching Boyle-Lab blacklist found for {args.genome_name}.")
            return 0

        bed_path = download_blacklist(blacklist, output_dir)
        process_blacklist(args.genome_name, bed_path, output_dir, sizes, identifiers)
        print(f"Processed blacklist {blacklist['name']} in {output_dir}")
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"prepare_genome: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
