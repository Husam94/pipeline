#!/usr/bin/env python3
"""Create full, low-resolution, autosomal, and windowed ENCODE currents."""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import poseigen_chisel.poseigen_chisel as chis
import poseigen_seaside.basics as se

from download_encode import FILE_ACCESSION_PATTERN, ensure_metadata_columns


LOWEST_RESOLUTION = 10
TARGET_RESOLUTION = 50
WINDOW_SIZE = 200
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODE_ROOT = PROJECT_ROOT / "data" / "ENCODE"
GENOME_ROOT = PROJECT_ROOT / "data" / "genomes"


def parse_accessions(values: list[str], input_file: Path | None) -> list[str]:
    if values and input_file:
        raise ValueError("Provide accessions directly or with --input-file, not both.")

    if input_file:
        text = input_file.read_text(encoding="utf-8")
        accessions = [part for part in re.split(r"[\s,]+", text) if part]
    else:
        accessions = values

    if not accessions:
        raise ValueError("Provide at least one ENCODE file accession.")

    invalid = [accession for accession in accessions
               if not FILE_ACCESSION_PATTERN.fullmatch(accession)]
    if invalid:
        raise ValueError(f"Invalid ENCODE file accession(s): {', '.join(invalid)}")

    return list(dict.fromkeys(accessions))


def load_genome_resources(
    genome: str = "hg38",
) -> tuple[list[int], list[str], list[str], list[str], list[np.ndarray]]:
    genome_root = GENOME_ROOT / genome
    required = (
        genome_root / f"{genome}_id_sizes.p",
        genome_root / f"{genome}_basic_id_sizes.p",
        genome_root / f"{genome}_basic_BLC_LR50.p",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {genome} processing resource(s): "
            + ", ".join(str(path) for path in missing)
        )

    genome_ids, genome_sizes = se.PickleLoad(
        str(genome_root / f"{genome}_id_sizes")
    )
    basic_ids, _ = se.PickleLoad(str(genome_root / f"{genome}_basic_id_sizes"))
    blacklist_current = se.PickleLoad(
        str(genome_root / f"{genome}_basic_BLC_LR50")
    )

    if len(basic_ids) < 4:
        raise ValueError(
            f"{genome} basic chromosome list is too short to select autosomes."
        )
    autosome_ids = basic_ids[:-3]
    if len(blacklist_current) != len(basic_ids):
        raise ValueError(
            f"{genome} blacklist current and basic chromosome list have different lengths."
        )

    return genome_sizes, genome_ids, basic_ids, autosome_ids, blacklist_current


def find_raw_file(accession: str, file_format: str, raw_dir: Path) -> Path:
    normalized_format = file_format.casefold()
    if normalized_format not in {"bam", "bigwig"}:
        raise ValueError(
            f"{accession}: unsupported ENCODE file format {file_format!r}; "
            "expected BAM or bigWig."
        )

    matches = [
        path for path in raw_dir.glob(f"{accession}.*")
        if path.is_file()
        and path.name[len(accession) + 1:].casefold() == normalized_format
    ]
    if not matches:
        raise FileNotFoundError(
            f"{accession}: no {file_format} file found in {raw_dir}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{accession}: multiple {file_format} files found in {raw_dir}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def read_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"ENCODE metadata file not found: {metadata_path}")
    ensure_metadata_columns(metadata_path)

    with metadata_path.open("r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file, delimiter="\t")
        required_columns = {"File name", "File format", "Mapped run type", "predictd"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{metadata_path} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            accession = (row.get("File name") or "").strip()
            if accession:
                rows[accession] = row
    return rows


def make_full_current(
    accession: str,
    row: dict[str, str],
    raw_dir: Path,
    genome_sizes: list[int],
    genome_ids: list[str],
    basic_ids: list[str],
):
    file_format = (row.get("File format") or "").strip()
    source_path = find_raw_file(accession, file_format, raw_dir)

    if file_format.casefold() == "bam":
        run_type = (row.get("Mapped run type") or "").strip().casefold()
        if run_type == "single-ended":
            predictd = (row.get("predictd") or "").strip()
            if not predictd:
                raise ValueError(
                    f"{accession}: single-ended BAM has no predictd value in "
                    "meta_info.txt; run predictd.py first."
                )
            try:
                read_length = int(predictd)
            except ValueError as error:
                raise ValueError(
                    f"{accession}: invalid predictd value {predictd!r}."
                ) from error
            if read_length <= 0:
                raise ValueError(f"{accession}: predictd must be a positive integer.")
            paired = False
        elif run_type == "paired-ended":
            read_length = None
            paired = True
        else:
            raise ValueError(
                f"{accession}: unsupported mapped run type {run_type!r} for BAM."
            )

        if not Path(f"{source_path}.bai").is_file():
            chis.BamMakeIndex(str(source_path))
        return chis.Bam2Current(
            str(source_path),
            genome_sizes,
            genome_ids,
            select_BS_ids=basic_ids,
            newreso=LOWEST_RESOLUTION,
            resomode=np.mean,
            stranded=False,
            read_length=read_length,
            paired=paired,
            dtype=np.float32,
        )

    return chis.Bw2Current(
        str(source_path),
        genome_sizes,
        genome_ids,
        select_BS_ids=basic_ids,
        newreso=LOWEST_RESOLUTION,
        resomode=np.mean,
        dtype=np.float32,
        parallel=True,
        n_jobs=2,
    )


def process_accession(
    accession: str,
    row: dict[str, str],
    resources: tuple[list[int], list[str], list[str], list[str], list[np.ndarray]],
    encode_root: Path = ENCODE_ROOT,
) -> None:
    genome_sizes, genome_ids, basic_ids, autosome_ids, blacklist_current = resources
    stage_dirs = {
        "full": encode_root / "full",
        "lowres": encode_root / "lowres",
        "autos": encode_root / "autos",
        "win": encode_root / "win",
    }
    for stage_dir in stage_dirs.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        stage: stage_dirs[stage] / f"{accession}.p"
        for stage in stage_dirs
    }

    if not output_paths["full"].is_file():
        full_current = make_full_current(
            accession, row, encode_root / "raw", genome_sizes, genome_ids, basic_ids
        )
        se.PickleDump(full_current, str(output_paths["full"]))
        del full_current

    if not output_paths["lowres"].is_file():
        full_current = se.PickleLoad(str(output_paths["full"]))
        lowres_current = chis.LowerResCurrent(
            full_current,
            LOWEST_RESOLUTION,
            TARGET_RESOLUTION,
            resomode=np.mean,
            dtype=np.float32,
        )
        se.PickleDump(lowres_current, str(output_paths["lowres"]))
        del full_current, lowres_current

    if not output_paths["autos"].is_file():
        lowres_current = se.PickleLoad(str(output_paths["lowres"]))
        autos_current = chis.CurrentSelect(
            lowres_current, basic_ids, autosome_ids
        )
        se.PickleDump(autos_current, str(output_paths["autos"]))
        del lowres_current, autos_current

    if not output_paths["win"].is_file():
        autos_current = se.PickleLoad(str(output_paths["autos"]))
        windowed_current = chis.CurrentWindower(
            autos_current,
            reso=TARGET_RESOLUTION,
            mode=np.mean,
            window=WINDOW_SIZE,
            center=0,
            extend=0,
            dtype=np.float32,
            indiv=True,
        )
        autosome_blacklist = chis.CurrentSelect(
            blacklist_current, basic_ids, autosome_ids
        )
        if len(windowed_current) != len(autosome_blacklist):
            raise ValueError(
                f"{accession}: windowed current and autosome blacklist counts differ."
            )
        for chrom, current, blacklist in zip(
            autosome_ids, windowed_current, autosome_blacklist
        ):
            if current.shape != blacklist.shape:
                raise ValueError(
                    f"{accession}: windowed current and blacklist shapes differ "
                    f"for {chrom}: {current.shape} vs {blacklist.shape}."
                )

        filtered_current = chis.CurrentModifier(
            windowed_current,
            autosome_blacklist,
            mode=np.multiply,
            dtype=np.float32,
            single=False,
        )
        se.PickleDump(filtered_current, str(output_paths["win"]))
        del autos_current, windowed_current, autosome_blacklist, filtered_current

    print(f"{accession}: processing complete.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create ENCODE full, low-resolution, autosomal, and "
            "windowed/blacklist-filtered currents."
        )
    )
    parser.add_argument(
        "accessions",
        nargs="*",
        help="ENCODE file accessions (for example, ENCFF721LSL).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Text file of accessions separated by tabs, commas, or newlines.",
    )
    parser.add_argument(
        "--genome",
        default="hg38",
        help="Genome name used for prepared resources (default: hg38).",
    )
    args = parser.parse_args()

    try:
        accessions = parse_accessions(args.accessions, args.input_file)
        metadata = read_metadata(ENCODE_ROOT / "meta_info.txt")
        resources = load_genome_resources(args.genome)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    failures = 0
    for accession in accessions:
        row = metadata.get(accession)
        if row is None:
            print(
                f"{accession}: not found in {ENCODE_ROOT / 'meta_info.txt'}.",
                file=sys.stderr,
            )
            failures += 1
            continue
        try:
            process_accession(accession, row, resources)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"{accession}: {error}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
