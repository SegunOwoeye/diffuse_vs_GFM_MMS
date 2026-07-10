"""Collect Report 2 outputs into report-facing folders."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger("organize_report2_outputs")

DEFAULT_SOURCES = (
    Path("results/quantitative"),
    Path("results/bubble_features"),
    Path("results/diagnostics"),
)

RAW_PARTS = {"raw", "runs"}
REPORT_SUFFIXES = {".csv", ".json", ".txt", ".md", ".png", ".html"}


@dataclass(frozen=True)
class OrganizedFile:
    category: str
    source: Path
    destination: Path


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Report 2 outputs into categorized report folders."
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        dest="sources",
        help="Source result root. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/report2_organized"),
        help="Destination for organized report outputs.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw solver CSVs and run directories.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the organizer output directory before copying.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def safe_clean_output_dir(path: Path) -> None:
    resolved = path.resolve()
    workspace = Path.cwd().resolve()

    if resolved == workspace or workspace not in resolved.parents:
        raise ValueError(f"Refusing to clean output outside workspace: {path}")

    if path.exists():
        shutil.rmtree(path)


def should_skip(path: Path, include_raw: bool) -> bool:
    if path.suffix.lower() not in REPORT_SUFFIXES:
        return True

    if include_raw:
        return False

    return any(part.lower() in RAW_PARTS for part in path.parts)


def category_for(path: Path) -> str:
    text = "/".join(part.lower() for part in path.parts)
    name = path.name.lower()
    suffix = path.suffix.lower()

    if (
        "performance" in text
        or "runtime" in name
        or "scaling" in text
        or "speedup" in text
        or "benchmark" in text
    ):
        return "performance"

    if (
        "error" in text
        or "convergence" in text
        or "exact" in text
        or "reference" in text
    ):
        return "convergence_error"

    if (
        "interface" in text
        or "conservation" in text
        or "reinit" in text
        or "alpha" in text
        or "material_metrics" in name
    ):
        return "interface_conservation"

    if (
        suffix in {".png", ".html"}
        or "feature" in text
        or "bubble" in text
        or "tracker" in text
        or "schlieren" in text
        or "figures" in text
        or "report_png" in text
    ):
        return "figures_features"

    return "interface_conservation"


def source_label(source_root: Path, path: Path) -> Path:
    relative = path.relative_to(source_root)
    if relative.parts:
        return Path(source_root.name) / relative
    return Path(source_root.name) / path.name


def collect_files(sources: list[Path], output_dir: Path, include_raw: bool) -> list[OrganizedFile]:
    organized: list[OrganizedFile] = []

    for source in sources:
        if not source.exists():
            LOGGER.warning("Skipping missing source: %s", source)
            continue

        for path in sorted(source.rglob("*")):
            if not path.is_file() or should_skip(path, include_raw):
                continue

            category = category_for(path)
            destination = output_dir / category / source_label(source, path)
            organized.append(OrganizedFile(category, path, destination))

    return organized


def copy_files(files: list[OrganizedFile], dry_run: bool) -> None:
    for item in files:
        if dry_run:
            LOGGER.info("Would copy %s -> %s", item.source, item.destination)
            continue

        item.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source, item.destination)


def write_manifest(files: list[OrganizedFile], output_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return

    manifest = output_dir / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["category", "source", "destination"],
        )
        writer.writeheader()
        for item in files:
            writer.writerow({
                "category": item.category,
                "source": item.source.as_posix(),
                "destination": item.destination.as_posix(),
            })


def summarize(files: list[OrganizedFile]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in files:
        counts[item.category] = counts.get(item.category, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    sources = args.sources if args.sources else list(DEFAULT_SOURCES)
    output_dir = args.output_dir

    if args.clean and not args.dry_run:
        safe_clean_output_dir(output_dir)

    files = collect_files(sources, output_dir, args.include_raw)
    copy_files(files, args.dry_run)
    write_manifest(files, output_dir, args.dry_run)

    counts = summarize(files)
    LOGGER.info("Organized %d files into %s", len(files), output_dir)
    for category in sorted(counts):
        LOGGER.info("%s: %d", category, counts[category])


if __name__ == "__main__":
    main()
