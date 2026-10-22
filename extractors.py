
from pathlib import Path
import hashlib
import json
import sys

import pymupdf
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_DIR = Path(r"C:\PythonInten\data\input")
OUTPUT_DIR = Path(r"C:\PythonInten\data\extracted")

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".md", ".markdown"}


# --------------------------------------------------
# File utilities
# --------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# --------------------------------------------------
# PDF extraction
# --------------------------------------------------

def extract_pdf(path: Path) -> list[dict]:
    """Extract text from each PDF page."""
    units = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                units.append({
                    "text": text,
                    "page_number": page_number,
                })

    return units


# --------------------------------------------------
# CSV extraction
# --------------------------------------------------

def extract_csv(path: Path) -> list[dict]:
    """Convert each CSV row into searchable text."""
    units = []

    for frame in pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        chunksize=1000,
        encoding="utf-8-sig",
    ):
        for row in frame.to_dict(orient="records"):
            text = "\n".join(
                f"{key}: {value}"
                for key, value in row.items()
                if str(value).strip()
            )

            if text.strip():
                units.append({
                    "text": text,
                    "row_number": len(units) + 1,
                })

    return units


# --------------------------------------------------
# Markdown extraction
# --------------------------------------------------

def extract_markdown(path: Path) -> list[dict]:
    """Read Markdown while preserving its headings and formatting."""
    text = path.read_text(encoding="utf-8-sig")

    if not text.strip():
        return []

    return [{"text": text}]


# --------------------------------------------------
# Main extraction function
# --------------------------------------------------

def extract_document(path: str | Path) -> dict:
    """Extract a supported document and preserve its metadata."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        units = extract_pdf(path)
    elif extension == ".csv":
        units = extract_csv(path)
    else:
        units = extract_markdown(path)

    return {
        "source": path.name,
        "source_path": str(path.resolve()),
        "file_type": extension.lstrip("."),
        "content_hash": sha256_file(path),
        "units": units,
        "unit_count": len(units),
    }


# --------------------------------------------------
# Batch processing
# --------------------------------------------------

def process_directory(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Extract every supported document in the input directory."""
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created input directory: {input_dir}")
        print("Add PDF, CSV, or Markdown files and run again.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported files found in: {input_dir}")
        print("Supported formats: PDF, CSV, Markdown")
        return

    successful = 0
    failed = 0

    print(f"\nFound {len(files)} document(s).\n")

    for path in files:
        try:
            result = extract_document(path)

            # Store each document's extraction output as JSON.
            output_file = (
                output_dir / f"{path.name}.extracted.json"
            )

            output_file.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print(
                f"[SUCCESS] {path.name} | "
                f"Units: {result['unit_count']} | "
                f"Type: {result['file_type']}"
            )

            successful += 1

        except Exception as error:
            print(f"[FAILED] {path}: {error}")
            failed += 1

    print("\n---------- Extraction Summary ----------")
    print(f"Total files : {len(files)}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {failed}")
    print(f"Output      : {output_dir}")
    print("-----------------------------------------\n")


# --------------------------------------------------
# Program entry point
# --------------------------------------------------

if __name__ == "__main__":
    try:
        process_directory()
    except KeyboardInterrupt:
        print("\nExtraction cancelled.")
        sys.exit(1)
