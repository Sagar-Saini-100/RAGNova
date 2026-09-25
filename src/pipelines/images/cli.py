"""Command-line interface for image ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.pipelines.images.index import index_image_files, index_images_directory
from src.pipelines.images.pipeline import ImageIngestionPipeline
from src.pipelines.images.models import ImageIngestionConfig

# The corrupt-image error message below (and any other message containing
# an em-dash) renders as "?" on an unconfigured Windows console — the same
# bug already found and fixed in ingest.py, smoke_test.py, and
# example_ingest.py this chapter; confirmed here too, live, while
# verifying the corrupt-image fix itself. Reconfigure both streams since
# this CLI writes JSON to stdout and errors to stderr.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest image(s) with Tesseract OCR + OpenCLIP embeddings producing one Chunk per image."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an image file or a directory containing images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write output JSON chunks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model architecture (default: ViT-B-32).",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained weights tag (default: laion2b_s34b_b79k).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', 'mps'). Default: auto.",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable Tesseract OCR extraction.",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable OpenCLIP embedding generation.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Path to tesseract binary executable.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ImageIngestionConfig(
        ocr_enabled=not args.no_ocr,
        tesseract_cmd=args.tesseract_cmd,
        embedding_enabled=not args.no_embedding,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    pipeline = ImageIngestionPipeline(config=config)
    target = Path(args.input_path)

    if not target.exists():
        sys.stderr.write(f"Error: Path '{target}' does not exist.\n")
        sys.exit(1)

    if target.is_dir():
        # ingest_directory()/ingest_batch() already skip an individual
        # unreadable file with a logged warning rather than raising — a
        # single corrupt image in a real directory scan doesn't need a
        # try/except here.
        chunks = pipeline.ingest_directory(target)
    else:
        # A single explicitly-named file, unlike a directory scan, is
        # exactly the case verified directly (Chapter 7 /verify pass): a
        # text file renamed .png raised a raw PIL.UnidentifiedImageError
        # traceback here, unlike the clean "Error: ..." message the
        # missing-path check above already gives. Same clean treatment now.
        try:
            chunks = [pipeline.ingest_image(target)]
        except ValueError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            sys.exit(1)

    chunk_dicts = [chunk.to_dict() for chunk in chunks]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
        print(f"Ingested {len(chunks)} image(s) -> saved to {out_path}")
    else:
        print(json.dumps(chunk_dicts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
