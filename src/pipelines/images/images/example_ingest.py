"""Example: Ingest a folder of images with the RAG image pipeline.

This script demonstrates end-to-end usage of the image ingestion pipeline:
  1. Tesseract OCR → extracts text from each image
  2. OpenCLIP      → generates a 512-d visual embedding per image
  3. ChunkStore    → stores all chunks in memory (swap for a vector DB)
  4. Summary       → prints a human-readable report

Usage
-----
Run from the repo root so that ``src`` is on the Python path:

    python src/pipelines/images/example_ingest.py --image-dir ./sample_images

Or ingest a single file:

    python src/pipelines/images/example_ingest.py --image-path ./photo.jpg

Optional flags
--------------
--no-ocr          Skip Tesseract OCR (useful if Tesseract is not installed).
--no-embedding    Skip OpenCLIP embedding (useful for quick smoke tests).
--output JSON     Path to write the ingested chunks as a JSON file.
--tesseract-cmd   Explicit path to the tesseract binary.
--device          Compute device for OpenCLIP: "cpu", "cuda", or "mps".
--verbose         Enable DEBUG-level logging.

Dependencies
------------
pip install pillow pytesseract open-clip-torch torch torchvision

Tesseract binary (separate install):
  - Windows : https://github.com/UB-Mannheim/tesseract/wiki
  - macOS   : brew install tesseract
  - Linux   : sudo apt install tesseract-ocr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure 'src' is importable when running from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipelines.images import (       # noqa: E402  (after sys.path fix)
    ChunkStore,
    ImageIngestionConfig,
    ImageIngestionPipeline,
)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a folder (or single file) of images into RAG Chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image-dir",
        type=str,
        metavar="DIR",
        help="Directory of images to ingest (PNG, JPG, WEBP, TIFF, BMP, GIF).",
    )
    group.add_argument(
        "--image-path",
        type=str,
        metavar="FILE",
        help="Single image file to ingest.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str, default=None, metavar="JSON",
        help="Write ingested chunks to this JSON file.",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable Tesseract OCR.")
    parser.add_argument("--no-embedding", action="store_true", help="Disable OpenCLIP embeddings.")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag.")
    parser.add_argument("--device", type=str, default=None, help="Compute device: cpu | cuda | mps.")
    parser.add_argument("--tesseract-cmd", type=str, default=None, metavar="PATH", help="Path to tesseract binary.")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_chunk_summary(chunks) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Ingestion complete — {len(chunks)} chunk(s) produced")
    print(sep)

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        ocr_meta = meta.get("ocr", {})
        ocr_words = ocr_meta.get("words_count", 0)
        ocr_conf = ocr_meta.get("confidence", ocr_meta.get("avg_confidence", 0.0))
        src = chunk.source or meta.get("source_path") or "<no path>"
        emb_dim = len(chunk.embedding) if chunk.embedding else 0

        print(f"\n  [{i}] id            : {chunk.chunk_id}")
        print(f"       modality      : {chunk.modality}")
        print(f"       source        : {src}")
        print(f"       image size    : {meta.get('width')}×{meta.get('height')} px")
        print(f"       format        : {meta.get('format')} / {meta.get('mode')}")
        print(f"       file_size     : {meta.get('file_size_bytes', 0):,} bytes")
        print(f"       ocr words     : {ocr_words}  (avg confidence {ocr_conf:.1f}%)")
        print(f"       embedding dim : {emb_dim}  (model: {chunk.embedding_model})")
        if chunk.text:
            preview = chunk.text[:120].replace("\n", " ")
            ellipsis = "…" if len(chunk.text) > 120 else ""
            print(f"       ocr preview   : \"{preview}{ellipsis}\"")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build configuration
    config = ImageIngestionConfig(
        ocr_enabled=not args.no_ocr,
        tesseract_cmd=args.tesseract_cmd,
        embedding_enabled=not args.no_embedding,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    # Build pipeline with a shared in-memory store
    store = ChunkStore(deduplicate=True)
    pipeline = ImageIngestionPipeline(config=config, store=store)

    # Run ingestion
    if args.image_dir:
        target = Path(args.image_dir)
        if not target.is_dir():
            sys.stderr.write(f"Error: '{target}' is not a directory.\n")
            sys.exit(1)
        print(f"Ingesting directory: {target.resolve()}")
        chunks = pipeline.ingest_directory(target)
    else:
        target = Path(args.image_path)
        if not target.is_file():
            sys.stderr.write(f"Error: '{target}' is not a file.\n")
            sys.exit(1)
        print(f"Ingesting single image: {target.resolve()}")
        chunks = [pipeline.ingest_image(target)]

    # Summary
    _print_chunk_summary(chunks)

    # Optional JSON export
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to: {out_path.resolve()}\n")

    # Demonstrate ChunkStore query API
    print(f"ChunkStore now holds {len(store)} chunk(s).")
    if chunks:
        first = store.get(chunks[0].chunk_id)
        print(f"store.get('{chunks[0].chunk_id}') → Chunk(modality={first.modality!r})\n")


# ---------------------------------------------------------------------------
# Programmatic usage (importable reference, not called by CLI)
# ---------------------------------------------------------------------------

def programmatic_example(image_folder: str) -> None:
    """Copy-pasteable reference for integrating the pipeline into other code.

    Parameters
    ----------
    image_folder:
        Path to a folder containing images to ingest.
    """
    # 1. Configure
    config = ImageIngestionConfig(
        ocr_enabled=True,
        embedding_enabled=True,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        normalize_embeddings=True,
        preprocess_for_ocr=True,
        ocr_lang="eng",
    )

    # 2. Build pipeline
    store = ChunkStore(deduplicate=True)
    pipeline = ImageIngestionPipeline(config=config, store=store)

    # 3. Ingest folder recursively
    chunks = pipeline.ingest_directory(image_folder)

    # 4. Inspect each chunk
    for chunk in chunks:
        print(f"chunk_id={chunk.chunk_id}")
        print(f"  modality       : {chunk.modality}")
        print(f"  embedding_model: {chunk.embedding_model}")
        print(f"  source         : {chunk.source}")
        print(f"  ocr_text       : {chunk.text[:80]!r}")
        print(f"  embedding_dim  : {len(chunk.embedding) if chunk.embedding else 0}")
        print(f"  metadata_keys  : {list(chunk.metadata.keys())}")
        print()

    # 5. Retrieve from store
    if chunks:
        retrieved = store.get(chunks[0].chunk_id)
        assert retrieved is not None
        print(f"Retrieved from store: {retrieved.chunk_id}")

    # 6. Export to JSON
    all_dicts = [c.to_dict() for c in store.all()]
    json_str = json.dumps(all_dicts, indent=2)
    print(f"JSON output ({len(json_str)} chars)")


if __name__ == "__main__":
    main()

