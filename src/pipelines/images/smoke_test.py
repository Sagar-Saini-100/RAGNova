"""Smoke test for the image ingestion pipeline.

Run from the repo root (d:\\dOWNLOADS):

    python src/pipelines/images/smoke_test.py

Tests performed
---------------
1. Import check      — all pipeline modules import without error
2. Synthetic image   — creates a PIL image in-memory (no file needed)
3. OCR engine        — TesseractOCREngine init + extract_text (graceful degradation)
4. OpenCLIP embedder — embed_image + embed_text (skipped if torch not installed)
5. Full pipeline     — ImageIngestionPipeline.ingest_image() on synthetic image
6. ChunkStore        — add / get / dedup / remove
7. Chunk validation  — modality, embedding_model, metadata keys
8. Serialisation     — to_dict / to_json / from_dict round-trip
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# ── ensure repo root is on sys.path ──────────────────────────────────────
# This file lives at src/pipelines/images/smoke_test.py
# Repo root is three levels up
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results: list[tuple[str, str, str]] = []   # (name, status, detail)


def check(name: str):
    """Simple context manager for a single named test."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        try:
            yield
            results.append((name, PASS, ""))
        except Exception as exc:
            results.append((name, FAIL, str(exc)))
            traceback.print_exc()
    return _ctx()


# ---------------------------------------------------------------------------
# 1. Import check
# ---------------------------------------------------------------------------
with check("1. Import — src.pipelines.images"):
    from src.pipelines.images import (
        ChunkStore,
        ImageIngestionConfig,
        ImageIngestionPipeline,
        OpenCLIPEmbedder,
        TesseractOCREngine,
    )
    from src.models.chunk import Chunk

# ---------------------------------------------------------------------------
# 2. Synthetic image (no file needed)
# ---------------------------------------------------------------------------
with check("2. Synthetic PIL image creation"):
    import random
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), color=(30, 30, 60))
    draw = ImageDraw.Draw(img)
    # White boxes with readable text so OCR has something to detect
    draw.rectangle([20, 20, 620, 100], fill=(255, 255, 255))
    draw.text((30, 35), "Hello OCR World 1234", fill=(0, 0, 0))
    draw.rectangle([20, 120, 620, 200], fill=(200, 220, 255))
    draw.text((30, 135), "RAG Image Pipeline Test", fill=(10, 10, 80))
    # Random coloured blocks to give the embedder some visual variation
    for _ in range(10):
        x = random.randint(0, 580)
        y = random.randint(210, 440)
        color = tuple(random.randint(50, 255) for _ in range(3))
        draw.rectangle([x, y, x + 50, y + 40], fill=color)  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# 3. OCR engine
# ---------------------------------------------------------------------------
with check("3. TesseractOCREngine — init"):
    config_ocr_only = ImageIngestionConfig(
        ocr_enabled=True,
        embedding_enabled=False,
    )
    ocr = TesseractOCREngine(config_ocr_only)
    print(f"     Tesseract available: {ocr.is_available}")

with check("4. TesseractOCREngine — extract_text (graceful on unavailable)"):
    result = ocr.extract_text(img)
    print(f"     OCR text    : {result.text[:80]!r}")
    print(f"     words_count : {result.words_count}")
    print(f"     confidence  : {result.confidence}")
    print(f"     status      : {result.metadata.get('status')}")
    # Must always return an OCRResult — never raise
    assert hasattr(result, "text")
    assert hasattr(result, "confidence")
    assert hasattr(result, "words_count")

# ---------------------------------------------------------------------------
# 4. OpenCLIP embedder  (skip if open_clip / torch not installed)
# ---------------------------------------------------------------------------
_OPENCLIP_OK = False
try:
    import open_clip  # type: ignore  # noqa: F401
    import torch      # type: ignore  # noqa: F401
    _OPENCLIP_OK = True
except ImportError:
    pass

if _OPENCLIP_OK:
    with check("5. OpenCLIPEmbedder — embed_image (ViT-B-32)"):
        embed_config = ImageIngestionConfig(
            ocr_enabled=False,
            embedding_enabled=True,
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        embedder = OpenCLIPEmbedder(embed_config)
        vec = embedder.embed_image(img)
        print(f"     embedding dim  : {len(vec)}")
        print(f"     first 5 values : {[round(v, 4) for v in vec[:5]]}")
        assert len(vec) == 512, f"Expected 512-d vector, got {len(vec)}"

    with check("6. OpenCLIPEmbedder — embed_text (multimodal)"):
        text_vec = embedder.embed_text("a photo of a document with text")
        assert len(text_vec) == 512
        print(f"     text embedding dim: {len(text_vec)}")
else:
    results.append(("5. OpenCLIPEmbedder — embed_image", SKIP, "open_clip/torch not installed"))
    results.append(("6. OpenCLIPEmbedder — embed_text",  SKIP, "open_clip/torch not installed"))

# ---------------------------------------------------------------------------
# 5. Full pipeline — ingest_image on PIL image
# ---------------------------------------------------------------------------
with check("7. ImageIngestionPipeline — ingest_image (PIL input)"):
    full_config = ImageIngestionConfig(
        ocr_enabled=True,
        embedding_enabled=_OPENCLIP_OK,
    )
    store = ChunkStore(deduplicate=True)
    pipeline = ImageIngestionPipeline(config=full_config, store=store)

    chunk: Chunk = pipeline.ingest_image(img)

    print(f"     chunk.id             : {chunk.id}")
    print(f"     chunk.modality       : {chunk.modality}")
    print(f"     chunk.embedding_model: {chunk.embedding_model}")
    print(f"     chunk.content[:60]   : {chunk.content[:60]!r}")
    print(f"     embedding present    : {chunk.embedding is not None}")
    print(f"     metadata keys        : {list(chunk.metadata.keys())}")

    assert chunk.modality == "image",               f"Expected 'image', got {chunk.modality!r}"
    assert chunk.embedding_model == "ViT-B-32/laion2b_s34b_b79k"
    assert "ocr" in chunk.metadata
    assert "width" in chunk.metadata
    assert "height" in chunk.metadata

# ---------------------------------------------------------------------------
# 6. ChunkStore operations
# ---------------------------------------------------------------------------
with check("8. ChunkStore — add / get / dedup / remove"):
    assert len(store) == 1, f"Expected 1 chunk in store, got {len(store)}"
    retrieved = store.get(chunk.id)
    assert retrieved is not None
    assert retrieved.id == chunk.id
    # Deduplication — same chunk should be rejected
    stored_again = store.add(chunk)
    assert stored_again is False, "Duplicate should have been rejected"
    assert len(store) == 1
    # Remove
    store.remove(chunk.id)
    assert len(store) == 0

# ---------------------------------------------------------------------------
# 7. Chunk serialisation
# ---------------------------------------------------------------------------
with check("9. Chunk.to_dict / to_json / from_dict round-trip"):
    import json
    d = chunk.to_dict()
    j = chunk.to_json()
    from_dict_chunk = chunk.__class__.from_dict(d)
    assert from_dict_chunk.id == chunk.id
    assert from_dict_chunk.modality == "image"
    parsed = json.loads(j)
    assert parsed["embedding_model"] == "ViT-B-32/laion2b_s34b_b79k"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print()
print(f"{BOLD}{'─' * 62}{RESET}")
print(f"{BOLD}  Smoke Test Results{RESET}")
print(f"{'─' * 62}")
for name, status, detail in results:
    tail = f"  ({detail})" if detail else ""
    print(f"  {status}  {name}{tail}")
print(f"{'─' * 62}")

passed  = sum(1 for _, s, _ in results if s == PASS)
skipped = sum(1 for _, s, _ in results if s == SKIP)
failed  = sum(1 for _, s, _ in results if s == FAIL)
total   = len(results)
print(f"  {BOLD}{passed}/{total} passed  |  {skipped} skipped  |  {failed} failed{RESET}")
print(f"{'─' * 62}\n")

sys.exit(1 if failed else 0)
