from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from churro_ocr.page_detection import DocumentPage, DocumentPageDetector, PageDetectionRequest
from churro_ocr.providers import LiteLLMTransportConfig, LLMPageDetector

_LIVE_FLAG = "CHURRO_RUN_LIVE_VERTEX_TESTS"
_MODEL = "vertex_ai/gemini-3.1-pro-preview"
_ARTIFACT_DIRNAME = "test-artifacts"


def _load_dotenv_if_present(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key or key in os.environ:
            continue
        if key == "GOOGLE_APPLICATION_CREDENTIALS" and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                value = str((path.parent / candidate).resolve())
        os.environ[key] = value


def _load_local_vertex_env() -> None:
    churro_root = Path(__file__).resolve().parents[1]
    repo_root = churro_root.parent
    _load_dotenv_if_present(repo_root / ".env")
    _load_dotenv_if_present(churro_root / ".env")


def _artifact_dir() -> Path:
    churro_root = Path(__file__).resolve().parents[1]
    artifact_dir = churro_root / "workdir" / _ARTIFACT_DIRNAME
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _write_synthetic_page_image(path: Path) -> tuple[int, int]:
    width = 1200
    height = 900
    image = Image.new("RGB", (width, height), color=(242, 240, 236))
    draw = ImageDraw.Draw(image)

    left = 140
    top = 90
    right = 1020
    bottom = 800
    draw.rectangle((left + 20, top + 24, right + 30, bottom + 34), fill=(184, 178, 166))
    draw.rectangle((left, top, right, bottom), fill=(255, 252, 244), outline=(60, 56, 50), width=6)

    header_y = top + 70
    draw.rectangle((left + 90, header_y, right - 90, header_y + 16), fill=(48, 44, 40))
    body_top = header_y + 70
    for index in range(12):
        line_top = body_top + (index * 34)
        line_right = right - 100 - ((index % 3) * 80)
        draw.rectangle((left + 80, line_top, line_right, line_top + 10), fill=(70, 66, 61))

    note_box = (right - 250, bottom - 170, right - 70, bottom - 70)
    draw.rectangle(note_box, outline=(90, 70, 40), width=4)
    draw.line((left - 40, top + 180, left - 10, top + 230), fill=(150, 150, 150), width=4)
    image.save(path)
    return width, height


def _save_detection_overlay(
    *,
    source_path: Path,
    overlay_path: Path,
    pages: list[DocumentPage],
) -> None:
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for page in pages:
        if page.polygon:
            draw.polygon(page.polygon, outline=(220, 40, 40), width=8)
            label_anchor = page.polygon[0]
        elif page.bbox is not None:
            draw.rectangle(page.bbox, outline=(220, 40, 40), width=8)
            label_anchor = (page.bbox[0], page.bbox[1])
        else:
            continue
        draw.text((label_anchor[0] + 12, label_anchor[1] + 12), f"page {page.page_index}", fill=(20, 90, 220))
    image.save(overlay_path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_page_detector_live_vertex_gemini_31_pro() -> None:
    if os.getenv(_LIVE_FLAG) != "1":
        pytest.skip(f"Set {_LIVE_FLAG}=1 to run live Vertex page-detection integration tests.")

    _load_local_vertex_env()
    missing = [key for key in ("GOOGLE_CLOUD_PROJECT", "VERTEX_AI_LOCATION") if not os.getenv(key)]
    if missing:
        pytest.skip(f"Missing required Vertex env vars: {', '.join(missing)}")

    image_path = _artifact_dir() / "vertex-page-detection.png"
    width, height = _write_synthetic_page_image(image_path)

    detector = DocumentPageDetector(
        backend=LLMPageDetector(
            model=_MODEL,
            transport=LiteLLMTransportConfig(
                image_detail="",
                completion_kwargs={
                    "vertex_location": os.environ["VERTEX_AI_LOCATION"],
                    "reasoning_effort": "high",
                },
            ),
            max_review_rounds=1,
        )
    )

    result = await detector.detect_image(PageDetectionRequest(image_path=image_path, trim_margin=0))
    overlay_path = _artifact_dir() / "vertex-page-detection-overlay.png"
    _save_detection_overlay(
        source_path=image_path,
        overlay_path=overlay_path,
        pages=result.pages,
    )

    assert result.source_type == "image"
    assert len(result.pages) >= 1

    first_page = result.pages[0]
    assert first_page.metadata["detector"] == "llm"
    assert first_page.bbox is not None
    left, top, right, bottom = first_page.bbox
    assert 0.0 <= left < right <= float(width)
    assert 0.0 <= top < bottom <= float(height)
    assert left > 40.0
    assert top > 30.0
    assert right < float(width - 40)
    assert bottom < float(height - 40)
    assert first_page.width < width
    assert first_page.height < height
