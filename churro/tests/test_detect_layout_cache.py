"""Tests for Azure Document Intelligence caching in detect_layout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from diskcache import Cache
from PIL import Image
import pytest

from churro.systems import detect_layout


class _StubPoller:
    def __init__(self, content: str) -> None:
        self._result = SimpleNamespace(content=content)

    async def result(self) -> SimpleNamespace:
        return self._result


class _StubDocumentIntelligenceClient:
    def __init__(self, content: str) -> None:
        self.call_count = 0
        self._content = content

    async def begin_analyze_document(self, *args: object, **kwargs: object) -> _StubPoller:
        """Return a stub poller that yields the preconfigured content."""
        self.call_count += 1
        return _StubPoller(self._content)


@pytest.mark.asyncio
async def test_run_azure_document_analysis_on_image_uses_diskcache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """First call should reach Azure client, second call should hit disk cache."""
    original_client = detect_layout.azure_document_intelligence_client
    original_cache = detect_layout.azure_di_cache
    original_total_cost = detect_layout.total_azure_cost

    cache_dir = tmp_path / "azure-di-cache"
    cache = Cache(str(cache_dir))
    stub_client = _StubDocumentIntelligenceClient(content="cached result")

    monkeypatch.setattr(
        detect_layout, "DocumentIntelligenceClient", _StubDocumentIntelligenceClient
    )
    monkeypatch.setattr(detect_layout, "adjust_image", lambda image, thresholding: image)
    monkeypatch.setattr(detect_layout, "_ensure_azure_di_initialized", lambda: None)

    detect_layout.azure_document_intelligence_client = stub_client
    detect_layout.azure_di_cache = cache

    try:
        image = Image.new("RGB", (32, 32), color="white")

        first_result = await detect_layout.run_azure_document_analysis_on_image(
            image=image,
            skip_paragraphs=True,
            output_ocr_text=True,
        )
        second_result = await detect_layout.run_azure_document_analysis_on_image(
            image=image,
            skip_paragraphs=True,
            output_ocr_text=True,
        )

        assert first_result == "cached result"
        assert second_result == "cached result"
        assert stub_client.call_count == 1, "Second invocation should be served from diskcache"
        assert detect_layout.total_azure_cost - original_total_cost == pytest.approx(
            detect_layout.AZURE_DI_COST_PER_PAGE_USD
        )
    finally:
        detect_layout.azure_document_intelligence_client = original_client
        detect_layout.azure_di_cache = original_cache
        detect_layout.total_azure_cost = original_total_cost
        cache.close()
