"""Tests for `run_pdf_pipeline` in `utils/pdf/`.

This test focuses on verifying that image files are produced for a PDF input.
We rely on the sample PDF `minimal-document.pdf` housed next to this test.
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz
from PIL import Image
import pytest

from churro.utils.pdf import run_pdf_pipeline


@pytest.mark.asyncio
async def test_process_one_pdf_file(tmp_path: Path) -> None:
    """Process a PDF and ensure at least one output PNG is created.

    Assertions:
      * Exactly `total_pages` output PNG files created (since no splitting occurs)
      * Filenames follow the expected naming pattern `<pdf_base>_page_XXXX.png`
      * Saved images can be opened by Pillow
    """
    # Arrange
    sample_pdf = Path(__file__).with_name("minimal-document.pdf")
    assert sample_pdf.exists(), "Sample PDF missing. Ensure it was downloaded before running tests."

    # Determine real page count so our stub aligns with batching logic
    with fitz.open(sample_pdf.as_posix()) as doc:  # type: ignore[attr-defined]
        total_pages = doc.page_count

    # Defensive: sample file should have at least one page
    assert total_pages >= 1

    # Act
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    await run_pdf_pipeline(
        pdf_paths=[sample_pdf.as_posix()],
        output_dir=output_dir.as_posix(),
        engine="gpt-4.1",
    )

    # Assert
    created_files = sorted(p for p in output_dir.glob("*.png"))
    assert len(created_files) == total_pages, (
        f"Expected {total_pages} output images, found {len(created_files)}"
    )

    # Validate naming pattern and that files are readable images
    base_name = sample_pdf.stem.replace(" ", "_")
    for idx, file_path in enumerate(created_files):
        expected = f"{base_name}_page_{idx:04d}.png"
        assert file_path.name == expected, (
            f"Unexpected filename {file_path.name}, expected {expected}"
        )
        with Image.open(file_path) as img:
            img.verify()  # Ensures file is a valid image

    # Cleanup is implicit via tmp_path fixture; still assert directory removal works when manually invoked
    # This ensures no lingering file locks.
    for f in created_files:
        os.remove(f)
    assert not any(output_dir.glob("*.png")), "Output PNG files not cleaned up"


@pytest.mark.asyncio
async def test_process_image_directory(tmp_path: Path) -> None:
    image_dir = Path(__file__).parent

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    await run_pdf_pipeline(
        pdf_paths=[],
        output_dir=output_dir.as_posix(),
        engine="gpt-4.1",
        trim=False,
        image_dir=image_dir.as_posix(),
    )

    created_files = sorted(output_dir.glob("*.png"))
    assert len(created_files) == 3, (
        f"Expected 3 output images, found {len(created_files)}"
    )  # The directory contains a one-page image and a two-page image
    for file_path in created_files:
        with Image.open(file_path) as img:
            img.verify()  # Ensures file is a valid image
