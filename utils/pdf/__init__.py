"""PDF processing utilities.

This package provides both a simple script-style interface (see
``pdfs_to_images.py``) and an asynchronous queued pipeline (``runner.py``)
for converting PDF documents into one or more processed page images.

Primary public entry point:
    ``run_pdf_pipeline`` – an asyncio based producer/consumer pipeline that:
      1. Rasterizes PDF pages in a separate process pool.
      2. Uses an LLM to decide if a scanned image contains one or two pages.
      3. Optionally trims page content using detected layout.
      4. Persists output PNG files while preserving logical ordering.

Multiprocessing:
    Always uses the ``spawn`` start method to avoid potential fork-related deadlocks with threads or async runtimes.
"""

from .runner import run_pdf_pipeline


__all__ = ["run_pdf_pipeline"]
