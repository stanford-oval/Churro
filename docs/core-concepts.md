# Core Concepts

Churro is easier to understand if you think about it as a document pipeline, not just an OCR call. The library takes an input source, turns it into one or more page objects, runs OCR on those page objects, and returns results that still preserve page-level structure.

If you keep that model in mind, the rest of the API becomes much simpler.

## The Mental Model

Most workflows follow the same shape:

1. Choose an OCR backend.
2. Start from an input source such as an image, a photographed spread, or a PDF.
3. Optionally detect page boundaries.
4. Work with one or more `DocumentPage` objects.
5. Attach OCR output to those pages.

In practice, the flow looks like this:

```text
raw image or PDF
  -> optional page detection
  -> one or more DocumentPage objects
  -> OCR
  -> pages with text, model info, and metadata
```

The key idea is that Churro keeps the page object all the way through the pipeline. You do not lose the cropped page image, page ordering, or page-level metadata just because OCR has been run.

## Start With The Shape Of Your Input

The easiest way to choose an API is to ask what your input already looks like.

| If your input looks like this | Start with | Why |
| --- | --- | --- |
| One image already equals one page | `OCRClient` | no page detection is needed |
| One image may contain multiple pages | `DocumentOCRPipeline` or `DocumentPageDetector` | detect crops first, then OCR |
| A PDF | `DocumentOCRPipeline` | rasterization, page detection, and OCR are handled for you |
| You only want page crops, not text yet | `DocumentPageDetector` | detection-only workflow |
| You want fine control over provider setup | `OCRBackendSpec` + `build_ocr_backend(...)` | backend configuration stays explicit |

This is the most important rule of thumb in the library: do not add page detection unless your input actually needs it.

## The Main Building Blocks

These are the public types most users need to understand.

| Object | What it represents | When you use it |
| --- | --- | --- |
| `OCRBackendSpec` | a declarative description of which provider and model to use | when configuring OCR backends |
| `build_ocr_backend(...)` | the factory that turns a spec into a runnable backend | right after choosing a provider |
| `OCRClient` | OCR for a single page image or a single `DocumentPage` | when each image is already one page |
| `DocumentPageDetector` | page detection without OCR | when you need crops only |
| `DocumentOCRPipeline` | page detection plus OCR in one workflow | when working with photographed spreads or PDFs |
| `DocumentPage` | the central page object passed through detection and OCR | almost everywhere |

For most applications, `DocumentPage` is the type to pay attention to. The other APIs mainly exist to create, transform, or enrich `DocumentPage` objects.

## `DocumentPage` Is The Core Object

A `DocumentPage` is one page image plus whatever Churro knows about that page.

Before OCR, a page may only have:

- an image
- a page position
- source metadata
- crop information such as `bbox` or `polygon`

After OCR, the same page object can also have:

- `text`
- `provider_name`
- `model_name`
- `ocr_metadata`

That makes it easy to treat page detection and OCR as one continuous workflow instead of converting between unrelated result types.

```python
from churro_ocr import DocumentPage, OCRClient
from churro_ocr.providers import OCRBackendSpec, build_ocr_backend

backend = build_ocr_backend(
    OCRBackendSpec(
        provider="litellm",
        model="vertex_ai/gemini-2.5-flash",
    )
)

page = DocumentPage.from_image_path("scan.png")
ocr_page = OCRClient(backend).ocr(page)

print(ocr_page.text)
print(ocr_page.provider_name)
print(ocr_page.model_name)
```

If your input is already one page per image, this is the simplest mental model: create or load a page, run OCR, then read the text from the returned page object.

## How Detection And OCR Fit Together

Churro separates page detection from OCR, but the two parts compose cleanly.

- `DocumentPageDetector` answers: "What are the page crops in this source?"
- `OCRClient` answers: "What text is on this one page?"
- `DocumentOCRPipeline` answers: "Take this document-shaped input and do the whole thing."

That means the library works well across very different input shapes:

- scanned pages where each image is already clean and single-page
- photographed book spreads where one image contains two visible pages
- PDFs that must be rasterized before OCR

If you want concrete usage examples for each case, see [OCR Workflows](guides/ocr-workflows.md) and [Page Detection](guides/page-detection.md).

## What The Result Types Mean

The result containers are small, but they serve different purposes.

| Type | What you get | Typical use |
| --- | --- | --- |
| `DocumentPage` | one page image, with or without OCR attached | most page-level code |
| `OCRResult` | plain OCR output without the page image | backend-facing code or `DocumentOCRResult.as_ocr_results()` |
| `PageDetectionResult` | detected pages from one image or PDF | detection-only workflows |
| `DocumentOCRResult` | OCR output across all pages in a document workflow | PDFs, spreads, or batched page flows |

`OCRResult` is the least user-facing type here. Most application code can stay at the `DocumentPage` or `DocumentOCRResult` level.

`DocumentOCRResult` is especially useful when you want both page structure and convenience helpers:

- `result.pages` keeps the full page objects
- `result.texts()` returns plain text per page
- `result.as_ocr_results()` converts to lightweight OCR-only results

## Understanding `page_index` And `source_index`

These two fields are easy to confuse, but they capture different ideas.

- `page_index` is the page position in the current output.
- `source_index` is the index of the original source item that produced that page.

Examples make this clearer:

- If `scan.png` is a single-page image, the page will usually have `page_index=0` and `source_index=0`.
- If `spread.jpg` contains two detected pages, the output pages may have `page_index=0` and `page_index=1`, but both still came from the same source image, so both have `source_index=0`.
- If a PDF has 10 pages and each PDF page becomes one detected page, the output pages will usually have matching `page_index` and `source_index`.
- If one PDF page is split into multiple detected crops, those crops get different `page_index` values but share the same `source_index` because they came from the same original PDF page.

In short, `page_index` tells you where a page ended up in the output sequence. `source_index` tells you where it came from.

## Understanding `metadata` And `ocr_metadata`

Churro keeps caller-side and provider-side metadata separate on purpose.

- `metadata` is your own metadata, or metadata produced during page detection.
- `ocr_metadata` is metadata returned by the OCR provider for that page.

This separation matters because the two kinds of metadata usually have different meanings.

Examples of `metadata`:

- a job ID you attach when submitting work
- page detection hints
- page ordering or dataset labels

Examples of `ocr_metadata`:

- provider response fields
- usage or timing information
- model-specific OCR details

On document-level results, `source_type` tells you whether the document came from an `"image"` or a `"pdf"` workflow.

## Sync And Async

Every high-level sync entrypoint has an async equivalent.

| Sync | Async |
| --- | --- |
| `ocr(...)` | `aocr(...)` |
| `ocr_image(...)` | `aocr_image(...)` |
| `detect_image_sync(...)` | `detect_image(...)` |
| `process_image_sync(...)` | `process_image(...)` |
| `process_pdf_sync(...)` | `process_pdf(...)` |

The default choice for most users is still the sync API. Use the async forms when:

- you are already inside an async application
- you want to coordinate OCR with other async work
- you want to manage concurrency explicitly

If you use `DocumentOCRPipeline`, the `max_concurrency` setting controls how many page OCR jobs run at once inside that pipeline.

## Practical Rules Of Thumb

- Use `OCRClient` when each input image is already a page.
- Use `DocumentOCRPipeline` for PDFs and photographed spreads.
- Use `DocumentPageDetector` when you want crops without OCR.
- Build your backend once and reuse it across calls.
- Pass exactly one of `image` or `image_path` when an API accepts both.

When you want concrete recipes, continue with [OCR Workflows](guides/ocr-workflows.md). When you need exact signatures and fields, use the [API Reference](api/index.md).
