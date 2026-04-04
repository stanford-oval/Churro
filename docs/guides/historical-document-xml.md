# HistoricalDocument XML

`HistoricalDocument` is the structured XML format used by CHURRO for rich transcriptions of historical sources.

Use it when you want more than plain OCR text, for example when you need:

- page-level structure rather than one flat string
- header, body, and footer separation
- inline markup for additions, deletions, and missing text
- a representation that can preserve reading order while still carrying layout-aware detail

You do **not** need this format to use the `churro-ocr` library. The public OCR APIs work perfectly well with plain text in `page.text`. This guide is specifically for understanding CHURRO-style structured outputs and the repo-local evaluation helpers around them.

## Example

```xml
<HistoricalDocument xmlns="http://example.com/historicaldocument">
  <Metadata>
    <Language>lat</Language>
    <Script>Latn</Script>
  </Metadata>
  <Page>
    <Header>
      <Line>Anno domini 1451</Line>
    </Header>
    <Body>
      <Paragraph>
        <Line>In nomine domini amen.</Line>
        <Line>
          Nos <Addition>humiles</Addition> notarii
          <Gap reason="illegible"/>
          subscripsimus.
        </Line>
      </Paragraph>
      <MarginalNote>
        <Line>Memorandum de censu.</Line>
      </MarginalNote>
    </Body>
    <Footer>
      <Line>Explicit.</Line>
    </Footer>
  </Page>
</HistoricalDocument>
```

That example shows the main idea:

- document metadata is separate from page content
- page content is divided into logical sections
- inline editorial markup can appear inside lines without losing reading order

## Document Structure

A typical document has:

- a root `<HistoricalDocument>` element
- optional `<Metadata>` describing languages, scripts, writing direction, or notes
- one or more `<Page>` blocks
- optional `<Header>` and `<Footer>` sections per page
- a `<Body>` section containing the main reading-order content

Within a page body, CHURRO-style XML can include structural tags such as:

- `<Paragraph>`
- `<MarginalNote>`
- `<Figure>`
- `<List>`

It can also include inline markup such as:

- `<Addition>`
- `<Deletion>`
- `<Gap/>`
- `<InterlinearNote>`
- `<Illegible>`

Those tags let the output capture features that plain OCR text would otherwise lose.

## Why Use XML Instead Of Plain Text?

Plain OCR text is usually the right output when you only need readable transcription. Structured XML is useful when you need to preserve editorial or layout-aware distinctions that affect downstream work.

Common cases:

- scholarly editing or diplomatic transcription
- distinguishing marginal notes from body text
- keeping track of gaps or illegible regions
- preserving additions or deletions for later analysis
- running evaluation on structured outputs before flattening them

## How The Repo Flattens XML For Evaluation

The repo-local evaluation helper `tooling.evaluation.xml_utils.extract_actual_text_from_xml()` converts `HistoricalDocument` XML into plain text for benchmarking and metric calculation.

Its behavior is intentionally lossy:

- if the input does not contain `HistoricalDocument`, it returns the input unchanged
- it removes `<Description>`, `<Deletion>`, `<Illegible>`, and `<Gap>` before parsing
- it then extracts text from each page’s `<Header>`, `<Body>`, and `<Footer>` in document order
- it joins sections within a page using single newlines
- it joins pages using blank lines
- if XML parsing fails, it returns an empty string

That means evaluation is focused on readable recovered text, not on preserving every markup distinction in the structured output.

## What Gets Lost During Flattening

Once XML is reduced to plain text:

- section boundaries become newline conventions
- deleted or illegible spans are dropped
- gaps are removed rather than represented explicitly
- metadata is not preserved
- structural distinctions such as paragraph versus marginal note are flattened into text order

If you care about those distinctions, keep the XML as your source of truth and flatten only for downstream tasks that truly require plain text.

## Namespace Handling

The evaluation helper matches elements by local tag name, so namespaced XML still works as long as the local names are the expected ones such as `HistoricalDocument`, `Page`, `Header`, `Body`, and `Footer`.

This is why XML such as:

```xml
<HistoricalDocument xmlns="urn:test">
  <Page>
    <Body>
      <Line>Example text.</Line>
    </Body>
  </Page>
</HistoricalDocument>
```

can still be flattened correctly by the evaluation tooling.

## Practical Guidance

- Store the raw XML if you may need structure later.
- Use flattened text only for search, quick display, or text-level evaluation metrics.
- Treat the flattened output as a derived view, not as a lossless representation.
- When comparing systems, make sure they are flattened the same way before scoring.
