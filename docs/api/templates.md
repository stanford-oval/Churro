# Template APIs

`churro_ocr.templates` is a convenience namespace that re-exports template protocols, chat-template helpers, and built-in model presets from the owning modules below.

Use the canonical module that defines each symbol:

| Convenience import | Canonical reference |
| --- | --- |
| `OCRConversation`, `OCRPromptTemplate`, `OCRPromptTemplateCallable`, `OCRPromptTemplateLike`, `build_ocr_conversation(...)` | `churro_ocr.templates.base` |
| `HFChatTemplate` | `churro_ocr.templates.hf` |
| `DEFAULT_OCR_TEMPLATE`, model ids, and built-in prompt presets | `churro_ocr.templates.presets` |

## `churro_ocr.templates.base`

```{eval-rst}
.. automodule:: churro_ocr.templates.base
   :members:
   :show-inheritance:
```

## `churro_ocr.templates.hf`

```{eval-rst}
.. automodule:: churro_ocr.templates.hf
   :members:
   :show-inheritance:
```

## `churro_ocr.templates.presets`

```{eval-rst}
.. automodule:: churro_ocr.templates.presets
   :members:
   :show-inheritance:
```
