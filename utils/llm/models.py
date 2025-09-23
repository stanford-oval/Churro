"""Model registry mapping logical model keys to provider candidates."""

from .config import GEMINI_SAFETY_SETTINGS, get_settings
from .types import ModelInfo


COMPLETION_TOKENS_FOR_REASONING_MODELS = 40_000
COMPLETION_TOKENS_FOR_STANDARD_MODELS = 20_000
CHURRO_MODEL_ID: str = "stanford-oval/churro-3B"

MODEL_MAP: dict[str, list[ModelInfo]] = {
    # Azure/OpenAI GPT family (standard mode)
    "gpt-5-low": [
        {
            "provider_model": "azure/gpt-5",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
        {
            "provider_model": "gpt-5",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
    ],
    "gpt-5-medium": [
        {
            "provider_model": "azure/gpt-5",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
        {
            "provider_model": "gpt-5",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
    ],
    "gpt-5-mini-low": [
        {
            "provider_model": "azure/gpt-5-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
        {
            "provider_model": "gpt-5-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
    ],
    "gpt-5-mini-medium": [
        {
            "provider_model": "azure/gpt-5-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
        {
            "provider_model": "gpt-5-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
    ],
    "gpt-5-nano-low": [
        {
            "provider_model": "azure/gpt-5-nano",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
        {
            "provider_model": "gpt-5-nano",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
    ],
    "gpt-5-nano-medium": [
        {
            "provider_model": "azure/gpt-5-nano",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
        {
            "provider_model": "gpt-5-nano",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
    ],
    "gpt-4o": [
        {"provider_model": "azure/gpt-4o", "max_completion_tokens": None},
        {"provider_model": "gpt-4o", "max_completion_tokens": None},
    ],
    "gpt-4o-mini": [
        {
            "provider_model": "azure/gpt-4o-mini",
            "max_completion_tokens": None,
        },
        {
            "provider_model": "gpt-4o-mini",
            "max_completion_tokens": None,
        },
    ],
    "gpt-4.1": [
        {"provider_model": "azure/gpt-4.1", "max_completion_tokens": None},
        {"provider_model": "gpt-4.1", "max_completion_tokens": None},
    ],
    "gpt-4.1-mini": [
        {
            "provider_model": "azure/gpt-4.1-mini",
            "max_completion_tokens": None,
        },
        {
            "provider_model": "gpt-4.1-mini",
            "max_completion_tokens": None,
        },
    ],
    "gpt-4.1-nano": [
        {
            "provider_model": "azure/gpt-4.1-nano",
            "max_completion_tokens": None,
        },
        {
            "provider_model": "gpt-4.1-nano",
            "max_completion_tokens": None,
        },
    ],
    # Reasoning models: use suffixed keys to denote effort level
    "o1-low": [
        {
            "provider_model": "azure/o1",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        }
    ],
    "o1-medium": [
        {
            "provider_model": "azure/o1",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        }
    ],
    "o3-low": [
        {
            "provider_model": "o3",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        }
    ],
    "o3-medium": [
        {
            "provider_model": "o3",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        }
    ],
    "o4-mini-low": [
        {
            "provider_model": "azure/o4-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
        {
            "provider_model": "o4-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "low"},
        },
    ],
    "o4-mini-medium": [
        {
            "provider_model": "azure/o4-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
        {
            "provider_model": "o4-mini",
            "max_completion_tokens": None,
            "static_params": {"reasoning_effort": "medium"},
        },
    ],
    "chatgpt-4o": [
        {
            "provider_model": "chatgpt-4o-latest",
            "max_completion_tokens": None,
        },
    ],
    # Gemini on Vertex AI
    "gemini-2.5-flash-noreasoning": [
        {
            "provider_model": "vertex_ai/gemini-2.5-flash",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "disable",
            },
        }
    ],
    "gemini-2.5-flash-low": [
        {
            "provider_model": "vertex_ai/gemini-2.5-flash",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "low",
            },
        }
    ],
    "gemini-2.5-pro-low": [
        {
            "provider_model": "vertex_ai/gemini-2.5-pro",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "low",
            },
        }
    ],
    "gemini-2.5-flash-medium": [
        {
            "provider_model": "vertex_ai/gemini-2.5-flash",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "medium",
            },
        }
    ],
    "gemini-2.5-pro-medium": [
        {
            "provider_model": "vertex_ai/gemini-2.5-pro",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "medium",
            },
        }
    ],
    "gemini-2.5-flash-high": [
        {
            "provider_model": "vertex_ai/gemini-2.5-flash",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "high",
            },
        }
    ],
    "gemini-2.5-pro-high": [
        {
            "provider_model": "vertex_ai/gemini-2.5-pro",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "safety_settings": GEMINI_SAFETY_SETTINGS,
                "vertex_location": get_settings().vertex_ai_location,
                "reasoning_effort": "high",
            },
        }
    ],
    # Claude on Vertex and Anthropic
    # Try Vertex first, then Anthropic
    "sonnet-3.7": [
        {
            "provider_model": "vertex_ai/claude-3-7-sonnet@20250219",
            "max_completion_tokens": None,
            "static_params": {
                "thinking": {"type": "disabled"},
                "vertex_location": get_settings().vertex_ai_location,
            },
        },
    ],
    "sonnet-3.7-low": [
        {
            "provider_model": "vertex_ai/claude-3-7-sonnet@20250219",
            "max_completion_tokens": None,
            "static_params": {
                "reasoning_effort": "low",
                "vertex_location": get_settings().vertex_ai_location,
            },
        },
    ],
    "sonnet-3.7-medium": [
        {
            "provider_model": "vertex_ai/claude-3-7-sonnet@20250219",
            "max_completion_tokens": None,
            "static_params": {
                "reasoning_effort": "medium",
                "vertex_location": get_settings().vertex_ai_location,
            },
        },
    ],
    "sonnet-4-medium": [
        {
            "provider_model": "vertex_ai/claude-sonnet-4@20250514",
            "max_completion_tokens": None,
            "static_params": {
                "reasoning_effort": "medium",
                "vertex_location": get_settings().vertex_ai_location,
            },
        },
    ],
    "opus-4.1-medium": [
        {
            "provider_model": "vertex_ai/claude-opus-4-1@20250805",
            "max_completion_tokens": None,
            "static_params": {
                "reasoning_effort": "medium",
                "vertex_location": get_settings().vertex_ai_location,
            },
        },
    ],
    # Hosted vLLM models
    "qwen25-3b": [
        {
            "provider_model": "vllm/qwen25-3b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        }
    ],
    "qwen25-7b": [
        {
            "provider_model": "vllm/qwen25-7b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
        }
    ],
    "qwen25-32b": [
        {
            "provider_model": "vllm/qwen25-32b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "Qwen/Qwen2.5-VL-32B-Instruct",
        }
    ],
    "qwen25-72b": [
        {
            "provider_model": "vllm/qwen25-72b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "Qwen/Qwen2.5-VL-72B-Instruct",
        }
    ],
    "aria": [
        {
            "provider_model": "vllm/aria",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "rhymes-ai/Aria",
        }
    ],
    "phi4-multimodal": [
        {
            "provider_model": "vllm/phi4-multimodal",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "microsoft/Phi-4-multimodal-instruct",
        }
    ],
    "aya-32b": [
        {
            "provider_model": "vllm/aya-32b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "CohereLabs/aya-vision-32b",
        }
    ],
    "mistral-small-3.2-24b": [
        {
            "provider_model": "vllm/mistral-small-3.2-24b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "unsloth/Mistral-Small-3.2-24B-Instruct-2506",  # The official Mistral model does not load due to a missing config file
        }
    ],
    "gemma-3-27b": [
        {
            "provider_model": "vllm/gemma-3-27b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "google/gemma-3-27b-it",
        }
    ],
    "mimo-vl-7b-rl": [
        {
            "provider_model": "vllm/mimo-vl-7b-rl",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
        }
    ],
    "nanonets-ocr-s": [
        {
            "provider_model": "vllm/nanonets-ocr-s",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "nanonets/Nanonets-OCR-s",
        }
    ],
    "skywork-r1v3-38b": [
        {
            "provider_model": "vllm/skywork-r1v3-38b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "Skywork/Skywork-R1V3-38B",
        }
    ],
    "numarkdown-8b": [
        {
            "provider_model": "vllm/numarkdown-8b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "api_base": get_settings().local_base_url,
            },
            "hf_repo": "numind/NuMarkdown-8B-Thinking",
        }
    ],
    "r-4b": [
        {
            "provider_model": "vllm/r-4b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "YannQi/R-4B",
        }
    ],
    "nemotron-nano-vl-8b": [
        {
            "provider_model": "vllm/nemotron-nano-vl-8b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
        }
    ],
    "internvl3.5-30b": [
        {
            "provider_model": "vllm/internvl3.5-30b",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_REASONING_MODELS,
            "static_params": {
                "api_base": get_settings().local_base_url,
            },
            "hf_repo": "OpenGVLab/InternVL3_5-30B-A3B",
        }
    ],
    "rolmocr": [
        {
            "provider_model": "vllm/rolmocr",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "reducto/RolmOCR",
        }
    ],
    "olmocr": [
        {
            "provider_model": "vllm/olmocr",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "allenai/olmOCR-7B-0825",
        }
    ],
    "minicpm-v-4.5": [
        {
            "provider_model": "vllm/minicpm-v-4.5",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url},
            "hf_repo": "openbmb/MiniCPM-V-4_5",
        }
    ],
    "churro": [
        {
            "provider_model": "vllm/churro",
            "max_completion_tokens": COMPLETION_TOKENS_FOR_STANDARD_MODELS,
            "static_params": {"api_base": get_settings().local_base_url, "temperature": 0.6},
            "hf_repo": CHURRO_MODEL_ID,
        }
    ],
}

EMBEDDING_MODEL_MAP: dict[str, str] = {
    "gemini-embedding": "vertex_ai/gemini-embedding-001",
}


def _validate_model_registry() -> None:
    """Validate MODEL_MAP structure at import time.

    Checks:
      - Keys are non-empty strings.
      - Each value is a non-empty list.
      - Required fields in each candidate: provider_model (str), max_completion_tokens (int|None).
      - static_params, if present, is a dict.
      - hf_repo, if present, is a non-empty string.
    Raises ValueError on first encountered issue to fail fast during startup/tests.
    """
    for logical_key, candidates in MODEL_MAP.items():
        if not isinstance(logical_key, str) or not logical_key.strip():
            raise ValueError(f"Model registry key must be non-empty string: {logical_key!r}")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"Model registry entry for '{logical_key}' must be a non-empty list")
        for idx, cand in enumerate(candidates):
            if "provider_model" not in cand or not isinstance(cand["provider_model"], str):
                raise ValueError(
                    f"Model '{logical_key}' candidate {idx} missing valid 'provider_model'"
                )
            if "max_completion_tokens" not in cand or not (
                isinstance(cand["max_completion_tokens"], (int, type(None)))
            ):
                raise ValueError(
                    f"Model '{logical_key}' candidate {idx} has invalid 'max_completion_tokens'"
                )
            if "static_params" in cand and not isinstance(cand["static_params"], dict):
                raise ValueError(
                    f"Model '{logical_key}' candidate {idx} 'static_params' must be a dict if present"
                )
            if "hf_repo" in cand and not (
                isinstance(cand["hf_repo"], str) and cand["hf_repo"].strip()
            ):
                raise ValueError(
                    f"Model '{logical_key}' candidate {idx} 'hf_repo' must be non-empty string if present"
                )


# Execute validation immediately so misconfigurations surface early.
_validate_model_registry()
