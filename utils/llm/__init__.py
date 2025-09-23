"""Top-level public API for LLM utilities.

This package exposes a clean, stable surface for:

- Chat/completions: run_llm_simple_async
- Embeddings: create_embedding_async
- Message preparation and image encoding: prepare_messages, encode_image
- Output parsing helpers: extract_tag_from_llm_output, string_to_list_of_floats, string_to_list_of_ints
- Cost tracking: log_total_llm_cost, get_llm_total_cost

Import from this module to avoid relying on internal structure:

    from utils.llm import run_llm_simple_async, create_embedding_async

"""

from .core import run_llm_async
from .cost import get_llm_total_cost, log_total_llm_cost
from .embeddings import create_embedding_async
from .messages import encode_image, prepare_messages
from .shutdown import shutdown_llm_clients
from .types import ImageDetail, MessageContent, Messages, ModelInfo
from .utils import (
    extract_tag_from_llm_output,
    string_to_list_of_floats,
    string_to_list_of_ints,
)


__all__ = [
    # core
    "run_llm_async",
    "shutdown_llm_clients",
    # embeddings
    "create_embedding_async",
    # messages
    "prepare_messages",
    "encode_image",
    # helpers
    "extract_tag_from_llm_output",
    "string_to_list_of_floats",
    "string_to_list_of_ints",
    # cost
    "log_total_llm_cost",
    "get_llm_total_cost",
    # types
    "MessageContent",
    "Messages",
    "ImageDetail",
    "ModelInfo",
]
