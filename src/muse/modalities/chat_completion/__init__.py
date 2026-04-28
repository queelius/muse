"""Chat completion modality: text-to-text LLM serving.

Wire contract: POST /v1/chat/completions with OpenAI-shape body
(messages, tools?, tool_choice?, response_format?, stream?, temperature?,
max_tokens?, stop?, seed?, logprobs?, top_logprobs?) returns OpenAI
ChatCompletion or, when stream=True, SSE-encoded ChatCompletionChunk
events.

Models declaring `modality = "chat/completion"` in their MANIFEST and
satisfying the ChatModel protocol plug into this modality.
"""
from muse.modalities.chat_completion.client import ChatClient
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatMessage,
    ChatModel,
    ChatResult,
)
from muse.modalities.chat_completion.routes import build_router

MODALITY = "chat/completion"

# Per-modality probe defaults read by `muse models probe`. An 8-token
# completion is enough to walk the prefill + a few decode steps so KV
# cache + attention buffers register in the peak measurement, while
# staying fast on small models.
PROBE_DEFAULTS = {
    "shape": "8-token completion",
    "call": lambda m: m.chat(
        messages=[{"role": "user", "content": "probe"}],
        max_tokens=8,
    ),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ChatClient",
    "ChatChoice",
    "ChatChunk",
    "ChatMessage",
    "ChatModel",
    "ChatResult",
]
