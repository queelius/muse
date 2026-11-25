"""Chat completion modality: text-to-text LLM serving.

Wire contract: POST /v1/chat/completions with OpenAI-shape body
(messages, tools?, tool_choice?, response_format?, stream?, temperature?,
max_tokens?, stop?, seed?, logprobs?, top_logprobs?) returns OpenAI
ChatCompletion or, when stream=True, SSE-encoded ChatCompletionChunk
events.

Models declaring `modality = "chat/completion"` in their MANIFEST and
satisfying the ChatModel protocol plug into this modality.
"""
from muse.modalities.chat_completion.client import ChatClient, ChatStreamError
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatMessage,
    ChatModel,
    ChatResult,
    CompletionChoice,
    CompletionChunk,
    CompletionResult,
)
from muse.modalities.chat_completion.routes import build_router

MODALITY = "chat/completion"

# Native llama-server clients omit `model` on POST /completion; the gateway
# resolves the first enabled chat/completion model in the catalog.
MODEL_OPTIONAL_PATHS = ("/completion",)

DISCOVERY = {
    "summary": (
        "Chat and vision-language completion. Answers messages with a "
        "language model, with optional tool calls, structured output, "
        "streaming and image inputs for vision models, plus raw text "
        "completions."
    ),
    "use_for": [
        "writing, summarizing, reasoning and drafting text",
        "describing or answering questions about images (vision models)",
        "tool-using agents through OpenAI-style tools",
        "raw prompt completions with optional GBNF grammar constraints",
    ],
    "not_for": [
        "bulk embedding (use embedding/text)",
        (
            "typed Choice or Score decisions with probabilities (use "
            "decision/evaluation)"
        ),
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "openai:chat.completions",
            "mcp_tool": "muse_chat",
            "file_fields": [],
            "media_fields": [],
            "example": {
                "model": "<model>",
                "messages": [
                    {"role": "user", "content": "Say hello."},
                ],
                "stream": False,
            },
        },
        {
            "method": "POST",
            "path": "/v1/completions",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "openai:completions",
            "mcp_tool": "muse_complete",
            "file_fields": [],
            "media_fields": [],
            "example": {
                "model": "<model>",
                "prompt": "Once upon a time,",
                "max_tokens": 32,
                "stream": False,
            },
        },
        {
            "method": "POST",
            "path": "/completion",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "llama_server:completion",
            "mcp_tool": None,
            "file_fields": [],
            "media_fields": [],
            "example": {
                "prompt": "Once upon a time,",
                "n_predict": 32,
                "stream": False,
            },
        },
    ],
}

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
    "MODEL_OPTIONAL_PATHS",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "ChatClient",
    "ChatStreamError",
    "ChatChoice",
    "ChatChunk",
    "ChatMessage",
    "ChatModel",
    "ChatResult",
    "CompletionChoice",
    "CompletionChunk",
    "CompletionResult",
]
