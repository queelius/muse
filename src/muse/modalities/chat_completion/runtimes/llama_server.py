"""LlamaServerModel: GGUF chat & completion runtime via supervised ``llama-server``.

Delegates inference to an isolated ``llama-server`` subprocess over local HTTP,
enabling multi-slot continuous batching (``--parallel`` + ``--cont-batching``),
KV prompt caching (``cache_prompt``), raw GBNF ``grammar``, and zero per-venv
``llama-cpp-python`` / CUDA wheel dependencies.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import httpx

from muse.core.llama_server import (
    LlamaServerProcess,
    ensure_llama_server_binary,
    probe_llama_server,
)
from muse.core.placement import native_placement
from muse.core.runtime_helpers import GGUF_DEFAULT_CONTEXT_LENGTH
from muse.modalities._common.gguf import gguf_path as _gguf_path, is_cpu_only
from muse.modalities.chat_completion.protocol import (
    ChatChoice,
    ChatChunk,
    ChatResult,
)

logger = logging.getLogger(__name__)

_CHAT_FORWARDED_KWARGS = frozenset({
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "tools",
    "tool_choice",
    "response_format",
    "logprobs",
    "top_logprobs",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
    "grammar",
    "logit_bias",
})

_COMPLETION_FORWARDED_KWARGS = frozenset({
    "temperature",
    "top_p",
    "seed",
    "grammar",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
    "logit_bias",
})


class LlamaServerModel:
    """GGUF chat + text-completion backend backed by a supervised ``llama-server``."""

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str = "",
        local_dir: str | None = None,
        gguf_file: str,
        context_length: int = GGUF_DEFAULT_CONTEXT_LENGTH,
        chat_format: str | None = None,
        chat_template: str | None = None,
        n_gpu_layers: int = -1,
        device: str = "auto",
        parallel: int = 4,
        cont_batching: bool = True,
        cache_prompt: bool = True,
        no_jinja: bool = False,
        n_threads: int | None = None,
        n_threads_batch: int | None = None,
        n_batch: int | None = None,
        flash_attn: bool | None = None,
        use_mlock: bool | None = None,
        type_k: int | str | None = None,
        type_v: int | str | None = None,
        supports_tools: bool | None = None,
        binary_path: str | None = None,
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.supports_tools = supports_tools
        self.cache_prompt = bool(cache_prompt)
        self.parallel = max(1, int(parallel))
        # ModalityRegistry.register preserves an existing _inference_lock,
        # allowing up to `parallel` concurrent slots in llama-server.
        self._inference_lock = threading.Semaphore(self.parallel)

        effective_chat_format = chat_format if chat_format is not None else chat_template
        model_path = _gguf_path(local_dir, gguf_file)
        binary = ensure_llama_server_binary(
            binary_path=Path(binary_path) if binary_path else None,
        )

        if is_cpu_only(device, n_gpu_layers):
            placement = native_placement(device, n_gpu_layers, gpu_offload=False)
        else:
            gpu_offload, gpu_backend = probe_llama_server(binary)
            placement = native_placement(
                device, n_gpu_layers, gpu_offload=gpu_offload, gpu_backend=gpu_backend,
            )

        self.device = placement.device
        effective_ngl = placement.n_gpu_layers

        logger.info(
            "starting llama-server for %s (%s, ctx=%d, ngl=%d, parallel=%d, device=%s)",
            model_id,
            model_path,
            context_length,
            effective_ngl,
            self.parallel,
            self.device,
        )

        self._server = LlamaServerProcess(
            binary=binary,
            model_path=model_path,
            context_length=context_length,
            n_gpu_layers=effective_ngl,
            parallel=self.parallel,
            cont_batching=bool(cont_batching),
            no_jinja=bool(no_jinja),
            chat_format=effective_chat_format,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_batch=n_batch,
            flash_attn=flash_attn,
            use_mlock=use_mlock,
            type_k=type_k,
            type_v=type_v,
        )
        self._server.start()
        self._client = httpx.Client(
            base_url=self._server.base_url,
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
        )

    def _chat_payload(
        self, messages: list[dict], *, stream: bool, kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        cache_prompt = kwargs.get("cache_prompt")
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "cache_prompt": self.cache_prompt if cache_prompt is None else bool(cache_prompt),
        }
        for key, value in kwargs.items():
            if key in _CHAT_FORWARDED_KWARGS and value is not None:
                payload[key] = value
        return payload

    def _completion_payload(
        self, prompt: str, *, stream: bool, kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        cache_prompt = kwargs.get("cache_prompt")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "stream": stream,
            "cache_prompt": self.cache_prompt if cache_prompt is None else bool(cache_prompt),
        }
        n_predict = kwargs.get("n_predict")
        if n_predict is None:
            n_predict = kwargs.get("max_tokens")
        if n_predict is not None:
            payload["n_predict"] = int(n_predict)

        n_probs = kwargs.get("n_probs")
        if n_probs is None:
            n_probs = kwargs.get("logprobs")
        if n_probs is not None:
            payload["n_probs"] = int(n_probs)

        stop = kwargs.get("stop")
        if stop is not None:
            payload["stop"] = [stop] if isinstance(stop, str) else list(stop)

        for key, value in kwargs.items():
            if key in _COMPLETION_FORWARDED_KWARGS and value is not None:
                payload[key] = value
        return payload

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        payload = self._chat_payload(messages, stream=False, kwargs=kwargs)
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return _dict_to_chat_result(resp.json(), model_id=self.model_id)

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[ChatChunk]:
        payload = self._chat_payload(messages, stream=True, kwargs=kwargs)
        with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for data in _iter_sse_json(resp):
                yield _dict_to_chat_chunk(data, model_id=self.model_id)

    def complete(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        payload = self._completion_payload(prompt, stream=False, kwargs=kwargs)
        resp = self._client.post("/completion", json=payload)
        resp.raise_for_status()
        return _normalize_completion_result(resp.json(), model_id=self.model_id)

    def complete_stream(self, prompt: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        payload = self._completion_payload(prompt, stream=True, kwargs=kwargs)
        with self._client.stream("POST", "/completion", json=payload) as resp:
            resp.raise_for_status()
            for data in _iter_sse_json(resp):
                yield _normalize_completion_chunk(data, model_id=self.model_id)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            logger.debug("error closing httpx client for %s", self.model_id, exc_info=True)
        self._server.stop()


def _iter_sse_json(resp: httpx.Response) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON payloads from an SSE stream, stopping at ``[DONE]``."""
    for raw_line in resp.iter_lines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body:
            continue
        if body == "[DONE]":
            break
        yield json.loads(body)


def _dict_to_chat_result(resp: dict[str, Any], *, model_id: str) -> ChatResult:
    return ChatResult(
        id=resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model_id=model_id,
        created=resp.get("created") or int(time.time()),
        choices=[
            ChatChoice(
                index=c.get("index", 0),
                message=c["message"],
                finish_reason=c.get("finish_reason"),
            )
            for c in resp.get("choices", [])
        ],
        usage=resp.get("usage")
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def _dict_to_chat_chunk(chunk: dict[str, Any], *, model_id: str) -> ChatChunk:
    choices = chunk.get("choices") or []
    c = choices[0] if choices else {"index": 0, "delta": {}, "finish_reason": None}
    return ChatChunk(
        id=chunk.get("id") or "chatcmpl-stream",
        model_id=model_id,
        created=chunk.get("created") or int(time.time()),
        choice_index=c.get("index", 0),
        delta=c.get("delta") or {},
        finish_reason=c.get("finish_reason"),
    )


def _completion_finish_reason(data: dict[str, Any], *, stopped: bool) -> str | None:
    if not stopped:
        return None
    stop_type = data.get("stop_type")
    if data.get("stopped_limit") or stop_type == "limit":
        return "length"
    return "stop"


def _normalize_completion_result(
    data: dict[str, Any], *, model_id: str,
) -> dict[str, Any]:
    stopped = bool(data.get("stop", True))
    return {
        "content": data.get("content", ""),
        "stop": stopped,
        "stop_type": data.get("stop_type"),
        "stopped_eos": bool(data.get("stopped_eos", False)),
        "stopped_limit": bool(data.get("stopped_limit", False)),
        "stopped_word": bool(data.get("stopped_word", False)),
        "finish_reason": _completion_finish_reason(data, stopped=stopped),
        "tokens_evaluated": int(data.get("tokens_evaluated") or 0),
        "tokens_predicted": int(data.get("tokens_predicted") or 0),
        "timings": data.get("timings") or {},
        "model_id": model_id,
    }


def _normalize_completion_chunk(
    data: dict[str, Any], *, model_id: str,
) -> dict[str, Any]:
    stopped = bool(data.get("stop", False))
    return {
        "content": data.get("content", ""),
        "stop": stopped,
        "finish_reason": _completion_finish_reason(data, stopped=stopped),
        "tokens_predicted": data.get("tokens_predicted"),
        "timings": data.get("timings"),
        "model_id": model_id,
    }
