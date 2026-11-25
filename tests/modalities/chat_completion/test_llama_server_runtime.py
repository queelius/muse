"""Tests for LlamaServerModel runtime and chat_completion HF plugin defaults."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from muse.core.registry import ModalityRegistry
from muse.modalities.chat_completion.protocol import ChatChunk, ChatResult
from muse.modalities.chat_completion.runtimes import llama_server as rt


def _make_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
    *,
    gpu_probe: tuple[bool, str | None] = (True, "cuda"),
    **kwargs,
):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    fake_bin = tmp_path / "llama-server"
    fake_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_bin.chmod(0o755)

    captured_server_kwargs: dict = {}
    stopped: list[bool] = []

    class _FakeServer:
        def __init__(self, **skw):
            captured_server_kwargs.update(skw)
            self.base_url = "http://127.0.0.1:9999"

        def start(self):
            return None

        def stop(self, timeout=5.0):
            stopped.append(True)

    monkeypatch.setattr(rt, "ensure_llama_server_binary", lambda **_: fake_bin)
    monkeypatch.setattr(rt, "probe_llama_server", lambda _bin: gpu_probe)
    monkeypatch.setattr(rt, "LlamaServerProcess", _FakeServer)

    real_client = httpx.Client
    monkeypatch.setattr(
        rt.httpx,
        "Client",
        lambda **ckw: real_client(base_url=ckw["base_url"], transport=transport),
    )

    model = rt.LlamaServerModel(
        model_id=kwargs.pop("model_id", "npc-qwen"),
        local_dir=str(tmp_path),
        gguf_file="model.gguf",
        **kwargs,
    )
    return model, captured_server_kwargs, stopped


def test_constructor_sets_semaphore_and_server_flags(tmp_path, monkeypatch):
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
    model, skw, stopped = _make_runtime(
        tmp_path,
        monkeypatch,
        transport,
        parallel=6,
        cont_batching=True,
        no_jinja=True,
        chat_format="chatml",
        n_gpu_layers=-1,
        device="auto",
        context_length=4096,
    )
    assert isinstance(model._inference_lock, threading.Semaphore)
    assert model.device == "cuda"
    assert skw["parallel"] == 6
    assert skw["cont_batching"] is True
    assert skw["no_jinja"] is True
    assert skw["context_length"] == 4096
    assert skw["n_gpu_layers"] == -1

    # ModalityRegistry preserves our Semaphore rather than replacing it with a Lock
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    assert isinstance(model._inference_lock, threading.Semaphore)

    model.close()
    assert stopped == [True]


def test_chat_forwards_grammar_and_cache_prompt_and_overrides_model_id(tmp_path, monkeypatch):
    seen_payloads: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 1700000000,
                "model": "/tmp/internal/model.gguf",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        )

    model, _, _ = _make_runtime(tmp_path, monkeypatch, httpx.MockTransport(_handler))
    res = model.chat(
        [{"role": "user", "content": "hi"}],
        grammar='root ::= "{" "\\"ok\\": true}"',
        temperature=0.2,
        max_tokens=32,
        ignored_kwarg="drop-me",
    )
    assert isinstance(res, ChatResult)
    assert res.model_id == "npc-qwen"
    assert res.choices[0].message["content"] == '{"ok": true}'
    assert res.usage["total_tokens"] == 16

    payload = seen_payloads[0]
    assert payload["stream"] is False
    assert payload["cache_prompt"] is True
    assert payload["grammar"] == 'root ::= "{" "\\"ok\\": true}"'
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 32
    assert "ignored_kwarg" not in payload
    model.close()


def test_chat_stream_parses_sse_and_overrides_model_id(tmp_path, monkeypatch):
    sse_body = (
        'data: {"id":"c1","created":1,"model":"/bad/path.gguf","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}\n\n'
        'data: {"id":"c1","created":1,"model":"/bad/path.gguf","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    model, _, _ = _make_runtime(tmp_path, monkeypatch, httpx.MockTransport(_handler))
    chunks = list(model.chat_stream([{"role": "user", "content": "hi"}]))
    assert len(chunks) == 2
    assert all(isinstance(c, ChatChunk) for c in chunks)
    assert all(c.model_id == "npc-qwen" for c in chunks)
    assert "".join(c.delta.get("content", "") for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    model.close()


def test_complete_posts_native_endpoint_and_normalizes_response(tmp_path, monkeypatch):
    seen_payloads: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/completion"
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "content": " world",
                "stop": True,
                "stop_type": "limit",
                "stopped_eos": False,
                "stopped_limit": True,
                "tokens_evaluated": 5,
                "tokens_predicted": 2,
                "timings": {"predicted_per_second": 120.5},
            },
        )

    model, _, _ = _make_runtime(tmp_path, monkeypatch, httpx.MockTransport(_handler))
    res = model.complete(
        "hello",
        max_tokens=2,
        stop="\n",
        grammar='root ::= " world"',
        cache_prompt=False,
        logprobs=3,
    )
    payload = seen_payloads[0]
    assert payload["prompt"] == "hello"
    assert payload["n_predict"] == 2
    assert payload["stop"] == ["\n"]
    assert payload["grammar"] == 'root ::= " world"'
    assert payload["cache_prompt"] is False
    assert payload["n_probs"] == 3

    assert res["content"] == " world"
    assert res["finish_reason"] == "length"
    assert res["stop_type"] == "limit"
    assert res["tokens_evaluated"] == 5
    assert res["tokens_predicted"] == 2
    assert res["timings"] == {"predicted_per_second": 120.5}
    assert res["model_id"] == "npc-qwen"
    model.close()


def test_complete_stream_yields_normalized_chunks(tmp_path, monkeypatch):
    sse_body = (
        'data: {"content":"foo","stop":false}\n\n'
        'data: {"content":"bar","stop":true,"stopped_eos":true,"tokens_predicted":2,"timings":{"predicted_ms":10}}\n\n'
    )

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/completion"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    model, _, _ = _make_runtime(tmp_path, monkeypatch, httpx.MockTransport(_handler))
    chunks = list(model.complete_stream("say foobar", n_predict=16))
    assert len(chunks) == 2
    assert chunks[0]["content"] == "foo"
    assert chunks[0]["stop"] is False
    assert chunks[0]["finish_reason"] is None
    assert chunks[1]["content"] == "bar"
    assert chunks[1]["stop"] is True
    assert chunks[1]["finish_reason"] == "stop"
    assert chunks[1]["tokens_predicted"] == 2
    assert chunks[1]["model_id"] == "npc-qwen"
    model.close()


def test_hf_plugin_resolves_gguf_to_llama_server():
    from muse.modalities.chat_completion.hf import HF_PLUGIN

    info = MagicMock()
    info.siblings = [MagicMock(rfilename="model-q4_k_m.gguf")]
    info.tags = []
    info.card_data = MagicMock(license="apache-2.0")

    with patch("muse.modalities.chat_completion.hf._try_sniff_tools_from_repo", return_value=False), \
         patch("muse.modalities.chat_completion.hf._try_sniff_context_length_from_repo", return_value=None), \
         patch("muse.modalities.chat_completion.hf.lookup_chat_format", return_value={}):
        resolved = HF_PLUGIN["resolve"]("org/Model-GGUF", "q4_k_m", info)

    assert resolved.backend_path == (
        "muse.modalities.chat_completion.runtimes.llama_server:LlamaServerModel"
    )
    assert resolved.manifest["pip_extras"] == []
    caps = resolved.manifest["capabilities"]
    assert caps["parallel"] == 4
    assert caps["cont_batching"] is True
    assert caps["cache_prompt"] is True
