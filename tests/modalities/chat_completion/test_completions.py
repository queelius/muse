"""Tests for POST /v1/completions, POST /completion, completion codecs, and ChatClient."""
from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.core.sse import iter_sse_events
from muse.modalities.chat_completion import (
    DISCOVERY,
    MODEL_OPTIONAL_PATHS,
    ChatClient,
    CompletionChoice,
    CompletionChunk,
    CompletionResult,
)
from muse.modalities.chat_completion.codec import (
    completion_chunk_to_llama_sse_data,
    completion_chunk_to_sse_data,
    completion_result_to_llama_dict,
    completion_result_to_openai_dict,
)
from muse.modalities.chat_completion.routes import build_router


class _FakeResultCompletionModel:
    """Backend returning CompletionResult / CompletionChunk dataclasses."""
    model_id = "fake-cmpl"

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_kwargs: dict = {}

    def chat(self, messages, **kwargs):
        raise NotImplementedError

    def chat_stream(self, messages, **kwargs):
        return iter([])

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        return CompletionResult(
            id="cmpl-fake-1",
            model_id=self.model_id,
            created=1_700_000_000,
            choices=[CompletionChoice(
                index=0,
                text=" world",
                finish_reason="stop",
            )],
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            metadata={"timings": {"predicted_per_second": 42.5}},
        )

    def complete_stream(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        yield CompletionChunk(
            id="cmpl-fake-2",
            model_id=self.model_id,
            created=1_700_000_001,
            choice_index=0,
            text=" wor",
            finish_reason=None,
        )
        yield CompletionChunk(
            id="cmpl-fake-2",
            model_id=self.model_id,
            created=1_700_000_001,
            choice_index=0,
            text="ld",
            finish_reason="stop",
            metadata={"tokens_predicted": 2, "timings": {"predicted_n": 2}},
        )


class _FakeDictCompletionModel:
    """Backend returning native llama-server dicts (like LlamaServerModel)."""
    model_id = "fake-llama"

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_kwargs: dict = {}

    def chat(self, messages, **kwargs):
        raise NotImplementedError

    def chat_stream(self, messages, **kwargs):
        return iter([])

    def complete(self, prompt: str, **kwargs) -> dict:
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        return {
            "content": "hello back",
            "stopped_eos": False,
            "stopped_limit": True,
            "tokens_evaluated": 5,
            "tokens_predicted": 2,
            "timings": {"prompt_n": 5, "predicted_n": 2},
        }

    def complete_stream(self, prompt: str, **kwargs):
        self.last_prompt = prompt
        self.last_kwargs = dict(kwargs)
        yield {"content": "hello", "stop": False, "tokens_predicted": 1}
        yield {
            "content": " back",
            "stop": True,
            "stopped_eos": True,
            "tokens_predicted": 2,
            "tokens_evaluated": 5,
            "timings": {"predicted_n": 2},
        }


class _ChatOnlyModel:
    """Model without complete() / complete_stream() (e.g. VLM)."""
    model_id = "vlm-only"

    def chat(self, messages, **kwargs):
        raise NotImplementedError

    def chat_stream(self, messages, **kwargs):
        return iter([])


def _make_client(model) -> TestClient:
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    return TestClient(app)


# --- Discovery & exports ----------------------------------------------------

def test_model_optional_paths_and_discovery():
    assert MODEL_OPTIONAL_PATHS == ("/completion",)
    routes = {r["path"]: r for r in DISCOVERY["routes"]}
    assert "/v1/completions" in routes
    assert routes["/v1/completions"]["mcp_tool"] == "muse_complete"
    assert routes["/v1/completions"]["compat"] == "openai:completions"
    assert "/completion" in routes
    assert routes["/completion"]["mcp_tool"] is None
    assert routes["/completion"]["compat"] == "llama_server:completion"


# --- Codec unit tests -------------------------------------------------------

def test_completion_result_to_openai_dict():
    res = CompletionResult(
        id="cmpl-1",
        model_id="m1",
        created=123,
        choices=[CompletionChoice(index=0, text="hi", finish_reason="stop", logprobs={"tokens": ["hi"]})],
        usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    )
    out = completion_result_to_openai_dict(res, system_fingerprint="fp_abc")
    assert out == {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 123,
        "model": "m1",
        "choices": [{
            "text": "hi",
            "index": 0,
            "logprobs": {"tokens": ["hi"]},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        "system_fingerprint": "fp_abc",
    }


def test_completion_chunk_to_sse_data():
    chunk = CompletionChunk(
        id="cmpl-2", model_id="m1", created=124, choice_index=0, text="tok", finish_reason=None,
    )
    parsed = json.loads(completion_chunk_to_sse_data(chunk))
    assert parsed["object"] == "text_completion"
    assert parsed["model"] == "m1"
    assert parsed["choices"] == [{
        "text": "tok",
        "index": 0,
        "logprobs": None,
        "finish_reason": None,
    }]


def test_completion_result_to_llama_dict():
    res = CompletionResult(
        id="cmpl-3",
        model_id="m1",
        created=125,
        choices=[CompletionChoice(index=0, text="done", finish_reason="length")],
        usage={"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26},
        metadata={"timings": {"predicted_n": 16}},
    )
    out = completion_result_to_llama_dict(res)
    assert out["content"] == "done"
    assert out["stop"] is True
    assert out["stopped_eos"] is False
    assert out["stopped_limit"] is True
    assert out["tokens_evaluated"] == 10
    assert out["tokens_predicted"] == 16
    assert out["timings"] == {"predicted_n": 16}
    assert out["model"] == "m1"


def test_completion_chunk_to_llama_sse_data():
    mid = CompletionChunk(
        id="c", model_id="m1", created=1, choice_index=0, text="a", finish_reason=None,
    )
    mid_parsed = json.loads(completion_chunk_to_llama_sse_data(mid))
    assert mid_parsed == {"content": "a", "stop": False}

    end = CompletionChunk(
        id="c", model_id="m1", created=1, choice_index=0, text="", finish_reason="stop",
        metadata={"tokens_predicted": 5, "timings": {"predicted_n": 5}},
    )
    end_parsed = json.loads(completion_chunk_to_llama_sse_data(end))
    assert end_parsed["stop"] is True
    assert end_parsed["stopped_eos"] is True
    assert end_parsed["stopped_limit"] is False
    assert end_parsed["tokens_predicted"] == 5
    assert end_parsed["model"] == "m1"


# --- POST /v1/completions ---------------------------------------------------

def test_v1_completions_non_streaming_dataclass_backend():
    model = _FakeResultCompletionModel()
    client = _make_client(model)
    gbnf = 'root ::= " world"'
    r = client.post(
        "/v1/completions",
        json={
            "model": "fake-cmpl",
            "prompt": "hello",
            "max_tokens": 16,
            "temperature": 0.1,
            "grammar": gbnf,
            "cache_prompt": True,
            "stop": ["\n"],
            "seed": 42,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "fake-cmpl"
    assert body["choices"][0]["text"] == " world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 4
    assert model.last_prompt == "hello"
    assert model.last_kwargs == {
        "max_tokens": 16,
        "temperature": 0.1,
        "grammar": gbnf,
        "cache_prompt": True,
        "stop": ["\n"],
        "seed": 42,
    }


def test_v1_completions_non_streaming_dict_backend():
    model = _FakeDictCompletionModel()
    client = _make_client(model)
    r = client.post(
        "/v1/completions",
        json={"model": "fake-llama", "prompt": "say hi", "n_predict": 8},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "fake-llama"
    assert body["choices"][0]["text"] == "hello back"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }
    assert model.last_kwargs["max_tokens"] == 8


def test_v1_completions_streaming_dataclass_and_dict_backends():
    for model_cls in (_FakeResultCompletionModel, _FakeDictCompletionModel):
        client = _make_client(model_cls())
        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "hi", "stream": True},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = r.read().decode()
        payloads = [
            data for event_type, data in iter_sse_events(body.splitlines())
            if event_type is None
        ]
        assert payloads[-1] == "[DONE]"
        parsed = [json.loads(p) for p in payloads[:-1]]
        assert len(parsed) == 2
        assert all(p["object"] == "text_completion" for p in parsed)
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


def test_v1_completions_prompt_list_handling():
    client = _make_client(_FakeResultCompletionModel())
    ok = client.post("/v1/completions", json={"prompt": ["single"]})
    assert ok.status_code == 200

    empty = client.post("/v1/completions", json={"prompt": []})
    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "invalid_prompt"

    batch = client.post("/v1/completions", json={"prompt": ["a", "b"]})
    assert batch.status_code == 400
    assert batch.json()["error"]["code"] == "invalid_prompt"


def test_v1_completions_unsupported_backend_returns_400():
    client = _make_client(_ChatOnlyModel())
    r = client.post("/v1/completions", json={"model": "vlm-only", "prompt": "hi"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "completions_not_supported"

    r_stream = client.post(
        "/v1/completions",
        json={"model": "vlm-only", "prompt": "hi", "stream": True},
    )
    assert r_stream.status_code == 400
    assert r_stream.json()["error"]["code"] == "completions_not_supported"


def test_v1_completions_unknown_model_returns_404():
    client = _make_client(_FakeResultCompletionModel())
    r = client.post("/v1/completions", json={"model": "missing", "prompt": "hi"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize("payload", [
    {"prompt": "hi", "max_tokens": 0},
    {"prompt": "hi", "n_predict": 0},
    {"prompt": "hi", "temperature": 2.5},
    {"prompt": "hi", "extra_body": {"grammar": "root ::= 'a'"}},
    {"prompt": "hi", "extra_body": {"max_new_tokens": 999999}},
])
def test_v1_completions_validation_422(payload):
    client = _make_client(_FakeResultCompletionModel())
    r = client.post("/v1/completions", json=payload)
    assert r.status_code == 422


def test_v1_completions_lock_and_stream_error():
    class _LockAndErrorModel:
        model_id = "lock-cmpl"

        def __init__(self) -> None:
            self._inference_lock = threading.Lock()
            self.lock_was_held = False

        def chat(self, messages, **kwargs):
            raise NotImplementedError

        def chat_stream(self, messages, **kwargs):
            return iter([])

        def complete(self, prompt: str, **kwargs):
            self.lock_was_held = self._inference_lock.locked()
            return {"content": "ok", "stopped_eos": True}

        def complete_stream(self, prompt: str, **kwargs):
            assert self._inference_lock.locked()
            yield {"content": "part", "stop": False}
            raise RuntimeError("boom")

    model = _LockAndErrorModel()
    client = _make_client(model)
    r = client.post("/v1/completions", json={"model": "lock-cmpl", "prompt": "hi"})
    assert r.status_code == 200
    assert model.lock_was_held is True

    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "lock-cmpl", "prompt": "hi", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode()

    events = list(iter_sse_events(body.splitlines()))
    assert events[-1] == (None, "[DONE]")
    error_events = [e for e in events if e[0] == "error"]
    assert len(error_events) == 1
    assert json.loads(error_events[0][1])["error"]["code"] == "internal_error"

    deadline = time.time() + 2.0
    while model._inference_lock.locked() and time.time() < deadline:
        time.sleep(0.01)
    assert not model._inference_lock.locked()


# --- POST /completion (native llama-server alias) ---------------------------

def test_llama_completion_non_streaming_omits_model():
    model = _FakeDictCompletionModel()
    client = _make_client(model)
    r = client.post(
        "/completion",
        json={
            "prompt": "raw prompt",
            "n_predict": 32,
            "grammar": 'root ::= "hello back"',
            "cache_prompt": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "hello back"
    assert body["stop"] is True
    assert body["stopped_eos"] is False
    assert body["stopped_limit"] is True
    assert body["tokens_evaluated"] == 5
    assert body["tokens_predicted"] == 2
    assert body["timings"] == {"prompt_n": 5, "predicted_n": 2}
    assert body["model"] == "fake-llama"
    assert model.last_kwargs["max_tokens"] == 32
    assert model.last_kwargs["grammar"] == 'root ::= "hello back"'
    assert model.last_kwargs["cache_prompt"] is True


def test_llama_completion_streaming_shape():
    model = _FakeDictCompletionModel()
    client = _make_client(model)
    with client.stream(
        "POST",
        "/completion",
        json={"prompt": "raw prompt", "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.read().decode()

    payloads = [
        data for event_type, data in iter_sse_events(body.splitlines())
        if event_type is None
    ]
    assert "[DONE]" not in payloads
    parsed = [json.loads(p) for p in payloads]
    assert len(parsed) == 2
    assert parsed[0]["content"] == "hello"
    assert parsed[0]["stop"] is False
    assert parsed[1]["content"] == " back"
    assert parsed[1]["stop"] is True
    assert parsed[1]["stopped_eos"] is True
    assert parsed[1]["model"] == "fake-llama"


# --- ChatClient.complete / complete_stream ----------------------------------

def test_chat_client_complete_and_stream(monkeypatch):
    model = _FakeResultCompletionModel()
    reg = ModalityRegistry()
    reg.register("chat/completion", model)
    app = create_app(registry=reg, routers={"chat/completion": build_router(reg)})
    sync_client = TestClient(app)

    def _fake_post(url, *, json=None, timeout=None):
        assert url == "http://testserver/v1/completions"
        return sync_client.post("/v1/completions", json=json)

    def _fake_stream(method, url, *, json=None, timeout=None):
        assert method == "POST"
        assert url == "http://testserver/v1/completions"
        return sync_client.stream("POST", "/v1/completions", json=json)

    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.setattr(httpx, "stream", _fake_stream)

    client = ChatClient(base_url="http://testserver")
    res = client.complete(model="fake-cmpl", prompt="hello", max_tokens=10)
    assert res["object"] == "text_completion"
    assert res["choices"][0]["text"] == " world"

    chunks = list(client.complete_stream(model="fake-cmpl", prompt="hello"))
    assert len(chunks) == 2
    assert "".join(c["choices"][0]["text"] for c in chunks) == " world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
