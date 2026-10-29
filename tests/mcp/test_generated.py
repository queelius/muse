"""Tools generated from the server's /v1/capabilities + /openapi.json."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
import wave

import httpx
import jsonschema
import pytest

from muse.core.discovery import modality_discovery
from muse.mcp.generated import (
    OUTPUT_SCHEMA,
    _render,
    call_generated,
    describe_tool,
    generate_tools,
)


def _gen(caps_doc, openapi_doc, name):
    return next(g for g in generate_tools(caps_doc, openapi_doc) if g.name == name)


def _wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))
    return buf.getvalue()


def _png(w: int = 4, h: int = 3) -> bytes:
    PIL = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL.new("RGB", (w, h), (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def _structured(blocks):
    [s] = [b for b in blocks if b["type"] == "structured"]
    return s["data"]


HEADERS = {"x-muse-model": "kokoro-82m", "x-muse-model-revision": "b" * 40}


# ---- schema generation ----

def test_generates_one_tool_per_declared_route(caps_doc, openapi_doc):
    names = {g.name for g in generate_tools(caps_doc, openapi_doc)}
    assert {"muse_speak", "muse_align", "muse_transcribe",
            "muse_assess_audio_quality"} <= names
    declared = {r["mcp_tool"] for d in modality_discovery().values()
                for r in d["routes"]}
    assert names == declared


def test_existing_tool_names_survive(caps_doc, openapi_doc):
    names = {g.name for g in generate_tools(caps_doc, openapi_doc)}
    old = {"muse_chat", "muse_summarize", "muse_rerank", "muse_classify",
           "muse_embed_text", "muse_translate", "muse_generate_image",
           "muse_edit_image", "muse_vary_image", "muse_upscale_image",
           "muse_segment_image", "muse_vectorize_image",
           "muse_generate_animation", "muse_embed_image", "muse_speak",
           "muse_transcribe", "muse_generate_music", "muse_generate_sfx",
           "muse_embed_audio", "muse_generate_video", "muse_decide"}
    assert old <= names


def test_file_fields_become_path_and_url(caps_doc, openapi_doc):
    gen = _gen(caps_doc, openapi_doc, "muse_align")
    props = gen.tool.inputSchema["properties"]
    assert "file_path" in props and "file_url" in props
    assert "file_b64" not in props and "file" not in props
    assert "text" in gen.tool.inputSchema["required"]
    # "Exactly one" is enforced at call time, not with a top-level oneOf
    # (the Claude API rejects top-level oneOf/anyOf/allOf in tool schemas).
    assert "file_path" not in gen.tool.inputSchema["required"]
    assert "exactly one" in props["file_path"]["description"].lower()


def test_list_file_field_becomes_paths_and_urls(caps_doc, openapi_doc):
    props = _gen(caps_doc, openapi_doc, "muse_embed_image").tool.inputSchema["properties"]
    assert props["input_paths"]["type"] == "array"
    assert props["input_urls"]["type"] == "array"
    assert "input" not in props


def test_model_ids_are_described_not_enumerated(caps_doc, openapi_doc):
    # No enum: the SDK validates against the client's cached schema, so an
    # enum would reject a model enabled mid-session.
    schema = _gen(caps_doc, openapi_doc, "muse_speak").tool.inputSchema
    prop = schema["properties"]["model"]
    assert "enum" not in prop
    assert "kokoro-82m" in prop["description"] and "soprano-80m" in prop["description"]
    assert "model" in schema["required"]


def test_description_has_summary_compat_and_model_lines(caps_doc, openapi_doc):
    desc = _gen(caps_doc, openapi_doc, "muse_speak").tool.description
    summary = modality_discovery()["audio/speech"]["summary"]
    assert desc.startswith(summary)
    assert "openai:audio.speech" in desc
    assert "kokoro-82m — good — kokoro-82m does audio/speech well." in desc
    assert desc.index(summary) < desc.index("openai:") < desc.index("kokoro-82m —")


def test_chat_forces_stream_false(caps_doc, openapi_doc, fake_http, media_env):
    gen = _gen(caps_doc, openapi_doc, "muse_chat")
    assert "stream" not in gen.tool.inputSchema["properties"]
    client = fake_http(lambda r: httpx.Response(200, json={"choices": []}))
    call_generated(client, gen, {"model": "qwen3.5-4b-q4",
                                 "messages": [{"role": "user", "content": "hi"}]})
    sent = json.loads(fake_http.requests[-1].content)
    assert sent["stream"] is False


def test_preview_only_on_image_producing_tools(caps_doc, openapi_doc):
    tools = {g.name: g.tool for g in generate_tools(caps_doc, openapi_doc)}
    assert "preview" in tools["muse_generate_image"].inputSchema["properties"]
    assert "preview" in tools["muse_edit_image"].inputSchema["properties"]
    assert "preview" not in tools["muse_speak"].inputSchema["properties"]
    assert "preview" not in tools["muse_generate_video"].inputSchema["properties"]
    assert "preview" not in tools["muse_generate_3d"].inputSchema["properties"]
    for t in tools.values():
        assert "out_path" in t.inputSchema["properties"]


def test_input_schemas_are_self_contained(caps_doc, openapi_doc):
    for g in generate_tools(caps_doc, openapi_doc):
        schema = g.tool.inputSchema
        assert "$ref" not in json.dumps(schema), g.name
        assert not {"oneOf", "anyOf", "allOf"} & set(schema), g.name
        jsonschema.Draft202012Validator.check_schema(schema)
        assert g.tool.outputSchema == OUTPUT_SCHEMA


def test_route_without_request_body_falls_back_to_example(caps_doc, openapi_doc):
    gen = _gen(caps_doc, openapi_doc, "muse_translate")
    schema = gen.tool.inputSchema
    assert {"q", "source", "target", "model"} <= set(schema["properties"])
    assert schema["additionalProperties"] is True
    assert "example" in gen.tool.description.lower()
    decide = _gen(caps_doc, openapi_doc, "muse_decide").tool.inputSchema
    assert {"state", "questions"} <= set(decide["required"])
    # Untyped: a Score/Choice state may be a string or an object.
    assert "type" not in decide["properties"]["state"]


def test_get_route_sends_query_params(caps_doc, openapi_doc, fake_http, media_env):
    gen = _gen(caps_doc, openapi_doc, "muse_list_voices")
    assert "model" in gen.tool.inputSchema["properties"]
    client = fake_http(lambda r: httpx.Response(200, json={"voices": ["af"]}))
    blocks = call_generated(client, gen, {"model": "kokoro-82m"})
    req = fake_http.requests[-1]
    assert req.method == "GET" and req.url.params["model"] == "kokoro-82m"
    assert _structured(blocks)["response"] == {"voices": ["af"]}


def test_describe_tool_shape():
    t = describe_tool()
    assert t.name == "muse_describe"
    assert set(t.inputSchema["properties"]) == {"modality", "model"}
    assert "before" in t.description.lower()


# ---- calls and results ----

def test_speak_schema_exposes_timestamps_and_pronunciations(caps_doc, openapi_doc):
    schema = _gen(caps_doc, openapi_doc, "muse_speak").tool.inputSchema
    props = schema["properties"]
    assert props["timestamps"]["type"] == "boolean"
    assert props["timestamps"]["default"] is False
    assert "pronunciations" in props
    jsonschema.validate({
        "model": "kokoro-82m", "input": "hi", "timestamps": True,
        "pronunciations": [{"term": "hi", "respell": "hello"}],
    }, schema)


@pytest.mark.parametrize("timestamps", [True, False], ids=["json", "binary"])
def test_speak_writes_audio_from_json_and_binary(caps_doc, openapi_doc, media_env,
                                                timestamps):
    wav = _wav()
    audio_sha256 = hashlib.sha256(wav).hexdigest()
    digest = "c" * 64
    headers = {
        **HEADERS,
        "X-Muse-Voice": "af_heart", "X-Muse-Speed": "1.25",
        "X-Muse-Audio-SHA256": audio_sha256,
        "X-Muse-Pronunciations-Digest": digest,
    }
    payload = {
        "object": "audio.speech", "model": "kokoro-82m",
        "voice": "af_heart", "speed": 1.25,
        "audio": {
            "format": "wav", "b64_json": base64.b64encode(wav).decode(),
            "sha256": audio_sha256, "sample_rate": 8000, "duration_seconds": 0.5,
        },
        "words": [{"word": "hi", "char_start": 0, "char_end": 2,
                   "start": 0.0, "end": 0.4, "measured": True}],
        "timing": {"source": "native", "measured_words": 1,
                   "estimated_words": 0, "calibration": None},
        "pronunciations": {
            "digest": digest, "applied": [{"term": "hi", "kind": "respell", "count": 1}],
        },
    }
    resp = (httpx.Response(200, json=payload, headers=headers) if timestamps else
            httpx.Response(200, content=wav,
                           headers={**headers, "content-type": "audio/wav"}))
    blocks = _render(resp, _gen(caps_doc, openapi_doc, "muse_speak"),
                     out_path=None, preview=False, model="kokoro-82m")
    data = _structured(blocks)
    jsonschema.validate(data, OUTPUT_SCHEMA)
    [written] = data["files"]
    assert written["mime"] == "audio/wav"
    assert written["sha256"] == audio_sha256
    assert open(written["path"], "rb").read() == wav
    assert data["model"] == "kokoro-82m" and data["revision"] == "b" * 40
    expected = {
        "voice": "af_heart", "speed": 1.25, "audio_sha256": audio_sha256,
        "pronunciations_digest": digest,
    }
    if timestamps:
        expected = {**payload, **expected,
                    "audio": {**payload["audio"], "b64_json": written["path"]}}
    assert data["response"] == expected


def test_speak_writes_wav_and_returns_link(caps_doc, openapi_doc, fake_http, media_env):
    wav = _wav(0.5)
    client = fake_http(lambda r: httpx.Response(
        200, content=wav, headers={"content-type": "audio/wav", **HEADERS}))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_speak"),
                            {"model": "kokoro-82m", "input": "hi"})
    assert any(b["type"] == "resource_link" for b in blocks)
    assert not any(b["type"] == "audio" for b in blocks)
    data = _structured(blocks)
    jsonschema.validate(data, OUTPUT_SCHEMA)
    [f] = data["files"]
    assert f["sha256"] == hashlib.sha256(wav).hexdigest()
    assert f["mime"] == "audio/wav" and f["duration_seconds"] == 0.5
    assert open(f["path"], "rb").read() == wav
    assert f["path"].startswith(str(media_env / "out"))
    assert data["model"] == "kokoro-82m" and data["revision"] == "b" * 40
    assert data["response"] is None
    text = blocks[0]["text"]
    assert blocks[0]["type"] == "text"
    assert "b" * 40 in text and f["path"] in text and "0.5" in text
    link = next(b for b in blocks if b["type"] == "resource_link")
    assert link["uri"].startswith("file://") and link["mimeType"] == "audio/wav"


def test_out_path_is_honored(caps_doc, openapi_doc, fake_http, media_env):
    client = fake_http(lambda r: httpx.Response(
        200, content=_wav(), headers={"content-type": "audio/wav"}))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_speak"),
                            {"model": "kokoro-82m", "input": "hi",
                             "out_path": "take1.wav"})
    assert _structured(blocks)["files"][0]["path"] == str(media_env / "take1.wav")
    assert (media_env / "take1.wav").exists()


def test_b64_media_fields_written_to_files(caps_doc, openapi_doc, fake_http, media_env):
    png = _png()
    client = fake_http(lambda r: httpx.Response(200, json={
        "created": 1,
        "data": [{"b64_json": base64.b64encode(png).decode(), "revised_prompt": "x"}],
    }))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_generate_image"),
                            {"model": "sd-turbo", "prompt": "a fox"})
    data = _structured(blocks)
    path = data["response"]["data"][0]["b64_json"]
    assert path == data["files"][0]["path"] and path.endswith(".png")
    assert open(path, "rb").read() == png
    assert data["response"]["data"][0]["revised_prompt"] == "x"
    assert data["files"][0]["width"] == 4 and data["files"][0]["height"] == 3
    assert base64.b64encode(png).decode() not in json.dumps(blocks[0])


def test_data_url_in_url_field_written(caps_doc, openapi_doc, fake_http, media_env):
    png = _png()
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    client = fake_http(lambda r: httpx.Response(200, json={"data": [{"url": url}]}))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_generate_image"),
                            {"model": "sd-turbo", "prompt": "a fox",
                             "response_format": "url"})
    data = _structured(blocks)
    assert data["response"]["data"][0]["url"] == data["files"][0]["path"]


def test_rle_masks_are_left_untouched(caps_doc, openapi_doc, fake_http, media_env):
    (media_env / "in.png").write_bytes(_png())
    rle = {"size": [3, 4], "counts": [12]}
    client = fake_http(lambda r: httpx.Response(200, json={
        "id": "s", "masks": [{"index": 0, "mask": rle, "score": 0.9}]}))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_segment_image"),
                            {"model": "sam2-hiera-tiny", "image_path": "in.png",
                             "mask_format": "rle"})
    data = _structured(blocks)
    assert data["files"] == []
    assert data["response"]["masks"][0]["mask"] == rle


def test_image_preview_only_on_request(caps_doc, openapi_doc, fake_http, media_env):
    png = _png(1024, 600)
    client = fake_http(lambda r: httpx.Response(200, json={
        "data": [{"b64_json": base64.b64encode(png).decode()}]}))
    gen = _gen(caps_doc, openapi_doc, "muse_generate_image")
    blocks = call_generated(client, gen, {"model": "sd-turbo", "prompt": "x"})
    assert not any(b["type"] == "image" for b in blocks)
    blocks = call_generated(client, gen, {"model": "sd-turbo", "prompt": "x",
                                          "preview": True})
    [img] = [b for b in blocks if b["type"] == "image"]
    from PIL import Image
    shown = Image.open(io.BytesIO(base64.b64decode(img["data"])))
    assert max(shown.size) == 512 and img["mimeType"] == "image/png"


def test_multipart_upload_from_local_path(caps_doc, openapi_doc, fake_http, media_env):
    wav = _wav()
    (media_env / "take.wav").write_bytes(wav)
    seen = {}

    def handler(request):
        seen["ct"] = request.headers["content-type"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"words": [{"word": "hi", "start": 0.0}]},
                              headers={"x-muse-model": "qwen3-forced-aligner-0.6b"})

    client = fake_http(handler)
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_align"),
                            {"model": "qwen3-forced-aligner-0.6b",
                             "file_path": "take.wav", "text": "hi"})
    assert seen["ct"].startswith("multipart/form-data")
    assert wav in seen["body"] and b'filename="take.wav"' in seen["body"]
    assert b'name="text"' in seen["body"]
    data = _structured(blocks)
    assert data["response"]["words"][0]["word"] == "hi"
    assert '"word": "hi"' in blocks[0]["text"]


def test_multipart_data_url_input_is_decoded(caps_doc, openapi_doc, fake_http, media_env):
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(200, json={"text": "ok"})

    url = "data:audio/wav;base64," + base64.b64encode(b"RIFFxxxx").decode()
    call_generated(fake_http(handler), _gen(caps_doc, openapi_doc, "muse_transcribe"),
                   {"model": "whisper-base", "file_url": url})
    assert b"RIFFxxxx" in seen["body"]


def test_json_body_image_path_becomes_data_url(caps_doc, openapi_doc, fake_http, media_env):
    png = _png()
    (media_env / "ref.png").write_bytes(png)
    client = fake_http(lambda r: httpx.Response(200, json={"data": []}))
    call_generated(client, _gen(caps_doc, openapi_doc, "muse_generate_image"),
                   {"model": "sd-turbo", "prompt": "x", "image_path": "ref.png"})
    sent = json.loads(fake_http.requests[-1].content)
    assert sent["image"] == "data:image/png;base64," + base64.b64encode(png).decode()
    assert "image_path" not in sent


def test_file_input_rules_checked_before_http(caps_doc, openapi_doc, fake_http, media_env):
    gen = _gen(caps_doc, openapi_doc, "muse_align")
    client = fake_http(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="file_path or file_url"):
        call_generated(client, gen, {"model": "qwen3-forced-aligner-0.6b", "text": "hi"})
    (media_env / "a.wav").write_bytes(_wav())
    with pytest.raises(ValueError, match="only one"):
        call_generated(client, gen, {"model": "qwen3-forced-aligner-0.6b", "text": "hi",
                                     "file_path": "a.wav", "file_url": "data:,x"})
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        call_generated(client, gen, {"model": "qwen3-forced-aligner-0.6b", "text": "hi",
                                     "file_path": "/etc/hostname"})
    assert not [r for r in fake_http.requests if r.url.path == "/v1/audio/alignments"]


def test_large_json_written_to_file(caps_doc, openapi_doc, fake_http, media_env):
    body = {"object": "list", "model": "m",
            "data": [{"index": 0, "embedding": [0.123456789] * 3000}]}
    client = fake_http(lambda r: httpx.Response(200, json=body))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_embed_text"),
                            {"model": "all-minilm-l6-v2", "input": ["a"]})
    data = _structured(blocks)
    jsonschema.validate(data, OUTPUT_SCHEMA)
    [f] = data["files"]
    assert f["mime"] == "application/json"
    assert json.load(open(f["path"])) == body
    assert data["response"]["written_to"] == f["path"]
    assert len(json.dumps(blocks[0])) < 5000


def test_http_error_raises(caps_doc, openapi_doc, fake_http, media_env):
    client = fake_http(lambda r: httpx.Response(
        400, json={"error": {"code": "bad", "message": "nope"}}))
    with pytest.raises(httpx.HTTPStatusError):
        call_generated(client, _gen(caps_doc, openapi_doc, "muse_speak"),
                       {"model": "kokoro-82m", "input": "hi"})


def test_routes_with_no_serving_model_take_free_model(caps_doc, openapi_doc):
    for m in caps_doc["modalities"]:
        if m["modality"] == "audio/quality":
            m["models"] = []
    schema = _gen(caps_doc, openapi_doc, "muse_assess_audio_quality").tool.inputSchema
    assert "enum" not in schema["properties"]["model"]
    assert "model" not in schema["required"]


# ---- review fixes ----

def _form_fields(body: bytes) -> list[tuple[str, str]]:
    """(name, value) for each non-file part of a multipart body."""
    import re

    return [(m.group(1).decode(), m.group(2).decode())
            for m in re.finditer(rb'name="([^"]+)"\r\n\r\n(.*?)\r\n--', body, re.S)]


def test_property_names_are_mcp_safe(caps_doc, openapi_doc):
    import re

    for g in generate_tools(caps_doc, openapi_doc):
        for name in g.tool.inputSchema["properties"]:
            assert re.match(r"^[A-Za-z0-9_.-]{1,64}$", name), (g.name, name)


def test_multipart_list_is_sent_as_repeated_fields(caps_doc, openapi_doc, fake_http,
                                                   media_env):
    (media_env / "a.wav").write_bytes(_wav())
    seen = {}

    def handler(request):
        seen["fields"] = _form_fields(request.read())
        return httpx.Response(200, json={"text": "hi"})

    gen = _gen(caps_doc, openapi_doc, "muse_transcribe")
    assert gen.tool.inputSchema["properties"]["timestamp_granularities"]["type"] == "array"
    call_generated(fake_http(handler), gen, {
        "model": "whisper-base", "file_path": "a.wav", "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"]})
    assert [v for k, v in seen["fields"] if k == "timestamp_granularities[]"] == [
        "word", "segment"]


def test_alignment_header_lands_in_response(caps_doc, openapi_doc, fake_http, media_env):
    alignment = [{"word": "hi", "start": 0.0, "end": 0.4}]
    client = fake_http(lambda r: httpx.Response(200, content=_wav(), headers={
        "content-type": "audio/wav", "x-alignment": json.dumps(alignment)}))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_speak"),
                            {"model": "kokoro-82m", "input": "hi", "align": True})
    data = _structured(blocks)
    jsonschema.validate(data, OUTPUT_SCHEMA)
    assert data["response"] == {"alignment": alignment}
    assert len(data["files"]) == 1
    assert '"alignment"' in blocks[0]["text"]


def test_preview_unavailable_is_reported(caps_doc, openapi_doc, fake_http, media_env,
                                         monkeypatch):
    from muse.mcp import generated

    monkeypatch.setattr(generated.files, "preview_png", lambda data: None)
    monkeypatch.setattr(generated, "_has_pillow", lambda: False)
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 16).decode()
    client = fake_http(lambda r: httpx.Response(200, json={"data": [{"b64_json": png}]}))
    blocks = call_generated(client, _gen(caps_doc, openapi_doc, "muse_generate_image"),
                            {"model": "sd-turbo", "prompt": "x", "preview": True})
    assert not any(b["type"] == "image" for b in blocks)
    assert "preview unavailable: Pillow not installed" in blocks[0]["text"]


def test_out_path_names_the_spill_on_json_only_routes(caps_doc, openapi_doc, fake_http,
                                                      media_env):
    gen = _gen(caps_doc, openapi_doc, "muse_embed_text")
    assert "too large" in gen.tool.inputSchema["properties"]["out_path"]["description"]
    body = {"data": [{"embedding": [0.123456789] * 3000}]}
    client = fake_http(lambda r: httpx.Response(200, json=body))
    blocks = call_generated(client, gen, {"model": "all-minilm-l6-v2", "input": ["a"],
                                          "out_path": "emb.json"})
    assert _structured(blocks)["files"][0]["path"] == str(media_env / "emb.json")
    assert json.load(open(media_env / "emb.json")) == body


SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><circle cx="4" cy="4" r="3"/></svg>'


def test_vectorize_svg_text_written_to_file(caps_doc, openapi_doc, fake_http, media_env):
    png = _png()
    (media_env / "logo.png").write_bytes(png)
    client = fake_http(lambda r: httpx.Response(200, json={
        "id": "v1", "object": "image.vectorization", "model": "m", "mime_type": "image/svg+xml",
        "svg": SVG, "width": 8, "height": 8, "view_box": "0 0 8 8",
    }))
    gen = _gen(caps_doc, openapi_doc, "muse_vectorize_image")
    assert "written to local files" in gen.tool.description
    blocks = call_generated(client, gen, {"model": "m", "image_path": "logo.png"})
    data = _structured(blocks)
    [f] = data["files"]
    assert f["mime"] == "image/svg+xml" and f["path"].endswith(".svg")
    assert data["response"]["svg"] == f["path"]
    assert open(f["path"], encoding="utf-8").read() == SVG
    assert data["response"]["view_box"] == "0 0 8 8"
    assert SVG not in json.dumps(blocks)


# ---- multipart list file fields ----

def _part_names(body: bytes) -> list[str]:
    import re

    return [m.decode() for m in re.findall(rb'name="([^"]+)"', body)]


def test_multipart_list_field_sends_repeated_parts(caps_doc, openapi_doc, fake_http,
                                                   media_env):
    for n in "abc":
        (media_env / f"{n}.png").write_bytes(_png())
    (media_env / "m.png").write_bytes(_png())
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(200, json={"data": []})

    gen = _gen(caps_doc, openapi_doc, "muse_edit_image")
    props = gen.tool.inputSchema["properties"]
    assert "image[]" not in props and "image" not in props
    assert "image_paths" in props and "mask_path" in props
    for key in ("image_paths", "image_urls"):
        assert "At least one image is required" in props[key]["description"]
        assert "Optional." not in props[key]["description"]
    assert props["mask_path"]["description"].endswith("Optional.")
    call_generated(fake_http(handler), gen, {
        "model": "sd-turbo", "prompt": "x",
        "image_paths": ["a.png", "b.png", "c.png"], "mask_path": "m.png"})
    names = _part_names(seen["body"])
    assert names.count("image[]") == 3
    assert "image" not in names and "mask" in names


def test_json_list_field_still_data_urls(caps_doc, openapi_doc, fake_http, media_env):
    png = _png()
    (media_env / "a.png").write_bytes(png)
    gen = _gen(caps_doc, openapi_doc, "muse_embed_image")
    call_generated(fake_http(lambda r: httpx.Response(200, json={"data": []})), gen,
                   {"model": "dinov2-small", "input_paths": ["a.png"]})
    sent = json.loads(fake_http.requests[-1].content)
    assert sent["input"] == ["data:image/png;base64," + base64.b64encode(png).decode()]


def test_multipart_list_url_fetched(caps_doc, openapi_doc, fake_http, media_env):
    png = _png()
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        return httpx.Response(200, json={"data": []})

    call_generated(fake_http(handler), _gen(caps_doc, openapi_doc, "muse_edit_image"),
                   {"model": "sd-turbo", "prompt": "x", "image_urls": [url, url]})
    assert _part_names(seen["body"]).count("image[]") == 2
    assert seen["body"].count(png) == 2


@pytest.mark.parametrize("value", ["123", "true", "null", "1e20"])
def test_speech_identity_headers_stay_strings(caps_doc, openapi_doc, media_env, value):
    resp = httpx.Response(200, content=_wav(), headers={
        "content-type": "audio/wav", "x-muse-voice": value,
        "x-muse-audio-sha256": value, "x-muse-pronunciations-digest": value,
        "x-muse-speed": "1.25",
    })
    data = _structured(_render(resp, _gen(caps_doc, openapi_doc, "muse_speak"),
                               out_path=None, preview=False, model="kokoro-82m"))
    assert data["response"] == dict(voice=value, audio_sha256=value,
                                    pronunciations_digest=value, speed=1.25)


# ---- mcp_defaults are checked against the route's request schema ----


def _defaults_docs(defaults: dict, with_schema: bool = True):
    route = {"method": "POST", "path": "/v1/x/run", "mcp_tool": "muse_x_run",
             "request_content_type": "application/json",
             "response_content_type": "application/json",
             "example": {"model": "<model>", "prompt": "hi"},
             "mcp_defaults": defaults}
    caps = {"modalities": [{"modality": "x/run", "summary": "x", "models": ["m"],
                            "routes": [route]}],
            "models": [{"id": "m", "modality": "x/run"}]}
    schema = {"type": "object", "properties": {
        "model": {"type": "string"}, "prompt": {"type": "string"},
        "response_format": {"type": "string", "default": "json"}}}
    openapi = {"paths": {"/v1/x/run": {"post": {"requestBody": {"content": {
        "application/json": {"schema": schema}}}}}}} if with_schema else {"paths": {}}
    return caps, openapi


def test_mcp_defaults_not_in_the_schema_are_dropped_with_one_warning(caplog):
    import logging

    caps, openapi = _defaults_docs({"response_format": "zip", "respons_format": "zip",
                                    "n": 2})
    with caplog.at_level(logging.WARNING, logger="muse.mcp.generated"):
        [gen] = generate_tools(caps, openapi)
        [again] = generate_tools(caps, openapi)
    assert gen.defaults == {"response_format": "zip"} == again.defaults
    props = gen.tool.inputSchema["properties"]
    assert props["response_format"]["default"] == "zip"
    assert "respons_format" not in props and "n" not in props
    assert "respons_format" not in gen.tool.description
    warned = [r.getMessage() for r in caplog.records if "mcp_defaults" in r.getMessage()]
    assert len(warned) == 2  # one per bad key, not one per generation
    assert any("respons_format" in m and "muse_x_run" in m for m in warned)


def test_valid_mcp_defaults_are_sent(fake_http):
    caps, openapi = _defaults_docs({"response_format": "zip", "bogus": True})
    [gen] = generate_tools(caps, openapi)
    seen = []

    def handler(request):
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"ok": True})

    call_generated(fake_http(handler), gen, {"model": "m", "prompt": "p"})
    assert seen[0]["response_format"] == "zip" and "bogus" not in seen[0]


def test_mcp_defaults_kept_when_the_route_publishes_no_schema():
    """Without a request schema the example is a guess and extra fields pass
    through, so the declared defaults are kept."""
    caps, openapi = _defaults_docs({"response_format": "zip"}, with_schema=False)
    [gen] = generate_tools(caps, openapi)
    assert gen.defaults == {"response_format": "zip"}
