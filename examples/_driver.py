"""Driver: exercise a running `muse serve` across modalities and record
outputs + provenance under examples/<modality>/.

Usage:
    python examples/_driver.py <modality> [<modality> ...]
    python examples/_driver.py all

Each modality function writes its artifact(s) and a metadata.json carrying
full provenance (model, endpoint, request, timing, version, commit, time).
Functions are independent and re-runnable; a failure in one is recorded and
does not stop the others.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # examples/
REPO = ROOT.parent
SERVER = os.environ.get("MUSE_SERVER", "http://localhost:8000")

try:
    from muse import __version__ as MUSE_VERSION
except Exception:
    MUSE_VERSION = "unknown"


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


COMMIT = _commit()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def modality_dir(name: str) -> Path:
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_metadata(modality: str, model: str, endpoint: str, runs: list[dict]) -> None:
    """Write examples/<modality>/metadata.json with provenance."""
    name = modality.replace("/", "_")
    meta = {
        "modality": modality,
        "model": model,
        "endpoint": endpoint,
        "muse_version": MUSE_VERSION,
        "server_commit": COMMIT,
        "server_url": SERVER,
        "device": "auto",
        "generated_at": _now(),
        "runs": runs,
    }
    (modality_dir(name) / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def wav_info(b: bytes) -> dict:
    import io
    try:
        with wave.open(io.BytesIO(b)) as w:
            return {
                "format": "wav",
                "bytes": len(b),
                "channels": w.getnchannels(),
                "sample_rate": w.getframerate(),
                "frames": w.getnframes(),
                "seconds": round(w.getnframes() / float(w.getframerate()), 3),
            }
    except Exception:
        return {"format": "wav", "bytes": len(b)}


def png_info(b: bytes) -> dict:
    info = {"format": "png", "bytes": len(b)}
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(b))
        info["size"] = f"{im.width}x{im.height}"
        info["mode"] = im.mode
    except Exception:
        pass
    return info


# ----------------------------------------------------------------------------
# Per-modality exercises
# ----------------------------------------------------------------------------

def speech() -> dict:
    from muse.modalities.audio_speech import SpeechClient
    name = "audio_speech"
    d = modality_dir(name)
    model = "kokoro-82m"
    text = "Hello from muse. This sentence was synthesized on a CPU by the Kokoro text to speech model."
    out = d / f"{model}_hello.wav"
    t0 = time.time()
    audio = SpeechClient().infer(text, out_path=str(out), response_format="wav", model=model)
    dt = round(time.time() - t0, 3)
    info = wav_info(audio)
    run = {
        "artifact": out.name,
        "request": {"endpoint": "POST /v1/audio/speech", "input": text,
                    "response_format": "wav", "model": model,
                    "voice": "af_heart (muse default; client omitted voice)"},
        "latency_seconds": dt,
        "output": info,
        "timestamp": _now(),
        "status": "ok",
    }
    write_metadata("audio/speech", model, "POST /v1/audio/speech", [run])
    print(f"[ok] audio/speech -> {out.name} ({info.get('seconds','?')}s audio, {dt}s latency)")
    return {"modality": "audio/speech", "status": "ok", "artifact": str(out)}


def transcription() -> dict:
    """Provenance chain: transcribe the speech we just generated."""
    from muse.modalities.audio_transcription import TranscriptionClient
    name = "audio_transcription"
    d = modality_dir(name)
    model = "whisper-base"
    src = ROOT / "audio_speech" / "kokoro-82m_hello.wav"
    if not src.exists():
        raise FileNotFoundError(f"need {src} first (run speech)")
    audio = src.read_bytes()
    t0 = time.time()
    result = TranscriptionClient().transcribe(
        audio=audio, filename=src.name, model=model, response_format="verbose_json",
    )
    dt = round(time.time() - t0, 3)
    out = d / "transcript.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    (d / "transcript.txt").write_text(text.strip() + "\n")
    run = {
        "artifact": "transcript.json",
        "request": {"endpoint": "POST /v1/audio/transcriptions", "model": model,
                    "response_format": "verbose_json"},
        "derived_from": "examples/audio_speech/kokoro-82m_hello.wav",
        "latency_seconds": dt,
        "output": {"text": text.strip()},
        "timestamp": _now(),
        "status": "ok",
    }
    write_metadata("audio/transcription", model, "POST /v1/audio/transcriptions", [run])
    print(f"[ok] audio/transcription -> '{text.strip()[:70]}...' ({dt}s)")
    return {"modality": "audio/transcription", "status": "ok", "artifact": str(out)}


def image_gen() -> dict:
    from muse.modalities.image_generation import GenerationsClient
    name = "image_generation"
    d = modality_dir(name)
    model = "sd-turbo"
    prompt = "a serene mountain lake at sunset, oil painting, warm light"
    t0 = time.time()
    imgs = GenerationsClient().generate(prompt, model=model, size="512x512", steps=1, seed=42)
    dt = round(time.time() - t0, 3)
    out = d / f"{model}_mountain_lake.png"
    out.write_bytes(imgs[0])
    info = png_info(imgs[0])
    run = {
        "artifact": out.name,
        "request": {"endpoint": "POST /v1/images/generations", "prompt": prompt,
                    "model": model, "size": "512x512", "steps": 1, "seed": 42},
        "latency_seconds": dt,
        "output": info,
        "timestamp": _now(),
        "status": "ok",
    }
    write_metadata("image/generation", model, "POST /v1/images/generations", [run])
    print(f"[ok] image/generation -> {out.name} ({info.get('size','?')}, {dt}s)")
    return {"modality": "image/generation", "status": "ok", "artifact": str(out)}


def embedding() -> dict:
    from muse.modalities.embedding_text import EmbeddingsClient
    name = "embedding_text"
    d = modality_dir(name)
    model = "qwen3-embedding-0.6b"
    inputs = ["The cat sat on the mat.", "A feline rested on the rug.", "Quarterly revenue rose 12%."]
    t0 = time.time()
    vecs = EmbeddingsClient().embed(inputs, model=model)
    dt = round(time.time() - t0, 3)

    def cos(a, b):
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return round(dot / (na * nb), 4)

    sims = {"0-1 (paraphrase)": cos(vecs[0], vecs[1]),
            "0-2 (unrelated)": cos(vecs[0], vecs[2])}
    out = d / "embeddings.json"
    out.write_text(json.dumps({"inputs": inputs, "dims": len(vecs[0]),
                               "cosine_similarities": sims,
                               "vectors_preview": [v[:8] for v in vecs]}, indent=2) + "\n")
    run = {
        "artifact": "embeddings.json",
        "request": {"endpoint": "POST /v1/embeddings", "model": model, "input": inputs},
        "latency_seconds": dt,
        "output": {"dims": len(vecs[0]), "n": len(vecs), "cosine_similarities": sims},
        "timestamp": _now(),
        "status": "ok",
    }
    write_metadata("embedding/text", model, "POST /v1/embeddings", [run])
    print(f"[ok] embedding/text -> {len(vecs)}x{len(vecs[0])} dims; paraphrase sim={sims['0-1 (paraphrase)']} ({dt}s)")
    return {"modality": "embedding/text", "status": "ok", "artifact": str(out)}


def classification() -> dict:
    from muse.modalities.text_classification import ModerationsClient
    name = "text_classification"
    d = modality_dir(name)
    model = "text-moderation"
    inputs = ["I love this, what a wonderful day!", "I will hurt you and everyone you love."]
    t0 = time.time()
    results = ModerationsClient().classify(inputs, model=model)
    dt = round(time.time() - t0, 3)
    out = d / "moderations.json"
    out.write_text(json.dumps({"inputs": inputs, "results": results}, indent=2) + "\n")
    run = {
        "artifact": "moderations.json",
        "request": {"endpoint": "POST /v1/moderations", "model": model, "input": inputs},
        "latency_seconds": dt,
        "output": {"n": len(results)},
        "timestamp": _now(),
        "status": "ok",
    }
    write_metadata("text/classification", model, "POST /v1/moderations", [run])
    print(f"[ok] text/classification -> {len(results)} results ({dt}s)")
    return {"modality": "text/classification", "status": "ok", "artifact": str(out)}


def rerank() -> dict:
    from muse.modalities.text_rerank import RerankClient
    name = "text_rerank"
    d = modality_dir(name)
    model = "bge-reranker-v2-m3"
    query = "How do I reset my password?"
    documents = [
        "To reset your password, click 'Forgot password' on the login page.",
        "Our office is open Monday to Friday, 9am to 5pm.",
        "Password requirements: at least 12 characters with a symbol.",
        "The weather today is sunny with a high of 24 degrees.",
    ]
    t0 = time.time()
    env = RerankClient().rerank(query=query, documents=documents, model=model, return_documents=True)
    dt = round(time.time() - t0, 3)
    out = d / "rerank.json"
    out.write_text(json.dumps({"query": query, "documents": documents, "response": env}, indent=2) + "\n")
    run = {
        "artifact": "rerank.json",
        "request": {"endpoint": "POST /v1/rerank", "model": model, "query": query,
                    "n_documents": len(documents)},
        "latency_seconds": dt,
        "output": {"top_index": env["results"][0]["index"],
                   "top_score": env["results"][0]["relevance_score"]},
        "timestamp": _now(),
        "status": "ok",
    }
    write_metadata("text/rerank", model, "POST /v1/rerank", [run])
    print(f"[ok] text/rerank -> top doc #{env['results'][0]['index']} score={env['results'][0]['relevance_score']:.4f} ({dt}s)")
    return {"modality": "text/rerank", "status": "ok", "artifact": str(out)}


def summarization() -> dict:
    from muse.modalities.text_summarization import SummarizationClient
    d = modality_dir("text_summarization")
    model = "bart-large-cnn"
    text = (
        "The James Webb Space Telescope, launched in December 2021, is the largest "
        "and most powerful space telescope ever built. Positioned at the second "
        "Lagrange point about 1.5 million kilometres from Earth, it observes the "
        "universe primarily in the infrared. Its 6.5-metre segmented gold-coated "
        "mirror lets it peer through cosmic dust and capture light from the first "
        "galaxies that formed after the Big Bang. In its first years of operation "
        "it has imaged star-forming regions, exoplanet atmospheres, and some of the "
        "most distant galaxies ever observed, reshaping our understanding of the "
        "early universe."
    )
    t0 = time.time()
    env = SummarizationClient().summarize(text=text, length="short", format="paragraph", model=model)
    dt = round(time.time() - t0, 3)
    summary = env.get("summary", "")
    (d / "summary.txt").write_text(summary.strip() + "\n")
    (d / "summary.json").write_text(json.dumps({"input": text, "response": env}, indent=2) + "\n")
    run = {
        "artifact": "summary.txt",
        "request": {"endpoint": "POST /v1/summarize", "model": model,
                    "length": "short", "format": "paragraph", "input_chars": len(text)},
        "latency_seconds": dt,
        "output": {"summary": summary.strip(), "summary_chars": len(summary)},
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("text/summarization", model, "POST /v1/summarize", [run])
    print(f"[ok] text/summarization -> '{summary.strip()[:70]}...' ({dt}s)")
    return {"modality": "text/summarization", "status": "ok"}


def image_embedding() -> dict:
    from muse.modalities.image_embedding import ImageEmbeddingsClient
    d = modality_dir("image_embedding")
    model = "dinov2-small"
    src = ROOT / "image_generation" / "sd-turbo_mountain_lake.png"
    img = src.read_bytes()
    t0 = time.time()
    vecs = ImageEmbeddingsClient().embed(img, model=model)
    dt = round(time.time() - t0, 3)
    (d / "embedding.json").write_text(json.dumps(
        {"dims": len(vecs[0]), "vector_preview": vecs[0][:12]}, indent=2) + "\n")
    run = {
        "artifact": "embedding.json",
        "request": {"endpoint": "POST /v1/images/embeddings", "model": model},
        "derived_from": "examples/image_generation/sd-turbo_mountain_lake.png",
        "latency_seconds": dt,
        "output": {"dims": len(vecs[0]), "n": len(vecs)},
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("image/embedding", model, "POST /v1/images/embeddings", [run])
    print(f"[ok] image/embedding -> 1x{len(vecs[0])} dims ({dt}s)")
    return {"modality": "image/embedding", "status": "ok"}


def image_cv() -> dict:
    """image/cv hosts depth + detection; exercise both, one metadata.json."""
    from muse.modalities.image_cv import DepthClient, ObjectDetectionClient
    from muse.modalities.image_generation import GenerationsClient
    d = modality_dir("image_cv")
    runs = []
    # --- depth on the existing landscape ---
    src = ROOT / "image_generation" / "sd-turbo_mountain_lake.png"
    img = src.read_bytes()
    t0 = time.time()
    resp = DepthClient().estimate_depth(img, model="depth-anything-v2-small", response_format="png16")
    dt = round(time.time() - t0, 3)
    (d / "depth_map.png").write_bytes(base64.b64decode(resp["depth_map"]))
    runs.append({
        "artifact": "depth_map.png",
        "request": {"endpoint": "POST /v1/images/depth", "model": "depth-anything-v2-small",
                    "response_format": "png16"},
        "derived_from": "examples/image_generation/sd-turbo_mountain_lake.png",
        "latency_seconds": dt,
        "output": {"format": resp.get("format"), "size": f"{resp.get('width')}x{resp.get('height')}",
                   "min_depth": resp.get("min_depth"), "max_depth": resp.get("max_depth")},
        "timestamp": _now(), "status": "ok",
    })
    print(f"[ok] image/cv depth -> depth_map.png ({dt}s)")
    # --- detection: generate a scene likely to contain COCO objects ---
    prompt = "a person riding a bicycle on a city street with parked cars, daytime, photograph"
    gimgs = GenerationsClient().generate(prompt, model="sd-turbo", size="512x512", steps=1, seed=7)
    (d / "detect_input.png").write_bytes(gimgs[0])
    t0 = time.time()
    det = ObjectDetectionClient().detect_objects(gimgs[0], model="detr-resnet-50", threshold=0.5)
    dt = round(time.time() - t0, 3)
    (d / "detections.json").write_text(json.dumps(det, indent=2) + "\n")
    labels = [f"{o.get('label')}({o.get('score'):.2f})" for o in det.get("detections", [])]
    runs.append({
        "artifact": "detections.json",
        "request": {"endpoint": "POST /v1/images/detect", "model": "detr-resnet-50",
                    "threshold": 0.5, "input_prompt": prompt},
        "derived_from": "examples/image_cv/detect_input.png (sd-turbo generated)",
        "latency_seconds": dt,
        "output": {"n_detections": len(det.get("detections", [])), "labels": labels},
        "timestamp": _now(), "status": "ok",
    })
    write_metadata("image/cv", "depth-anything-v2-small, detr-resnet-50",
                   "POST /v1/images/{depth,detect}", runs)
    print(f"[ok] image/cv detect -> {len(labels)} objects: {labels} ({dt}s)")
    return {"modality": "image/cv", "status": "ok"}


def chat() -> dict:
    """chat/completion via smolvlm-256m-instruct (a VLM that also does text).

    Note: smolvlm's supports_vision gate was a real bug (the route 400'd every
    image request because the bundled manifest's capabilities were lost); that
    is FIXED (see INDEX, P4 - get_manifest now recovers capabilities for
    runtime-aliased bundled scripts). The route now correctly ACCEPTS image
    input. Image (VLM) inference itself, however, is impractically slow on this
    4-core CPU (a 256M VLM tiling/encoding an image takes minutes), so the
    demonstrated call here is text-only for a fast, reliable artifact.
    """
    from muse.modalities.chat_completion import ChatClient
    d = modality_dir("chat_completion")
    model = "smolvlm-256m-instruct"
    prompt = "In one short sentence, describe a serene mountain lake at sunset."
    messages = [{"role": "user", "content": prompt}]
    t0 = time.time()
    resp = ChatClient().chat(model=model, messages=messages, max_tokens=48)
    dt = round(time.time() - t0, 3)
    content = resp["choices"][0]["message"]["content"]
    (d / "response.txt").write_text(content.strip() + "\n")
    (d / "response.json").write_text(json.dumps(resp, indent=2) + "\n")
    run = {
        "artifact": "response.txt",
        "request": {"endpoint": "POST /v1/chat/completions", "model": model,
                    "prompt": prompt, "max_tokens": 48, "mode": "text-only"},
        "latency_seconds": dt,
        "output": {"content": content.strip(), "usage": resp.get("usage")},
        "note": ("model is a VLM; supports_vision gate fixed (P4) and the route "
                 "accepts image_url input, but CPU VLM image inference is too "
                 "slow to demo here, so this call is text-only"),
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("chat/completion", model, "POST /v1/chat/completions", [run])
    print(f"[ok] chat/completion (text) -> '{content.strip()[:70]}...' ({dt}s)")
    return {"modality": "chat/completion", "status": "ok"}


def audio_classification() -> dict:
    from muse.modalities.audio_classification import AudioClassificationsClient
    d = modality_dir("audio_classification")
    model = "ast-audioset"
    src = ROOT / "audio_speech" / "kokoro-82m_hello.wav"
    t0 = time.time()
    resp = AudioClassificationsClient().classify(src.read_bytes(), model=model, top_k=5)
    dt = round(time.time() - t0, 3)
    (d / "classifications.json").write_text(json.dumps(resp, indent=2) + "\n")
    # response mirrors text classifications: per-input list of {label, score}
    top = resp
    run = {
        "artifact": "classifications.json",
        "request": {"endpoint": "POST /v1/audio/classifications", "model": model, "top_k": 5},
        "derived_from": "examples/audio_speech/kokoro-82m_hello.wav",
        "latency_seconds": dt,
        "output": top,
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("audio/classification", model, "POST /v1/audio/classifications", [run])
    print(f"[ok] audio/classification -> top labels recorded ({dt}s)")
    return {"modality": "audio/classification", "status": "ok"}


def image_segmentation() -> dict:
    from muse.modalities.image_segmentation import ImageSegmentationClient
    d = modality_dir("image_segmentation")
    model = "sam2-hiera-tiny"
    src = ROOT / "image_generation" / "sd-turbo_mountain_lake.png"
    t0 = time.time()
    resp = ImageSegmentationClient().segment(image=src.read_bytes(), model=model,
                                             mode="auto", mask_format="png_b64", max_masks=8)
    dt = round(time.time() - t0, 3)
    masks = resp.get("masks", [])
    for i, m in enumerate(masks[:5]):
        (d / f"mask_{i}.png").write_bytes(base64.b64decode(m["mask"]))
    summary = {"image_size": resp.get("image_size"), "mode": resp.get("mode"),
               "n_masks": len(masks),
               "masks": [{"index": m["index"], "score": m["score"], "bbox": m["bbox"], "area": m["area"]}
                         for m in masks]}
    (d / "segmentation.json").write_text(json.dumps(summary, indent=2) + "\n")
    run = {
        "artifact": f"{min(len(masks),5)} mask PNGs + segmentation.json",
        "request": {"endpoint": "POST /v1/images/segment", "model": model,
                    "mode": "auto", "max_masks": 8},
        "derived_from": "examples/image_generation/sd-turbo_mountain_lake.png",
        "latency_seconds": dt,
        "output": {"n_masks": len(masks), "image_size": resp.get("image_size")},
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("image/segmentation", model, "POST /v1/images/segment", [run])
    print(f"[ok] image/segmentation -> {len(masks)} masks ({dt}s)")
    return {"modality": "image/segmentation", "status": "ok"}


def image_ocr() -> dict:
    from muse.modalities.image_ocr import OcrClient
    from PIL import Image, ImageDraw
    import io
    d = modality_dir("image_ocr")
    model = "trocr-base-printed"
    text = "Muse OCR test: the quick brown fox 12345"
    im = Image.new("RGB", (640, 64), "white")
    ImageDraw.Draw(im).text((10, 20), text, fill="black")
    buf = io.BytesIO(); im.save(buf, format="PNG"); png = buf.getvalue()
    (d / "ocr_input.png").write_bytes(png)
    t0 = time.time()
    resp = OcrClient().ocr(png, model=model)
    dt = round(time.time() - t0, 3)
    recognized = resp.get("text", "")
    (d / "ocr_result.json").write_text(json.dumps({"rendered_text": text, "response": resp}, indent=2) + "\n")
    run = {
        "artifact": "ocr_result.json + ocr_input.png",
        "request": {"endpoint": "POST /v1/images/ocr", "model": model},
        "derived_from": "examples/image_ocr/ocr_input.png (synthetic rendered text)",
        "latency_seconds": dt,
        "output": {"rendered_text": text, "recognized_text": recognized},
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("image/ocr", model, "POST /v1/images/ocr", [run])
    print(f"[ok] image/ocr -> rendered='{text}' recognized='{recognized}' ({dt}s)")
    return {"modality": "image/ocr", "status": "ok"}


def audio_embedding() -> dict:
    from muse.modalities.audio_embedding import AudioEmbeddingsClient
    d = modality_dir("audio_embedding")
    model = "mert-v1-95m"
    src = ROOT / "audio_speech" / "kokoro-82m_hello.wav"
    t0 = time.time()
    vecs = AudioEmbeddingsClient().embed(src.read_bytes(), model=model)
    dt = round(time.time() - t0, 3)
    (d / "embedding.json").write_text(json.dumps(
        {"dims": len(vecs[0]), "vector_preview": vecs[0][:12]}, indent=2) + "\n")
    run = {
        "artifact": "embedding.json",
        "request": {"endpoint": "POST /v1/audio/embeddings", "model": model},
        "derived_from": "examples/audio_speech/kokoro-82m_hello.wav",
        "latency_seconds": dt,
        "output": {"dims": len(vecs[0]), "n": len(vecs)},
        "timestamp": _now(), "status": "ok",
    }
    write_metadata("audio/embedding", model, "POST /v1/audio/embeddings", [run])
    print(f"[ok] audio/embedding -> 1x{len(vecs[0])} dims ({dt}s)")
    return {"modality": "audio/embedding", "status": "ok"}


REGISTRY = {
    "speech": speech,
    "transcription": transcription,
    "image_gen": image_gen,
    "embedding": embedding,
    "classification": classification,
    "rerank": rerank,
    "summarization": summarization,
    "image_embedding": image_embedding,
    "image_cv": image_cv,
    "chat": chat,
    "audio_classification": audio_classification,
    "image_segmentation": image_segmentation,
    "image_ocr": image_ocr,
    "audio_embedding": audio_embedding,
}


def main(argv: list[str]) -> int:
    names = list(REGISTRY) if (not argv or argv == ["all"]) else argv
    rc = 0
    for n in names:
        fn = REGISTRY.get(n)
        if fn is None:
            print(f"[skip] unknown modality '{n}'")
            continue
        try:
            fn()
        except Exception as e:
            rc = 1
            print(f"[FAIL] {n}: {type(e).__name__}: {e}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
