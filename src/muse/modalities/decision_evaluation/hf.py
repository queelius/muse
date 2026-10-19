"""Hugging Face resolver for the two supported decision models.

Exact-repo only. A decision model's prompt or head format cannot be
inferred from tags (Decima's card says "zero-shot-classification", the
Decision GGUF ships `.gguf` files a chat plugin would claim), so this
plugin claims exactly the repositories in `SOURCES`, at a priority ahead
of every other plugin, and refuses any revision but the pinned one.

Each source pins a commit, an exact download allow-list, the size (and
for large files the SHA-256) of every required file, the runtime, its
pip extras and its manifest capabilities. Downloads are verified before
the pull commits.

Single-file import (see docs/HF_PLUGINS.md): stdlib, huggingface_hub and
muse.core only.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import (
    ResolvedModel,
    ResolverError,
    SearchResult,
    hf_commit_revision,
)


_MODALITY = "decision/evaluation"
_RUNTIMES = "muse.modalities.decision_evaluation.runtimes"
_QUESTION_TYPES = ["choice", "score", "noul"]
_LIMITS = {
    "max_questions": 16,
    "max_choice_options": 24,
    "max_score_levels": 10,
    "max_body_bytes": 128 * 1024,
    "max_state_bytes": 64 * 1024,
}

# Per repo (lowercase key): everything a pull of that repo is allowed to do.
# files: {path: (size_bytes, sha256 or None)}; every file must be present
# after download with exactly that size (and hash, when given).
SOURCES: dict[str, dict] = {
    "evalengine/decision-0.8b-gguf": {
        "model_id": "decision-0.8b-q8",
        "revision": "ce993792ad283053004bf0230b1709ff7351bb24",
        "runtime_path": f"{_RUNTIMES}.llama_cpp:DecisionLlamaCppModel",
        "pip_extras": ("llama-cpp-python==0.3.33",),
        "license": "apache-2.0",
        "description": (
            "Decision-0.8B Q8_0 GGUF (Eval Engine): Choice/Score/Noul by "
            "candidate-letter logits over Qwen3.5-0.8B; probabilities uncalibrated"
        ),
        "files": {
            "decision-0.8b-Q8_0.gguf": (
                811_843_168,
                "efa1614fe5c7d84acb6b8acdf9fd0dda4a26acf110302732baa573d3332a4fef",
            ),
            "LICENSE": (11_355, None),
            "NOTICE": (2_385, None),
            "licenses/QWEN-LICENSE": (11_544, None),
            "licenses/TEV-LICENSE": (1_078, None),
        },
        "capabilities": {
            "gguf_file": "decision-0.8b-Q8_0.gguf",
            "device": "auto",
            "n_gpu_layers": -1,
            "context_length": 2048,
            "n_threads": 2,
            # File size plus a 2,048-token context and runtime overhead.
            "memory_gb": 1.5,
            "max_concurrency": 1,
            "question_types": _QUESTION_TYPES,
            "input_modalities": ["text"],
            "max_input_tokens_per_question": 2048,
            "precision": "Q8_0",
            "probabilities": "softmax over the listed option letters' final logits",
            "calibrated": False,
            **_LIMITS,
        },
    },
    "amyrmahdy/decima-base": {
        "model_id": "decima-base",
        "revision": "e5d84ab3a8436cedc45b0daefa91979f68319455",
        "runtime_path": f"{_RUNTIMES}.decima_onnx:DecimaOnnxModel",
        "pip_extras": ("decima-ai==2.2.0", "onnxruntime==1.24.4"),
        "license": "apache-2.0",
        "description": (
            "Decima-base INT8 ONNX (A. M. Madani): Choice/Score/Noul by a "
            "late-interaction scorer on mmBERT-base, temperature-calibrated, CPU; "
            "training-data licensing caveats: "
            "https://github.com/amyrmahdy/decima/blob/main/release/LICENSING.md"
        ),
        "files": {
            "config.json": (350, None),
            "LICENSE": (11_358, None),
            "onnx/int8/decima.json": (45_015, None),
            "onnx/int8/encoder.onnx": (
                322_447_126,
                "2cc99fa53555ecddb42c5cedc933cfeedb448bee89dd2d7868525fea1d25b3b9",
            ),
            "onnx/int8/scorer.onnx": (
                14_436_752,
                "98b64990f7f8c0a758f74ccefe15aa91a6afb4e1d037156d7e662c82779c43a4",
            ),
            "onnx/int8/tokenizer/tokenizer.json": (
                34_363_440,
                "d941237f7b665542273cd22102ae14fdacf54d3aaf0bd45da55022c46b4cc006",
            ),
            "onnx/int8/tokenizer/tokenizer_config.json": (735, None),
        },
        "capabilities": {
            "device": "cpu",
            "n_threads": 2,
            "option_cache_size": 256,
            "temperature": 0.7269418400253115,
            # ~371 MB of INT8 weights and tokenizer, plus onnxruntime arenas.
            "memory_gb": 1.5,
            "max_concurrency": 1,
            "question_types": _QUESTION_TYPES,
            "input_modalities": ["text"],
            "max_state_tokens": 512,
            "max_option_tokens": 64,
            "precision": "int8",
            "probabilities": "temperature-scaled softmax (choose/verify) or ordinal head (score)",
            "calibrated": True,
            "calibration": "one fitted temperature from the publisher; not validated for arbitrary tasks",
            **_LIMITS,
        },
    },
}


def _source(repo_id: str) -> dict | None:
    return SOURCES.get((repo_id or "").lower())


def _sniff(info) -> bool:
    # Exact repo only: tags and configs are not evidence of a decision
    # format. A claimed repo with the wrong shape fails in _resolve rather
    # than falling through to a generic plugin.
    return _source(getattr(info, "id", "") or "") is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, files: dict[str, tuple[int, str | None]]) -> None:
    """Raise ResolverError unless every pinned file matches size and hash."""
    for name, (size, sha256) in files.items():
        path = root / name
        if not path.is_file():
            raise ResolverError(f"download is missing pinned file {name}")
        actual = path.stat().st_size
        if actual != size:
            raise ResolverError(
                f"pinned file {name} is {actual} bytes; expected {size}"
            )
        if sha256 is not None and _sha256(path) != sha256:
            raise ResolverError(f"pinned file {name} does not match its SHA-256")


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    source = _source(repo_id)
    if source is None:
        raise ResolverError(f"{repo_id!r} is not a supported decision model")
    if variant:
        raise ResolverError(f"{repo_id!r} has no variants; drop '@{variant}'")
    revision = hf_commit_revision(info)
    if revision != source["revision"]:
        raise ResolverError(
            f"{repo_id!r} is supported only at commit {source['revision']}; "
            f"the Hub resolved {revision!r}. Pull the curated id "
            f"{source['model_id']!r}, which pins that commit."
        )
    siblings = {s.rfilename for s in (getattr(info, "siblings", None) or [])}
    missing = sorted(set(source["files"]) - siblings)
    if missing:
        raise ResolverError(
            f"{repo_id!r} at {revision} lacks pinned files: {', '.join(missing)}"
        )

    allow_patterns = list(source["files"])
    manifest = {
        "model_id": source["model_id"],
        "modality": _MODALITY,
        "hf_repo": repo_id,
        "revision": revision,
        "description": source["description"],
        "license": source["license"],
        "pip_extras": list(source["pip_extras"]),
        "system_packages": [],
        "capabilities": dict(source["capabilities"]),
    }

    def _download(cache_root: Path) -> Path:
        root = Path(snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))
        verify_files(root, source["files"])
        return root

    return ResolvedModel(
        manifest=manifest,
        backend_path=source["runtime_path"],
        download=_download,
        artifact_provenance=({
            "repo_id": repo_id,
            "revision": revision,
            "subdir": ".",
            "allow_patterns": allow_patterns,
            "required_patterns": allow_patterns,
        },),
    )


_SEARCH_WORDS = (
    "decision decima jev systemone system one typesafe choice score noul "
    "classify classifier judge evaluation"
)
_SIZES_GB = {"evalengine/decision-0.8b-gguf": 0.81, "amyrmahdy/decima-base": 0.37}


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
    if limit <= 0 or not all(t in _SEARCH_WORDS for t in tokens):
        return
    for key, source in list(SOURCES.items())[:limit]:
        yield SearchResult(
            uri=f"hf://{key}",
            model_id=source["model_id"],
            modality=_MODALITY,
            size_gb=_SIZES_GB[key],
            downloads=None,
            license=source["license"],
            description=source["description"],
            metadata={"revision": source["revision"]},
        )


HF_PLUGIN = {
    "modality": _MODALITY,
    "runtime_path": SOURCES["evalengine/decision-0.8b-gguf"]["runtime_path"],
    "pip_extras": (),
    "system_packages": (),
    # Ahead of every generic plugin (chat GGUF 100, text classification 200).
    "priority": 80,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
