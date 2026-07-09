"""HF resolver plugin for text/translation.

Sniffs the `translation` task tag, plus a repo-name fallback for the
four families TranslationRuntime supports: m2m100, nllb, opus-mt,
madlad. The name fallback exists because HF repos in this space are
inconsistently tagged -- some carry `text2text-generation` instead of
(or in addition to) `translation`, and the name pattern is a reliable
tell regardless of tagging.

Disambiguation vs text/summarization's plugin (spec-normative, see
docs/superpowers/specs/2026-07-09-text-translation-design.md): a repo
tagged BOTH `summarization` and `translation` is claimed here only when
its name matches one of the four translation families; otherwise it is
left to text/summarization's plugin.

KNOWN LANDMINE (documented per docs/HF_PLUGINS.md priority-tie rule):
text/summarization's plugin also sits at priority 110, and
HFResolver.resolve() iterates plugins in (priority, modality) order,
first sniff wins. Tie-break at equal priority is alphabetical by
modality string, and "text/summarization" < "text/translation", so
summarization's plugin is *always* consulted before this one. Its own
sniff (`"summarization" in tags`) claims ANY summarization-tagged repo
unconditionally -- it does not consult the translation tag or the repo
name. The disambiguation logic in `_sniff` below therefore only fully
governs resolution when text/summarization's plugin has NOT already
claimed the repo (i.e. no `summarization` tag present at all); in the
both-tagged case, summarization's plugin wins the real dispatch
regardless of what this function returns. This is a real gap between
the spec's stated rule and the current priority scheme, not a bug in
this file -- fixing it would require lowering this plugin's priority
below 110 (a cross-cutting change out of scope for this task) or
special-casing modality order in the shared dispatcher. No known real
HF repo triggers the gap today: m2m100/nllb/opus-mt/madlad repos are
not tagged `summarization` in practice. `_sniff` still implements the
spec's ideal rule (and is covered by direct unit tests below) so the
correct behavior is pinned and ready the moment priority ordering is
revisited.

Loaded via single-file import; no relative imports. See
docs/HF_PLUGINS.md for authoring rules.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.text_translation.runtimes.hf_translation"
    ":TranslationRuntime"
)
_PIP_EXTRAS = ("torch>=2.1.0", "transformers>=4.36.0", "sentencepiece")

# Case-insensitive substring markers identifying the four families
# TranslationRuntime dispatches on. Mirrors
# muse.modalities.text_translation.runtimes.hf_translation._FAMILY_MARKERS
# (not imported -- hf.py may not import modality siblings; see
# docs/HF_PLUGINS.md authoring rules).
_FAMILY_PATTERNS = ("m2m100", "nllb", "opus-mt", "madlad")

# opus-mt-{src}-{tgt} where src/tgt are exactly two lowercase letters
# (plain ISO-639-1 codes). Anything else -- a language-group name like
# "ROMANCE", a 3-letter macro-language code, or a multi-part variant
# name like "opus-mt-tc-big-en-pt" -- does not match, and pair
# extraction is skipped rather than guessed.
_OPUS_PAIR_RE = re.compile(r"^opus-mt-([a-z]{2})-([a-z]{2})$")


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _matches_family(repo_id: str) -> bool:
    name = (repo_id or "").lower()
    return any(pattern in name for pattern in _FAMILY_PATTERNS)


def _opus_pair(repo_id: str) -> tuple[str, str] | None:
    """Parse a `src`/`tgt` ISO-639-1 pair from an opus-mt-shaped model
    name. Returns None (no guessing) when the name doesn't match the
    strict two-letter-pair shape.
    """
    name = _model_id(repo_id)
    m = _OPUS_PAIR_RE.fullmatch(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    repo_id = getattr(info, "id", "") or ""
    family_match = _matches_family(repo_id)

    if "summarization" in tags:
        # Both-tagged (or mistagged) disambiguation: only claim here
        # when the name unambiguously identifies a translation family.
        # A generic summarization repo (no translation tag, no family
        # name) must fall through to text/summarization's plugin.
        return family_match

    return "translation" in tags or family_match


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    capabilities: dict = {"device": "auto"}
    pair = _opus_pair(repo_id)
    if pair is not None:
        capabilities["source_language"], capabilities["target_language"] = pair

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "text/translation",
        "hf_repo": repo_id,
        "description": f"Translation model: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        # Keep weights light: prefer safetensors, drop tf/flax/onnx.
        siblings = [s.rfilename for s in getattr(info, "siblings", [])]
        has_fp16 = any(".fp16." in f for f in siblings)
        if has_fp16:
            allow_patterns = ["*.fp16.safetensors", "*.json", "*.txt", "*.md"]
        else:
            allow_patterns = ["*.safetensors", "*.json", "*.txt", "*.md"]
        # Older seq2seq translation repos still ship pytorch_model.bin
        # (no safetensors).
        allow_patterns.append("pytorch_model.bin")
        # Tokenizer artifacts vary by family: m2m100/nllb use a shared
        # sentencepiece.bpe.model, opus-mt ships per-direction
        # source.spm/target.spm + vocab.json, madlad (T5-shape) uses
        # spiece.model. Pull them all defensively.
        allow_patterns.extend([
            "tokenizer*", "spiece.model", "sentencepiece.bpe.model",
            "source.spm", "target.spm", "vocab.json", "merges.txt",
            "special_tokens_map.json", "added_tokens.json",
            "generation_config.json",
        ])
        return Path(snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    """Search HuggingFace for translation-tagged repos.

    Filter: translation task tag. Returns one row per matching repo.
    Mirrors text/summarization's search shape.
    """
    repos = api.list_models(
        search=query, filter="translation",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="text/translation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "text/translation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    # 110: tag-based, more specific than text-classification's catch-all
    # (200) but loses to file-pattern plugins (100). Same slot as
    # text/summarization, image_ocr, image_segmentation, audio_embedding.
    # See the module docstring's KNOWN LANDMINE note for the priority-tie
    # interaction with text/summarization's plugin.
    "priority": 110,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
