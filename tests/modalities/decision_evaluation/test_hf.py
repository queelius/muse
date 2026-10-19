"""Exact-repo HF plugin for decision/evaluation."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import muse.modalities.decision_evaluation.hf as plugin_mod
from muse.core.discovery import (
    REQUIRED_HF_PLUGIN_KEYS,
    _default_hf_plugin_dirs,
    discover_hf_plugins,
)
from muse.core.resolvers import ResolvedModel, ResolverError
from muse.core.resolvers_hf import HFResolver
from muse.modalities.decision_evaluation.hf import HF_PLUGIN, SOURCES


DECISION = "evalengine/decision-0.8b-gguf"
DECIMA = "amyrmahdy/decima-base"
DECISION_SHA = "ce993792ad283053004bf0230b1709ff7351bb24"
DECIMA_SHA = "e5d84ab3a8436cedc45b0daefa91979f68319455"
DECIMA_HEAD = "cf9dcd1037935450d1ea34c986feae9946ba61f7"

# Sibling lists and tags as the Hub reports them at the pinned commits.
_DECISION_SIBLINGS = [
    ".gitattributes", "LICENSE", "NOTICE", "README.md", "SOURCES.json",
    "SOURCE_REVIEW.md", "benchmark.png", "decision-0.8b-F16.gguf",
    "decision-0.8b-Q8_0.gguf", "evaluation/EXPORT_VERIFICATION.md",
    "licenses/QWEN-LICENSE", "licenses/TEV-LICENSE", "publication-manifest.json",
]
_DECISION_TAGS = [
    "gguf", "llama.cpp", "decision-model", "jev", "on-device", "text-classification",
    "en", "license:apache-2.0", "conversational",
]
_DECIMA_SIBLINGS = [
    ".gitattributes", "LICENSE", "README.md", "config.json", "decima-cli.gif",
    "onnx/fp32/encoder.onnx", "onnx/int8/decima.json", "onnx/int8/encoder.onnx",
    "onnx/int8/quantize.txt", "onnx/int8/scorer.onnx",
    "onnx/int8/tokenizer/tokenizer.json", "onnx/int8/tokenizer/tokenizer_config.json",
    "pytorch/model.safetensors",
]
_DECIMA_TAGS = [
    "onnx", "safetensors", "decima", "decision-model", "system-one", "jev",
    "zero-shot-classification", "en", "license:apache-2.0",
]


def _info(repo_id, sha, siblings, tags):
    return SimpleNamespace(
        id=repo_id, sha=sha, tags=list(tags),
        siblings=[SimpleNamespace(rfilename=f) for f in siblings],
        card_data=SimpleNamespace(license="apache-2.0"),
    )


def decision_info(**kw):
    return _info(kw.get("id", DECISION), kw.get("sha", DECISION_SHA),
                 kw.get("siblings", _DECISION_SIBLINGS), kw.get("tags", _DECISION_TAGS))


def decima_info(**kw):
    return _info(kw.get("id", DECIMA), kw.get("sha", DECIMA_SHA),
                 kw.get("siblings", _DECIMA_SIBLINGS), kw.get("tags", _DECIMA_TAGS))


# --- metadata and import hygiene ------------------------------------------


def test_plugin_has_required_keys_and_runs_before_every_other_plugin():
    assert set(REQUIRED_HF_PLUGIN_KEYS) <= set(HF_PLUGIN)
    assert HF_PLUGIN["modality"] == "decision/evaluation"
    others = [p for p in discover_hf_plugins(_default_hf_plugin_dirs())
              if p["modality"] != "decision/evaluation"]
    assert HF_PLUGIN["priority"] < min(p["priority"] for p in others)


def test_plugin_imports_only_stdlib_hub_and_core():
    tree = ast.parse(Path(plugin_mod.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports in single-file plugins"
            roots.add(node.module)
    for name in roots:
        assert (
            name in {"__future__", "hashlib", "re", "pathlib", "typing", "huggingface_hub"}
            or name.startswith("muse.core.")
        ), name


def test_discovered_as_a_single_file_plugin():
    found = [p for p in discover_hf_plugins(_default_hf_plugin_dirs())
             if p["modality"] == "decision/evaluation"]
    assert len(found) == 1 and found[0]["priority"] == HF_PLUGIN["priority"]


# --- sniff ----------------------------------------------------------------


@pytest.mark.parametrize("repo", [DECISION, DECIMA, DECISION.upper(), "AmyrMahdy/Decima-Base"])
def test_sniff_claims_the_exact_repos_case_insensitively(repo):
    assert HF_PLUGIN["sniff"](SimpleNamespace(id=repo, tags=[], siblings=[]))


@pytest.mark.parametrize("repo, tags, siblings", [
    ("evalengine/decision-4b-gguf", _DECISION_TAGS, _DECISION_SIBLINGS),
    ("amyrmahdy/decima-small", _DECIMA_TAGS, _DECIMA_SIBLINGS),
    ("amyrmahdy/decima-agent", _DECIMA_TAGS, _DECIMA_SIBLINGS),
    ("mallory/decima-base", _DECIMA_TAGS, _DECIMA_SIBLINGS),          # copied layout
    ("someone/judge", ["decision-model", "jev"], ["config.json"]),    # just the tags
    ("evalengine/decision-0.8b", ["decision-model"], ["adapter_model.safetensors"]),
])
def test_sniff_refuses_everything_else(repo, tags, siblings):
    assert not HF_PLUGIN["sniff"](_info(repo, DECISION_SHA, siblings, tags))


# --- resolve --------------------------------------------------------------


def test_decision_resolves_to_pinned_manifest():
    r = HF_PLUGIN["resolve"](DECISION, None, decision_info())
    assert isinstance(r, ResolvedModel)
    m = r.manifest
    assert m["model_id"] == "decision-0.8b-q8"
    assert m["modality"] == "decision/evaluation"
    assert m["hf_repo"] == DECISION and m["revision"] == DECISION_SHA
    assert m["pip_extras"] == ["llama-cpp-python==0.3.33"]
    assert r.backend_path == (
        "muse.modalities.decision_evaluation.runtimes.llama_cpp:DecisionLlamaCppModel"
    )
    caps = m["capabilities"]
    assert caps["gguf_file"] == "decision-0.8b-Q8_0.gguf"
    assert caps["max_concurrency"] == 1
    assert caps["context_length"] == 2048
    assert caps["calibrated"] is False
    assert caps["question_types"] == ["choice", "score", "noul"]


def test_decima_resolves_to_pinned_cpu_manifest():
    r = HF_PLUGIN["resolve"](DECIMA, None, decima_info())
    m = r.manifest
    assert m["model_id"] == "decima-base"
    assert m["revision"] == DECIMA_SHA
    assert m["pip_extras"] == ["decima-ai==2.2.0", "onnxruntime==1.24.4"]
    assert r.backend_path == (
        "muse.modalities.decision_evaluation.runtimes.decima_onnx:DecimaOnnxModel"
    )
    caps = m["capabilities"]
    assert caps["device"] == "cpu"
    assert caps["temperature"] == 0.7269418400253115
    assert caps["max_concurrency"] == 1
    assert caps["max_state_tokens"] == 512 and caps["max_option_tokens"] == 64
    assert "LICENSING.md" in m["description"]


def test_capabilities_are_copies_not_the_shared_table():
    r = HF_PLUGIN["resolve"](DECIMA, None, decima_info())
    r.manifest["capabilities"]["device"] = "cuda"
    assert SOURCES[DECIMA]["capabilities"]["device"] == "cpu"


@pytest.mark.parametrize("info", [
    decima_info(sha=DECIMA_HEAD),       # the repo's current head, not the pin
    decima_info(sha=None),
    decision_info(sha="0" * 40),
])
def test_only_the_pinned_revision_resolves(info):
    with pytest.raises(ResolverError) as exc:
        HF_PLUGIN["resolve"](info.id, None, info)
    assert SOURCES[info.id]["model_id"] in str(exc.value)  # points at the curated id


def test_variants_are_refused():
    with pytest.raises(ResolverError):
        HF_PLUGIN["resolve"](DECISION, "q4_k_m", decision_info())


def test_missing_pinned_file_on_the_hub_is_refused():
    siblings = [s for s in _DECIMA_SIBLINGS if not s.endswith("scorer.onnx")]
    with pytest.raises(ResolverError, match="scorer.onnx"):
        HF_PLUGIN["resolve"](DECIMA, None, decima_info(siblings=siblings))


def test_unsupported_repo_is_refused_by_resolve():
    with pytest.raises(ResolverError):
        HF_PLUGIN["resolve"]("someone/else", None, decima_info(id="someone/else"))


# --- download allow-lists and verification --------------------------------


@pytest.mark.parametrize("repo, info", [(DECISION, decision_info()), (DECIMA, decima_info())])
def test_download_uses_exact_allow_list_at_the_pinned_revision(repo, info, tmp_path):
    r = HF_PLUGIN["resolve"](repo, None, info)
    with patch.object(plugin_mod, "snapshot_download", return_value=str(tmp_path)) as snap, \
         patch.object(plugin_mod, "verify_files") as verify:
        assert r.download(tmp_path / "cache") == tmp_path
    kw = snap.call_args.kwargs
    assert kw["repo_id"] == repo and kw["revision"] == SOURCES[repo]["revision"]
    assert sorted(kw["allow_patterns"]) == sorted(SOURCES[repo]["files"])
    verify.assert_called_once_with(tmp_path, SOURCES[repo]["files"])
    assert r.artifact_provenance[0]["allow_patterns"] == kw["allow_patterns"]


def test_allow_lists_match_the_specified_subsets():
    assert set(SOURCES[DECISION]["files"]) == {
        "decision-0.8b-Q8_0.gguf", "LICENSE", "NOTICE",
        "licenses/QWEN-LICENSE", "licenses/TEV-LICENSE",
    }
    assert set(SOURCES[DECIMA]["files"]) == {
        "config.json", "LICENSE", "onnx/int8/decima.json", "onnx/int8/encoder.onnx",
        "onnx/int8/scorer.onnx", "onnx/int8/tokenizer/tokenizer.json",
        "onnx/int8/tokenizer/tokenizer_config.json",
    }
    assert SOURCES[DECISION]["files"]["decision-0.8b-Q8_0.gguf"][0] == 811_843_168


def test_allow_lists_select_only_pinned_files_with_hub_semantics():
    from huggingface_hub.utils import filter_repo_objects

    for repo, siblings in ((DECISION, _DECISION_SIBLINGS), (DECIMA, _DECIMA_SIBLINGS)):
        kept = set(filter_repo_objects(siblings, allow_patterns=list(SOURCES[repo]["files"])))
        assert kept == set(SOURCES[repo]["files"]) & set(siblings)
    assert not any("fp32" in f or "pytorch" in f or "F16" in f
                   for repo in SOURCES for f in SOURCES[repo]["files"])


def _write(root: Path, name: str, data: bytes) -> tuple[int, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def test_verify_files_accepts_exact_matches(tmp_path):
    a = _write(tmp_path, "a.bin", b"weights")
    b = _write(tmp_path, "sub/b.txt", b"text")
    plugin_mod.verify_files(tmp_path, {"a.bin": a, "sub/b.txt": (b[0], None)})


@pytest.mark.parametrize("files, error", [
    ({"missing.bin": (1, None)}, "missing"),
    ({"a.bin": (8, None)}, "bytes"),
    ({"a.bin": (7, "0" * 64)}, "SHA-256"),
])
def test_verify_files_rejects_mismatches(files, error, tmp_path):
    _write(tmp_path, "a.bin", b"weights")
    with pytest.raises(ResolverError, match=error):
        plugin_mod.verify_files(tmp_path, files)


# --- real dispatch across every discovered plugin -------------------------


def _resolver(info):
    r = HFResolver(plugins=discover_hf_plugins(_default_hf_plugin_dirs()))
    r._api = MagicMock()
    r._api.repo_info.return_value = info
    return r


def test_decima_dispatches_to_decision_not_nli_classifier():
    """Regression: Decima's card tags it zero-shot-classification, so the
    text/classification plugin used to claim it as an NLI pipeline."""
    resolved = _resolver(decima_info()).resolve(f"hf://{DECIMA}", revision=DECIMA_SHA)
    assert resolved.manifest["modality"] == "decision/evaluation"
    assert resolved.manifest["model_id"] == "decima-base"


def test_without_the_decision_plugin_decima_was_misdispatched():
    plugins = [p for p in discover_hf_plugins(_default_hf_plugin_dirs())
               if p["modality"] != "decision/evaluation"]
    r = HFResolver(plugins=plugins)
    r._api = MagicMock()
    r._api.repo_info.return_value = decima_info()
    assert r.resolve(f"hf://{DECIMA}").manifest["modality"] == "text/classification"


def test_decision_gguf_dispatches_to_decision_not_chat():
    resolved = _resolver(decision_info()).resolve(f"hf://{DECISION}")
    assert resolved.manifest["modality"] == "decision/evaluation"
    assert resolved.manifest["revision"] == DECISION_SHA


def test_bare_uri_at_a_moved_head_fails_instead_of_falling_through():
    with pytest.raises(ResolverError, match="decima-base"):
        _resolver(decima_info(sha=DECIMA_HEAD)).resolve(f"hf://{DECIMA}")


def test_other_repos_keep_their_dispatch():
    other = _info("cross-encoder/nli-deberta-v3-small", DECISION_SHA,
                  ["config.json", "model.safetensors", "tokenizer.json"],
                  ["zero-shot-classification", "transformers"])
    assert _resolver(other).resolve(
        "hf://cross-encoder/nli-deberta-v3-small",
    ).manifest["modality"] == "text/classification"


# --- search ---------------------------------------------------------------


def test_search_yields_both_sources_for_matching_queries():
    rows = list(HF_PLUGIN["search"](MagicMock(), "decision", sort="downloads", limit=20))
    assert {r.model_id for r in rows} == {"decision-0.8b-q8", "decima-base"}
    assert all(r.modality == "decision/evaluation" for r in rows)
    assert all(r.uri.startswith("hf://") for r in rows)


def test_search_ignores_unrelated_queries_and_never_calls_the_hub():
    api = MagicMock()
    assert list(HF_PLUGIN["search"](api, "qwen coder", sort="downloads", limit=20)) == []
    assert list(HF_PLUGIN["search"](api, "decima", sort="downloads", limit=0)) == []
    assert api.method_calls == []
