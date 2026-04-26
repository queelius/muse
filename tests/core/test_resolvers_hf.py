"""Tests for HFResolver (huggingface_hub mocked; no network)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from muse.core.resolvers import ResolverError, _reset_registry_for_tests


@pytest.fixture(autouse=True)
def _clean_registry():
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


def _fake_repo_info(siblings=(), tags=()):
    """Build a MagicMock that looks like HfApi().repo_info() output."""
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f, size=1_000_000) for f in siblings]
    info.tags = list(tags)
    info.card_data = MagicMock(license="apache-2.0")
    info.downloads = 123
    return info


def test_sniff_recognizes_sentence_transformers_via_tag():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["config.json", "tokenizer.json"],
        tags=["sentence-transformers"],
    )
    assert _sniff_repo_shape(info) == "sentence-transformers"


def test_sniff_recognizes_sentence_transformers_via_config_file():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["config.json", "sentence_transformers_config.json"],
        tags=[],
    )
    assert _sniff_repo_shape(info) == "sentence-transformers"


def test_sniff_returns_unknown_for_unrecognized_repo():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = _fake_repo_info(
        siblings=["model.safetensors", "config.json"],
        tags=["some-unrelated-tag"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_resolve_gguf_requires_variant():
    """GGUF repos MUST specify @variant; no magic default."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["a-q4_k_m.gguf", "a-q5_k_m.gguf", "a-q8_0.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="variant.*required.*available"):
            r.resolve("hf://org/repo-gguf")


def test_resolve_gguf_variant_not_found_lists_available():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["a-q4_k_m.gguf", "a-q5_k_m.gguf"],
            tags=["text-generation"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="variant.*q8_0.*not found"):
            r.resolve("hf://org/repo-gguf@q8_0")


def test_resolve_sentence_transformer_repo():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["config.json", "sentence_transformers_config.json"],
            tags=["sentence-transformers"],
        )
        r = HFResolver()
        rm = r.resolve("hf://sentence-transformers/all-MiniLM-L6-v2")
        assert rm.manifest["modality"] == "embedding/text"
        assert rm.manifest["hf_repo"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert "sentence-transformers" in " ".join(rm.manifest["pip_extras"])
        assert rm.backend_path.endswith(":SentenceTransformerModel")


def test_resolve_rejects_non_hf_scheme():
    from muse.core.resolvers_hf import HFResolver
    r = HFResolver()
    with pytest.raises(ResolverError):
        r.resolve("civitai://something")


def test_resolve_rejects_non_uri():
    from muse.core.resolvers_hf import HFResolver
    r = HFResolver()
    with pytest.raises(ResolverError):
        r.resolve("not-a-uri")


def test_resolve_unrecognized_repo_shape_raises():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        MockApi.return_value.repo_info.return_value = _fake_repo_info(
            siblings=["model.safetensors"],
            tags=["some-unsupported-tag"],
        )
        r = HFResolver()
        with pytest.raises(ResolverError, match="no HF plugin matched"):
            r.resolve("hf://org/weird-repo")


def test_search_gguf_returns_variant_rows():
    """Each GGUF file in a matched repo becomes a separate SearchResult."""
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        fake_repo = MagicMock(
            id="org/Qwen3-8B-GGUF",
            downloads=1000,
            tags=["text-generation"],
            siblings=[
                MagicMock(rfilename="x-q4_k_m.gguf", size=4_500_000_000),
                MagicMock(rfilename="x-q8_0.gguf", size=8_500_000_000),
                MagicMock(rfilename="README.md", size=10_000),
            ],
        )
        api.list_models.return_value = [fake_repo]
        r = HFResolver()
        results = list(r.search("qwen3", modality="chat/completion"))
        assert len(results) == 2
        uris = {res.uri for res in results}
        assert "hf://org/Qwen3-8B-GGUF@q4_k_m" in uris
        assert "hf://org/Qwen3-8B-GGUF@q8_0" in uris


def test_search_gguf_dedupes_variants_per_repo():
    """Sharded GGUFs (model-q4_k_m-00001-of-00003.gguf) and repos that
    publish the same quant in multiple files emit ONE row per (repo, variant)
    with sizes summed across files. Without this dedup, search output is
    flooded with duplicates (the bug v0.10.2 fixes)."""
    from muse.core.resolvers_hf import HFResolver
    # list_models returns a repo without sibling info; resolver falls back
    # to repo_info(files_metadata=True) to fetch siblings + sizes.
    list_repo = MagicMock(id="unsloth/Qwen3-122B-GGUF", downloads=500_000, tags=[])
    list_repo.siblings = []  # force the repo_info fallback
    info = MagicMock()
    info.siblings = [
        # Three shards of one bf16 quant
        MagicMock(rfilename="m-bf16-00001-of-00003.gguf", size=80_000_000_000),
        MagicMock(rfilename="m-bf16-00002-of-00003.gguf", size=80_000_000_000),
        MagicMock(rfilename="m-bf16-00003-of-00003.gguf", size=80_000_000_000),
        # Also a single-file q4_k_m
        MagicMock(rfilename="m-q4_k_m.gguf", size=12_000_000_000),
    ]
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [list_repo]
        api.repo_info.return_value = info
        r = HFResolver()
        results = list(r.search("qwen3", modality="chat/completion"))

    uris = [res.uri for res in results]
    # Exactly one row per variant
    assert uris.count("hf://unsloth/Qwen3-122B-GGUF@bf16") == 1
    assert uris.count("hf://unsloth/Qwen3-122B-GGUF@q4_k_m") == 1
    assert len(results) == 2
    # Sharded bf16's size is the sum of all three shards (240 GB)
    bf16 = next(r for r in results if r.uri.endswith("@bf16"))
    assert abs(bf16.size_gb - 240.0) < 0.001
    # Single-file q4_k_m is 12 GB
    q4 = next(r for r in results if r.uri.endswith("@q4_k_m"))
    assert abs(q4.size_gb - 12.0) < 0.001


def test_search_gguf_passes_files_metadata_when_repo_info_called():
    """Without files_metadata=True, RepoSibling.size is None and our
    --max-size-gb filter is meaningless. v0.10.2 fix: always request it."""
    from muse.core.resolvers_hf import HFResolver
    list_repo = MagicMock(id="org/repo-gguf", downloads=1, tags=[])
    list_repo.siblings = []  # force fallback
    info = MagicMock()
    info.siblings = [MagicMock(rfilename="model-q4_k_m.gguf", size=4_000_000_000)]
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [list_repo]
        api.repo_info.return_value = info
        list(HFResolver().search("anything", modality="chat/completion"))
        # The fallback repo_info call must include files_metadata=True
        api.repo_info.assert_called_once()
        kwargs = api.repo_info.call_args.kwargs
        assert kwargs.get("files_metadata") is True


def test_search_embeddings_returns_repo_rows():
    from muse.core.resolvers_hf import HFResolver
    with patch("muse.core.resolvers_hf.HfApi") as MockApi:
        api = MockApi.return_value
        api.list_models.return_value = [
            MagicMock(
                id="sentence-transformers/all-MiniLM-L6-v2",
                downloads=50_000_000,
                tags=["sentence-transformers", "feature-extraction"],
                siblings=[MagicMock(rfilename="config.json", size=1000)],
            ),
        ]
        r = HFResolver()
        results = list(r.search("minilm", modality="embedding/text"))
        assert len(results) == 1
        assert results[0].uri == "hf://sentence-transformers/all-MiniLM-L6-v2"
        assert results[0].modality == "embedding/text"


def test_hf_resolver_registers_on_import():
    """Importing muse.core.resolvers_hf should register an HFResolver."""
    import importlib
    from muse.core import resolvers_hf  # noqa: F401
    importlib.reload(resolvers_hf)  # _clean_registry fixture cleared the prior registration
    from muse.core.resolvers import get_resolver
    r = get_resolver("hf://anything/anywhere")
    assert r.scheme == "hf"


# --- faster-whisper branch ---

def _fake_ct2_whisper_siblings():
    return [
        SimpleNamespace(rfilename="model.bin"),
        SimpleNamespace(rfilename="config.json"),
        SimpleNamespace(rfilename="vocabulary.txt"),
        SimpleNamespace(rfilename="README.md"),
    ]


def test_sniff_detects_faster_whisper_shape():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["automatic-speech-recognition", "whisper"],
    )
    assert _sniff_repo_shape(info) == "faster-whisper"


def test_sniff_rejects_ct2_shape_without_asr_tag():
    """CT2 alone is not enough: could be an NMT repo."""
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["machine-translation"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_sniff_rejects_without_model_bin():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["automatic-speech-recognition"],
    )
    assert _sniff_repo_shape(info) == "unknown"


def test_resolve_faster_whisper_synthesizes_manifest():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=_fake_ct2_whisper_siblings(),
        tags=["automatic-speech-recognition"],
        card_data=SimpleNamespace(license="mit"),
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://Systran/faster-whisper-tiny")
    assert resolved.manifest["modality"] == "audio/transcription"
    assert resolved.manifest["hf_repo"] == "Systran/faster-whisper-tiny"
    assert resolved.manifest["model_id"] == "faster-whisper-tiny"
    assert "faster-whisper>=1.0.0" in resolved.manifest["pip_extras"]
    assert "ffmpeg" in resolved.manifest["system_packages"]
    assert resolved.backend_path == (
        "muse.modalities.audio_transcription.runtimes.faster_whisper"
        ":FasterWhisperModel"
    )


def test_search_faster_whisper_yields_results():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    fake_repos = [
        SimpleNamespace(id="Systran/faster-whisper-tiny", downloads=12345, siblings=[]),
        SimpleNamespace(id="Systran/faster-whisper-base", downloads=8000, siblings=[]),
    ]
    with patch.object(resolver._api, "list_models", return_value=fake_repos):
        results = list(resolver.search("whisper", modality="audio/transcription"))
    assert len(results) == 2
    assert all(r.modality == "audio/transcription" for r in results)
    assert all(r.uri.startswith("hf://Systran/faster-whisper-") for r in results)
    assert results[0].model_id == "faster-whisper-tiny"
    assert results[1].model_id == "faster-whisper-base"


# --- text-classification branch ---

def test_sniff_detects_text_classification_tag():
    from muse.core.resolvers_hf import _sniff_repo_shape
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["text-classification", "transformers"],
    )
    assert _sniff_repo_shape(info) == "text-classification"


def test_resolve_text_classification_synthesizes_manifest():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=[
            SimpleNamespace(rfilename="config.json"),
            SimpleNamespace(rfilename="model.safetensors"),
            SimpleNamespace(rfilename="tokenizer.json"),
        ],
        tags=["text-classification"],
        card_data=SimpleNamespace(license="apache-2.0"),
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        resolved = resolver.resolve("hf://KoalaAI/Text-Moderation")
    assert resolved.manifest["modality"] == "text/classification"
    assert resolved.manifest["hf_repo"] == "KoalaAI/Text-Moderation"
    assert resolved.manifest["model_id"] == "text-moderation"
    assert "transformers>=4.36.0" in resolved.manifest["pip_extras"]
    assert "torch>=2.1.0" in resolved.manifest["pip_extras"]
    assert resolved.backend_path == (
        "muse.modalities.text_classification.runtimes.hf_text_classifier"
        ":HFTextClassifier"
    )


def test_search_text_classification_yields_results():
    from muse.core.resolvers_hf import HFResolver
    resolver = HFResolver()
    fake_repos = [
        SimpleNamespace(id="KoalaAI/Text-Moderation", downloads=5000, siblings=[]),
        SimpleNamespace(id="unitary/toxic-bert", downloads=12000, siblings=[]),
    ]
    with patch.object(resolver._api, "list_models", return_value=fake_repos):
        results = list(resolver.search("toxic", modality="text/classification"))
    assert len(results) == 2
    assert all(r.modality == "text/classification" for r in results)
    assert results[0].model_id == "text-moderation"


def test_resolve_unknown_error_message_includes_repo_diagnostics():
    """When no plugin matches and the legacy fallback also misses, the error
    surfaces the repo id plus the seen tags and siblings so users can debug."""
    from muse.core.resolvers_hf import HFResolver, ResolverError
    resolver = HFResolver()
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename="random.bin")],
        tags=["something-unknown"],
    )
    with patch.object(resolver._api, "repo_info", return_value=info):
        try:
            resolver.resolve("hf://x/y")
        except ResolverError as e:
            msg = str(e)
            assert "no HF plugin matched" in msg
            assert "x/y" in msg
            assert "something-unknown" in msg
            assert "random.bin" in msg
        else:
            raise AssertionError("expected ResolverError")
