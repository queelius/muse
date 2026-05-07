"""Bundled smolvlm-256m-instruct script: manifest fields + alias preservation."""


def test_manifest_fields():
    from muse.models.smolvlm_256m_instruct import MANIFEST
    assert MANIFEST["model_id"] == "smolvlm-256m-instruct"
    assert MANIFEST["modality"] == "chat/completion"
    assert MANIFEST["hf_repo"] == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert MANIFEST["license"] == "apache-2.0"
    assert "transformers>=4.46.0" in MANIFEST["pip_extras"]
    caps = MANIFEST["capabilities"]
    assert caps["supports_vision"] is True
    assert caps["supports_multi_image"] is True
    assert caps["supports_tools"] is False
    assert caps["device"] == "cpu"
    assert caps["memory_gb"] == 1.0


def test_model_alias_is_hfvisionlanguagemodel():
    """The script exposes Model = HFVisionLanguageModel so discovery
    picks up the runtime via the bundled-script surface."""
    from muse.models.smolvlm_256m_instruct import Model
    from muse.modalities.chat_completion.runtimes.transformers_vlm import (
        HFVisionLanguageModel,
    )
    assert Model is HFVisionLanguageModel


def test_model_id_in_manifest_matches_filename():
    """Catalog convention: bundled script filename mirrors model_id with
    underscores."""
    from muse.models import smolvlm_256m_instruct
    assert smolvlm_256m_instruct.MANIFEST["model_id"] == "smolvlm-256m-instruct"
