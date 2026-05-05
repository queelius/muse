"""HFAudioClassifier: mocked-dep tests.

Module-level sentinels (torch, AutoModelForAudioClassification,
AutoFeatureExtractor, librosa) get patched per-test; the autouse
fixture restores them on teardown.
"""
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_module_sentinels():
    """Reset module-top sentinels between tests (deferred-imports pattern)."""
    import muse.modalities.audio_classification.runtimes.hf_audio_classifier as mod
    orig = (
        mod.torch, mod.AutoModelForAudioClassification,
        mod.AutoFeatureExtractor, mod.librosa,
    )
    yield
    (mod.torch, mod.AutoModelForAudioClassification,
     mod.AutoFeatureExtractor, mod.librosa) = orig


def _wire_basic_runtime(mod, *, problem_type=None, id2label=None):
    """Install fake torch + transformers + librosa.

    `problem_type` simulates `model.config.problem_type`. None => the
    checkpoint is missing the field (mirrors AST AudioSet, which has
    multi-label semantics but no problem_type in config.json).
    """
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends = MagicMock(mps=None)
    mod.torch = fake_torch

    extractor = MagicMock()
    extractor.sampling_rate = 16000
    extractor_factory = MagicMock()
    extractor_factory.from_pretrained = MagicMock(return_value=extractor)
    mod.AutoFeatureExtractor = extractor_factory

    cfg = MagicMock()
    cfg.problem_type = problem_type
    cfg.id2label = id2label or {0: "speech", 1: "music"}

    model_obj = MagicMock()
    model_obj.config = cfg
    model_obj.to = MagicMock(return_value=model_obj)

    model_factory = MagicMock()
    model_factory.from_pretrained = MagicMock(return_value=model_obj)
    mod.AutoModelForAudioClassification = model_factory

    mod.librosa = MagicMock()

    return extractor, model_obj


def test_default_multi_label_from_config():
    """When config declares multi_label_classification, runtime honors it."""
    import muse.modalities.audio_classification.runtimes.hf_audio_classifier as mod
    _wire_basic_runtime(mod, problem_type="multi_label_classification")
    runtime = mod.HFAudioClassifier(
        model_id="m", hf_repo="x", device="cpu",
    )
    assert runtime._multi_label is True


def test_default_single_label_from_config_absent():
    """No problem_type and no hint => single-label (softmax)."""
    import muse.modalities.audio_classification.runtimes.hf_audio_classifier as mod
    _wire_basic_runtime(mod, problem_type=None)
    runtime = mod.HFAudioClassifier(
        model_id="m", hf_repo="x", device="cpu",
    )
    assert runtime._multi_label is False


def test_manifest_hint_flips_to_multi_label():
    """Regression: AST AudioSet manifest claims multi_label=True but its
    config lacks problem_type. The hint must promote the flag."""
    import muse.modalities.audio_classification.runtimes.hf_audio_classifier as mod
    _wire_basic_runtime(mod, problem_type=None)
    runtime = mod.HFAudioClassifier(
        model_id="ast", hf_repo="x", device="cpu", multi_label=True,
    )
    assert runtime._multi_label is True


def test_manifest_hint_false_does_not_clobber_config():
    """A False hint cannot override a config-declared True. ORing means
    hints are additive: they can flip False->True but not True->False.

    Rationale: the model card is authoritative on its own architecture;
    a manifest hint is a fallback for checkpoints with stale configs.
    """
    import muse.modalities.audio_classification.runtimes.hf_audio_classifier as mod
    _wire_basic_runtime(mod, problem_type="multi_label_classification")
    runtime = mod.HFAudioClassifier(
        model_id="m", hf_repo="x", device="cpu", multi_label=False,
    )
    assert runtime._multi_label is True
