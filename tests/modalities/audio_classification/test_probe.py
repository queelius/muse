"""PROBE_DEFAULTS["call"] cleanup tests."""
import os
from unittest.mock import MagicMock

import pytest

from muse.modalities.audio_classification import (
    AudioClassificationResult,
    PROBE_DEFAULTS,
)


def test_probe_unlinks_temp_wav_on_success():
    """The probe should clean up the temp WAV after classify returns."""
    captured = {}

    def _classify(path):
        captured["path"] = path
        assert os.path.exists(path), "temp WAV should exist during call"
        return [AudioClassificationResult(scores={"x": 1.0}, multi_label=False)]

    fake = MagicMock()
    fake.classify = _classify
    PROBE_DEFAULTS["call"](fake)
    assert "path" in captured
    assert not os.path.exists(captured["path"]), "temp WAV must be unlinked"


def test_probe_unlinks_temp_wav_on_error():
    """Regression: backend.classify() can raise (OOM, transient model
    error). The temp WAV must still be unlinked, not leaked under /tmp.
    """
    captured = {}

    def _classify(path):
        captured["path"] = path
        raise RuntimeError("simulated backend failure")

    fake = MagicMock()
    fake.classify = _classify
    with pytest.raises(RuntimeError, match="simulated"):
        PROBE_DEFAULTS["call"](fake)
    assert "path" in captured
    assert not os.path.exists(captured["path"]), (
        "temp WAV must be unlinked even when classify raised"
    )
