"""PROBE_DEFAULTS["call"] cleanup tests for 3d/generation."""
import os
from unittest.mock import MagicMock

import pytest

from muse.modalities.model_3d_generation import (
    Generation3DResult,
    PROBE_DEFAULTS,
)


def test_probe_default_shape_label():
    assert "call" in PROBE_DEFAULTS
    assert "shape" in PROBE_DEFAULTS
    assert "256" in PROBE_DEFAULTS["shape"]


def test_probe_unlinks_temp_png_on_success():
    """The probe should clean up the temp PNG after image_to_3d returns."""
    captured = {}

    def _image_to_3d(path):
        captured["path"] = path
        assert os.path.exists(path), "temp PNG should exist during call"
        return [Generation3DResult(glb_bytes=b"GLB", model_id="probe")]

    fake = MagicMock()
    fake.image_to_3d = _image_to_3d
    PROBE_DEFAULTS["call"](fake)
    assert "path" in captured
    assert not os.path.exists(captured["path"]), "temp PNG must be unlinked"


def test_probe_unlinks_temp_png_on_error():
    """Regression: backend.image_to_3d() can raise (OOM, transient
    model error). The temp PNG must still be unlinked, not leaked
    under /tmp.
    """
    captured = {}

    def _image_to_3d(path):
        captured["path"] = path
        raise RuntimeError("simulated backend failure")

    fake = MagicMock()
    fake.image_to_3d = _image_to_3d
    with pytest.raises(RuntimeError, match="simulated"):
        PROBE_DEFAULTS["call"](fake)
    assert "path" in captured
    assert not os.path.exists(captured["path"]), (
        "temp PNG must be unlinked even when image_to_3d raised"
    )
