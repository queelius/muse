"""PROBE_DEFAULTS for image/generation: generate, then edit if supported."""
from unittest.mock import MagicMock

from muse.modalities.image_generation import PROBE_DEFAULTS


def test_probe_runs_edit_when_supported():
    m = MagicMock()
    m.supports_reference_images = True
    out = PROBE_DEFAULTS["call"](m)
    m.generate.assert_called_once_with("probe scene")
    m.edit.assert_called_once()
    args, kwargs = m.edit.call_args
    assert args == ("the same scene at night",)
    assert kwargs["images"] == [m.generate.return_value.image]
    assert out is m.generate.return_value


def test_probe_skips_edit_otherwise():
    m = MagicMock(spec=["generate"])
    out = PROBE_DEFAULTS["call"](m)
    m.generate.assert_called_once_with("probe scene")
    assert out is m.generate.return_value
