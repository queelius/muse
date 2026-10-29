"""PROBE_DEFAULTS for image/generation: generate, then edit if supported."""
from unittest.mock import MagicMock

from muse.modalities.image_generation import PROBE_DEFAULTS


def test_probe_runs_edit_when_supported():
    m = MagicMock()
    m.supports_reference_images = True
    m.generate.return_value.image = _scene(256)
    m.edit.return_value.image = _scene(256)
    out = PROBE_DEFAULTS["call"](m)
    m.generate.assert_called_once_with("probe scene")
    m.edit.assert_called_once()
    args, kwargs = m.edit.call_args
    assert args == ("the same scene at night",)
    assert kwargs["images"] == [m.generate.return_value.image]
    assert out is m.generate.return_value


def test_probe_skips_edit_otherwise():
    m = MagicMock(spec=["generate"])
    m.generate.return_value.image = _scene(256)
    out = PROBE_DEFAULTS["call"](m)
    m.generate.assert_called_once_with("probe scene")
    assert out is m.generate.return_value


# ---------------- degenerate-output check ----------------

import numpy as np
import pytest
from PIL import Image

from muse.modalities.image_generation.probe_check import (
    AUTOCORR_FLOOR,
    DegenerateImageError,
    STD_FLOOR,
    check_not_degenerate,
    image_stats,
)


def _noise(size, mean=128.0, std=50.0, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(mean, std, (size, size, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _scene(size=512):
    """A smooth synthetic 'image': gradients plus a few solid shapes."""
    y, x = np.mgrid[0:size, 0:size] / size
    r = 255 * x
    g = 255 * y
    b = 128 + 100 * np.sin(6 * x) * np.cos(4 * y)
    arr = np.stack([r, g, b], -1)
    arr[size // 4: size // 2, size // 4: size // 2] = (240, 30, 30)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _sprite(size):
    """Pixel-art-like: flat background, a blocky figure on an 8 px grid."""
    rng = np.random.default_rng(3)
    cells = size // 8
    grid = rng.integers(0, 4, (cells, cells))
    grid[: cells // 4] = 0
    grid[:, : cells // 4] = 0
    palette = np.array([(30, 30, 40), (200, 60, 60), (240, 220, 120), (60, 140, 200)])
    arr = palette[grid].repeat(8, 0).repeat(8, 1).astype(np.uint8)
    return Image.fromarray(arr)


def test_thresholds_are_the_documented_values():
    assert (STD_FLOOR, AUTOCORR_FLOOR) == (2.0, 0.5)


@pytest.mark.parametrize("img", [
    _noise(512),                                       # gaussian noise
    _noise(1024, mean=20, std=15),                     # near-black noise
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (512, 512, 3), dtype=np.uint8)),
    Image.new("RGB", (512, 512), (128, 128, 128)),     # flat grey
    Image.new("RGB", (1024, 1024), (0, 0, 0)),         # flat black
    _noise(512, mean=128, std=1.0),                    # flat with sensor-level jitter
], ids=["gaussian", "near-black", "uniform", "grey", "black", "jitter"])
def test_degenerate_images_flagged(img):
    with pytest.raises(DegenerateImageError) as e:
        check_not_degenerate(img)
    assert str(e.value).startswith("probe output looks degenerate: std=")
    assert "autocorr=" in str(e.value)


@pytest.mark.parametrize("img", [
    _scene(512), _scene(1024), _scene(200), _sprite(512), _sprite(256), _sprite(64),
    _scene(512).convert("RGBA"), _scene(512).convert("L"),
], ids=["scene512", "scene1024", "scene200", "sprite512", "sprite256",
        "sprite64", "rgba", "gray"])
def test_real_looking_images_pass(img):
    check_not_degenerate(img)
    std, ac = image_stats(img)
    assert std >= 2.0
    assert ac is None or ac > AUTOCORR_FLOOR


def test_stats_on_noise_and_scene():
    std, ac = image_stats(_noise(512))
    assert std > 2.0 and abs(ac) < 0.1
    std, ac = image_stats(_scene(512))
    assert ac > 0.95


def test_large_images_are_downsampled_to_256(monkeypatch):
    # Same answer at 2048 as at 512 for noise: still ~0 after a box filter.
    std, ac = image_stats(_noise(2048))
    assert abs(ac) < 0.1


def test_autocorr_skipped_under_128px():
    # 16 px pixel art reads below 0.5 (calibration: 0.44-0.48), so tiny
    # images get the flatness check only. Nearest upsampling would hide
    # noise instead (k-fold nearest noise has autocorr (k-1)/k).
    std, ac = image_stats(_noise(64))
    assert ac is None and std > 2.0
    check_not_degenerate(_noise(64))
    with pytest.raises(DegenerateImageError, match="autocorr=n/a"):
        check_not_degenerate(Image.new("RGB", (64, 64), (9, 9, 9)))
    assert image_stats(_noise(128))[1] is not None


def test_wide_image_keeps_the_noise_check():
    # 1024x256 downsamples to 256x64; the skip is decided on the source
    # short side (256 px), so noise is still caught.
    rng = np.random.default_rng(5)
    wide = Image.fromarray(rng.integers(0, 256, (256, 1024, 3), dtype=np.uint8))
    std, ac = image_stats(wide)
    assert ac is not None and abs(ac) < 0.1
    with pytest.raises(DegenerateImageError):
        check_not_degenerate(wide)
    check_not_degenerate(_scene(1024).resize((1024, 256)))


def test_numpy_and_float_inputs_are_normalized():
    arr = np.asarray(_scene(256), dtype=np.float32) / 255.0
    check_not_degenerate(arr)
    with pytest.raises(DegenerateImageError):
        check_not_degenerate(np.zeros((256, 256, 3), dtype=np.uint8))


class _Result:
    def __init__(self, image):
        self.image = image


def test_probe_raises_on_degenerate_generate():
    m = MagicMock(spec=["generate"])
    m.generate.return_value = _Result(_noise(512))
    with pytest.raises(DegenerateImageError, match="probe output looks degenerate"):
        PROBE_DEFAULTS["call"](m)


def test_probe_raises_on_degenerate_edit():
    m = MagicMock()
    m.supports_reference_images = True
    m.generate.return_value = _Result(_scene(512))
    m.edit.return_value = _Result(Image.new("RGB", (512, 512)))
    with pytest.raises(DegenerateImageError, match="edit"):
        PROBE_DEFAULTS["call"](m)


def test_probe_passes_good_outputs():
    m = MagicMock()
    m.supports_reference_images = True
    m.generate.return_value = _Result(_scene(512))
    m.edit.return_value = _Result(_scene(512))
    assert PROBE_DEFAULTS["call"](m) is m.generate.return_value


def test_probe_failure_fails_the_probe_record(tmp_path, monkeypatch, capsys):
    """End to end over the probe seams: the in-venv worker runs the real
    image/generation PROBE_DEFAULTS on a backend that draws noise and
    records inference_error with the metric values; run_probe, fed that
    record through its run_owned_command seam, returns 6 (which makes
    muse pull disable the model)."""
    import json
    from unittest.mock import patch

    from muse.cli_impl.probe import run_probe
    from muse.cli_impl.probe_worker import run_probe_worker
    from muse.core import catalog as catalog_mod
    from muse.core.catalog import CatalogEntry, _reset_read_catalog_cache, _write_catalog

    entry = CatalogEntry(
        model_id="noisy-lora", modality="image/generation",
        backend_path="muse.modalities.image_generation.runtimes.reference:DiffusersReferenceModel",
        hf_repo="someone/noisy-lora", description="", pip_extras=(),
        system_packages=(), extra={"device": "cpu"},
    )
    backend = MagicMock(spec=["generate"])
    backend.generate.return_value = _Result(_noise(512))
    with patch.object(catalog_mod, "known_models", return_value={"noisy-lora": entry}), \
         patch.object(catalog_mod, "load_backend", return_value=backend):
        assert run_probe_worker(model_id="noisy-lora", device="cpu", run_inference=True) == 0
    out = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    record = json.loads(out[-1])
    assert record["inference_error"].startswith(
        "DegenerateImageError: probe output looks degenerate: std=")
    assert "autocorr=" in record["inference_error"]

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _write_catalog({"noisy-lora": {
        "pulled_at": "2026-10-28T00:00:00+00:00", "hf_repo": "someone/noisy-lora",
        "local_dir": str(tmp_path / "w"), "venv_path": str(tmp_path / "v"),
        "python_path": str(tmp_path / "v" / "bin" / "python"), "enabled": True,
    }})
    _reset_read_catalog_cache()
    done = MagicMock(returncode=0, stdout=json.dumps(record) + "\n", stderr="")
    with patch("muse.cli_impl.probe.run_owned_command", return_value=done), \
         patch("muse.cli_impl.probe.known_models", return_value={"noisy-lora": entry}):
        rc = run_probe(model_id="noisy-lora", no_inference=False, device=None, as_json=True)
    assert rc == 6
