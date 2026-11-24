"""Unit tests for the deterministic CPU ``SpriteMatte`` runtime."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from muse.modalities.image_matting import (
    PROBE_DEFAULTS,
    ImageMattingModel,
    MattingOptions,
)
from muse.modalities.image_matting.runtimes.sprite_matte import SpriteMatte


def _box_on_bg(
    size: int = 48,
    bg: tuple[int, int, int] = (255, 255, 255),
    fg: tuple[int, int, int] = (30, 110, 200),
    margin: int = 12,
) -> Image.Image:
    arr = np.empty((size, size, 3), dtype=np.uint8)
    arr[...] = bg
    arr[margin : size - margin, margin : size - margin] = fg
    return Image.fromarray(arr)


def test_sprite_matte_satisfies_protocol():
    m = SpriteMatte()
    assert isinstance(m, ImageMattingModel)
    assert m.model_id == "sprite-matte"
    assert m.device == "cpu"


def test_white_background_auto_and_explicit():
    m = SpriteMatte()
    img = _box_on_bg(size=40, bg=(252, 252, 252), fg=(40, 120, 60), margin=10)

    for bg_mode in ("auto", "white"):
        res = m.matte(img, MattingOptions(background=bg_mode, defringe=False))
        assert res.image.mode == "RGBA"
        assert (res.width, res.height) == (40, 40)
        rgba = np.asarray(res.image)
        # Exterior 10px margin is fully transparent; interior 20x20 box is opaque
        assert (rgba[:10, :, 3] == 0).all()
        assert (rgba[10:30, 10:30, 3] == 255).all()
        # 20*20 / (40*40) = 0.25 opaque, 0.75 transparent
        assert res.opaque_fraction == pytest.approx(0.25)
        assert res.transparent_fraction == pytest.approx(0.75)


def test_auto_colored_backdrop_keying():
    m = SpriteMatte()
    # Magenta backdrop with slight noise within color_tolerance
    rng = np.random.default_rng(1)
    arr = np.empty((48, 48, 3), dtype=np.int16)
    arr[...] = (255, 0, 255)
    arr += rng.integers(-6, 7, size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr[12:36, 12:36] = (40, 140, 60)
    img = Image.fromarray(arr)

    res = m.matte(img, MattingOptions(background="auto", defringe=False))
    rgba = np.asarray(res.image)
    assert (rgba[:12, :, 3] == 0).all()
    assert (rgba[12:36, 12:36, 3] == 255).all()
    assert res.metadata["background_rgb"] is not None


def test_corners_mode_handles_one_occluded_corner_and_edge_pocket():
    m = SpriteMatte()
    # 3 corners are green screen (0, 220, 0), top-left corner has foreground touching
    arr = np.empty((48, 48, 3), dtype=np.uint8)
    arr[...] = (0, 220, 0)
    arr[10:38, 10:38] = (180, 60, 40)
    arr[0:6, 0:6] = (180, 60, 40)  # occludes 1 corner only
    img = Image.fromarray(arr)

    res = m.matte(img, MattingOptions(background="corners", defringe=False))
    rgba = np.asarray(res.image)
    # Green background connected to the other 3 corners is cleared
    assert rgba[0, 47, 3] == 0
    assert rgba[47, 0, 3] == 0
    assert rgba[47, 47, 3] == 0
    # Foreground box and top-left foreground remain opaque
    assert (rgba[10:38, 10:38, 3] == 255).all()
    assert rgba[0, 0, 3] == 255


def test_clear_enclosed_white_removes_arch_pocket_keeps_specular_glint():
    m = SpriteMatte()
    arr = np.full((64, 64, 3), 255, dtype=np.uint8)
    # Ring-shaped dark foreground
    arr[10:54, 10:54] = (50, 70, 90)
    # 12x12 enclosed white pocket (e.g. gap between roots / limbs)
    arr[26:38, 26:38] = (252, 252, 252)
    # 1px white eye glint inside the solid band (must survive)
    arr[15, 15] = (255, 255, 255)
    img = Image.fromarray(arr)

    # Off by default: enclosed pocket stays opaque
    res_off = m.matte(
        img, MattingOptions(background="white", clear_enclosed_white=False, defringe=False)
    )
    rgba_off = np.asarray(res_off.image)
    assert (rgba_off[26:38, 26:38, 3] == 255).all()
    assert res_off.metadata["enclosed_pockets_cleared"] == 0

    # Enabled: 12x12 enclosed pocket cleared, 1px glint preserved
    res_on = m.matte(
        img, MattingOptions(background="white", clear_enclosed_white=True, defringe=False)
    )
    rgba_on = np.asarray(res_on.image)
    assert (rgba_on[26:38, 26:38, 3] == 0).all()
    assert rgba_on[15, 15, 3] == 255
    assert res_on.metadata["enclosed_pockets_cleared"] == 1


def test_defringe_cleans_1px_white_halo_and_decontaminates_rgb():
    m = SpriteMatte()
    arr = np.full((48, 48, 3), 255, dtype=np.uint8)
    # 1px anti-aliased light halo (224, 224, 224) around dark green core
    arr[12:36, 12:36] = (224, 224, 224)
    arr[13:35, 13:35] = (30, 90, 40)
    img = Image.fromarray(arr)

    no_defringe = m.matte(img, MattingOptions(background="white", defringe=False))
    with_defringe = m.matte(
        img, MattingOptions(background="white", defringe=True, fringe_threshold=212)
    )

    a_no = np.asarray(no_defringe.image)
    a_yes = np.asarray(with_defringe.image)

    # Without defringe, the 224 halo stays fully opaque and bright gray
    assert a_no[12, 20, 3] == 255
    assert tuple(a_no[12, 20, :3]) == (224, 224, 224)

    # With defringe, alpha is attenuated and RGB is decontaminated toward the green core
    assert a_yes[12, 20, 3] < 255
    assert a_yes[12, 20, 1] > a_yes[12, 20, 0]  # green channel dominates
    assert with_defringe.metadata["defringed_pixels"] > 0
    # Interior core stays untouched
    assert (a_yes[13:35, 13:35, 3] == 255).all()


def test_hard_alpha_and_crop():
    m = SpriteMatte()
    arr = np.full((48, 48, 3), 255, dtype=np.uint8)
    arr[12:36, 12:36] = (226, 226, 226)  # halo below 128 alpha after defringe
    arr[13:35, 13:35] = (20, 60, 160)
    img = Image.fromarray(arr)

    res = m.matte(
        img,
        MattingOptions(
            background="white",
            defringe=True,
            fringe_threshold=212,
            hard_alpha=True,
            alpha_threshold=128,
            crop=True,
        ),
    )
    rgba = np.asarray(res.image)
    assert set(np.unique(rgba[..., 3]).tolist()) <= {0, 255}
    # Halo was attenuated below 128 and dropped, leaving tight 22x22 core
    assert (res.width, res.height) == (22, 22)
    assert res.opaque_fraction == pytest.approx(1.0)
    assert res.transparent_fraction == pytest.approx(0.0)
    assert res.metadata["cropped"] is True


def test_crop_noop_on_all_transparent_image():
    m = SpriteMatte()
    img = Image.new("RGB", (32, 24), (255, 255, 255))
    res = m.matte(img, MattingOptions(background="white", crop=True))
    assert (res.width, res.height) == (32, 24)
    assert res.transparent_fraction == pytest.approx(1.0)
    assert res.metadata["cropped"] is False


def test_already_transparent_border_preserved():
    m = SpriteMatte()
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[8:24, 8:24] = (200, 100, 50, 255)
    img = Image.fromarray(arr)

    res = m.matte(img, MattingOptions(background="auto", crop=True))
    assert (res.width, res.height) == (16, 16)
    assert res.opaque_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"background": "magenta"},
        {"bg_threshold": -1},
        {"bg_threshold": 256},
        {"color_tolerance": 300},
        {"alpha_threshold": -5},
        {"fringe_threshold": 999},
    ],
)
def test_invalid_options_raise_value_error(kwargs):
    m = SpriteMatte()
    with pytest.raises(ValueError):
        m.matte(_box_on_bg(16), MattingOptions(**kwargs))


def test_probe_defaults_runs_on_sprite_matte():
    m = SpriteMatte()
    info = PROBE_DEFAULTS["call"](m)
    assert info["width"] == 64 and info["height"] == 64
    assert 0.5 < info["transparent_fraction"] < 1.0
    assert 0.1 < info["opaque_fraction"] < 0.5
