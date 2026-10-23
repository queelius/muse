"""Tests for image request option helpers."""
import pytest

from muse.modalities.image_generation.options import (
    OptionError, check_background, check_output, ignored_fields,
    parse_size, resolve_quality, snap_size, snaps,
)

caps = {
    "size_multiple": 16, "max_pixels": 4194304, "default_size": [1024, 1024],
    "quality_presets": {
        "low": {"steps": 4, "max_pixels": 1048576},
        "medium": {"steps": 4}, "high": {"steps": 6},
    },
}


def test_parse_size_default_for_snapping_model():
    assert parse_size(None, caps) == (1024, 1024)


def test_parse_size_auto_matches_omitted():
    assert parse_size("auto", caps) == (1024, 1024)


def test_parse_size_default_for_legacy_model():
    assert parse_size(None, {"default_size": [1024, 1024]}) == (512, 512)


def test_parse_size_explicit():
    assert parse_size("640x480", caps) == (640, 480)


@pytest.mark.parametrize("bad", ["4096x64", "63x64", "abc", "512"])
def test_parse_size_rejects_bad(bad):
    with pytest.raises(OptionError) as e:
        parse_size(bad, caps)
    assert e.value.code == "invalid_parameter"
    assert str(e.value) == e.value.message


def test_snaps():
    assert snaps(caps) and not snaps({"default_size": [1, 1]})


def test_snap_keeps_fitting_size():
    assert snap_size(2048, 1152, caps, {}) == (2048, 1152)


def test_snap_rounds_down_to_multiple():
    assert snap_size(2000, 1125, caps, {}) == (2000, 1120)


def test_snap_low_preset_caps_pixels():
    assert snap_size(1536, 1024, caps, caps["quality_presets"]["low"]) == (1248, 832)


def test_snap_never_enlarges():
    assert snap_size(100, 70, caps, {}) == (96, 64)


def test_snap_odd_size_stays_valid():
    w, h = snap_size(1000, 1001, caps, caps["quality_presets"]["low"])
    assert w % 16 == 0 and h % 16 == 0 and w >= 64 and h >= 64 and w * h <= 1048576


def test_snap_noop_for_legacy_model():
    assert snap_size(1000, 1001, {}, {}) == (1000, 1001)


def test_quality_auto_is_medium():
    assert resolve_quality("auto", caps) == ("medium", {"steps": 4})
    assert resolve_quality(None, caps) == ("medium", {"steps": 4})


def test_quality_without_presets_passthrough():
    assert resolve_quality("high", {}) == ("high", {})
    assert resolve_quality(None, {}) == ("auto", {})


def test_quality_invalid():
    with pytest.raises(OptionError) as e:
        resolve_quality("ultra", caps)
    assert e.value.code == "invalid_parameter"


def test_background():
    assert check_background(None) == "auto"
    assert check_background("auto") == "auto"
    assert check_background("opaque") == "opaque"
    with pytest.raises(OptionError) as e:
        check_background("transparent")
    assert e.value.code == "background_not_supported"
    with pytest.raises(OptionError) as e:
        check_background("red")
    assert e.value.code == "invalid_parameter"


def test_output_check():
    assert check_output(None, None) == "png"
    assert check_output("webp", 50) == "webp"
    for args in (("gif", None), ("png", 101), ("png", -1)):
        with pytest.raises(OptionError) as e:
            check_output(*args)
        assert e.value.code == "invalid_parameter"


def test_ignored_fields():
    sent = {"moderation": "low", "negative_prompt": "x", "guidance": 3.0,
            "quality": "low", "output_format": "png", "output_compression": 80}
    assert ignored_fields(sent, {"supports_negative_prompt": False, "supports_guidance": False}) == \
        ["guidance", "moderation", "negative_prompt", "output_compression", "quality"]
    assert ignored_fields({"moderation": None}, caps) == []
    assert ignored_fields({"output_format": "webp", "output_compression": 80, "quality": "low"}, caps) == []
