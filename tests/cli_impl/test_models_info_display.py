"""Tests for muse.cli_impl.models_info_display."""
from __future__ import annotations

import pytest

from muse.cli_impl.models_info_display import (
    KNOWN_CAPABILITIES,
    _format_uptime,
    _render_capability_value,
    format_info,
)
from muse.core.catalog import CatalogEntry


def _entry(model_id="kokoro-82m", modality="audio/speech", **kwargs):
    return CatalogEntry(
        model_id=model_id,
        modality=modality,
        backend_path="muse.models.kokoro_82m:Model",
        hf_repo=kwargs.get("hf_repo", "hexgrad/Kokoro-82M"),
        description=kwargs.get("description", "Tiny TTS"),
        pip_extras=tuple(kwargs.get("pip_extras", ())),
        system_packages=tuple(kwargs.get("system_packages", ())),
        extra=kwargs.get("extra", {}),
    )


class TestFormatUptime:
    def test_seconds(self):
        assert _format_uptime(12) == "12s"

    def test_minutes_and_seconds(self):
        assert _format_uptime(3 * 60 + 5) == "3m 5s"

    def test_hours_and_minutes(self):
        assert _format_uptime(2 * 3600 + 15 * 60 + 7) == "2h 15m"

    def test_zero(self):
        assert _format_uptime(0) == "0s"


class TestRenderCapabilityValue:
    def test_known_image_generation_text_to_image(self):
        out = _render_capability_value("image/generation", "supports_text_to_image", True)
        assert out == ("text-to-image", "yes")

    def test_unknown_modality_returns_none(self):
        assert _render_capability_value("modality/none", "anything", 1) is None

    def test_unknown_key_returns_none(self):
        assert _render_capability_value("audio/speech", "future_flag", 1) is None

    def test_known_capabilities_table_covers_all_bundled_modalities(self):
        # Smoke check that the table includes the main modalities. Acts
        # as a regression sentinel; growing the table is a deliberate
        # action when adding a new modality.
        for k in [
            "image/generation", "audio/speech", "audio/transcription",
            "chat/completion", "embedding/text",
        ]:
            assert k in KNOWN_CAPABILITIES


class TestFormatInfo:
    def test_offline_unpulled_shows_not_pulled_status(self):
        catalog_known = {"kokoro-82m": _entry()}
        out = format_info(
            "kokoro-82m",
            catalog_known=catalog_known,
            catalog_data={},
            online_status=None,
        )
        assert "not pulled" in out
        assert "Worker status:" in out
        assert "not running" in out

    def test_offline_enabled_not_loaded_shows_disabled_section(self):
        catalog_known = {"kokoro-82m": _entry()}
        out = format_info(
            "kokoro-82m",
            catalog_known=catalog_known,
            catalog_data={"enabled": True, "venv_path": "/v"},
            online_status=None,
        )
        assert "enabled, not loaded" in out
        assert "not running" in out

    def test_offline_disabled(self):
        catalog_known = {"kokoro-82m": _entry()}
        out = format_info(
            "kokoro-82m",
            catalog_known=catalog_known,
            catalog_data={"enabled": False},
            online_status=None,
        )
        assert "disabled" in out

    def test_online_loaded_shows_worker_section(self):
        catalog_known = {"kokoro-82m": _entry()}
        out = format_info(
            "kokoro-82m",
            catalog_known=catalog_known,
            catalog_data={"enabled": True, "venv_path": "/v"},
            online_status={
                "loaded": True,
                "worker_port": 9001,
                "worker_pid": 4711,
                "worker_uptime_seconds": 60 * 60 + 30 * 60 + 5,
                "worker_status": "running",
                "restart_count": 0,
                "last_error": None,
            },
        )
        assert "loaded on worker port 9001" in out
        assert "pid:" in out
        assert "4711" in out
        assert "1h 30m" in out

    def test_known_capability_keys_render_with_label(self):
        catalog_known = {"sd-turbo": _entry(
            model_id="sd-turbo",
            modality="image/generation",
            extra={
                "supports_text_to_image": True,
                "supports_img2img": False,
                "default_size": "512x512",
            },
        )}
        out = format_info(
            "sd-turbo",
            catalog_known=catalog_known,
            catalog_data={"enabled": True},
            online_status=None,
        )
        assert "text-to-image:" in out
        assert "img2img:" in out
        assert "default size:" in out

    def test_unknown_capability_keys_roll_up_to_other(self):
        catalog_known = {"sd-turbo": _entry(
            model_id="sd-turbo",
            modality="image/generation",
            extra={
                "supports_text_to_image": True,
                "secret_flag": "x",  # unknown
                "weird_thing": 42,    # unknown
            },
        )}
        out = format_info(
            "sd-turbo",
            catalog_known=catalog_known,
            catalog_data={"enabled": True},
            online_status=None,
        )
        assert "(other capabilities:" in out
        assert "secret_flag" in out
        assert "weird_thing" in out
        # The known one still gets its proper label
        assert "text-to-image:" in out

    def test_memory_section_with_annotation_and_measurement(self):
        catalog_known = {"k": _entry(
            model_id="k", modality="audio/speech",
            extra={"memory_gb": 2.5},
        )}
        out = format_info(
            "k",
            catalog_known=catalog_known,
            catalog_data={
                "enabled": True,
                "measurements": {
                    "cuda": {
                        "weights_bytes": 2 * 1024**3,
                        "peak_bytes": 3 * 1024**3,
                        "ran_inference": True,
                        "shape": "1x1024",
                        "probed_at": "2026-04-28T00:00:00",
                    },
                },
            },
            online_status=None,
        )
        assert "annotated peak:" in out
        assert "2.5 GB" in out
        assert "measured (cuda):" in out
        assert "weights 2.00 GB" in out
        assert "peak 3.00 GB at 1x1024" in out

    def test_unknown_model_returns_error(self):
        out = format_info(
            "ghost",
            catalog_known={},
            catalog_data={},
            online_status=None,
        )
        assert out.startswith("error:")
        assert "ghost" in out
