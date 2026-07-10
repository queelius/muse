"""Tests for pip and system-package helpers."""
from unittest.mock import patch

from muse.core.install import check_system_packages


def test_install_pip_extras_removed():
    """install_pip_extras was a dead footgun (never called; if called, it
    would install into sys.executable's env -- the supervisor env --
    defeating per-venv isolation). It must not exist as a public helper."""
    import muse.core.install as install_module
    assert not hasattr(install_module, "install_pip_extras")


class TestCheckSystemPackages:
    @patch("muse.core.install.shutil.which")
    def test_returns_missing(self, mock_which):
        mock_which.side_effect = lambda x: "/usr/bin/ffmpeg" if x == "ffmpeg" else None
        missing = check_system_packages(["ffmpeg", "espeak-ng"])
        assert missing == ["espeak-ng"]

    @patch("muse.core.install.shutil.which")
    def test_empty_when_all_present(self, mock_which):
        mock_which.return_value = "/usr/bin/cmd"
        assert check_system_packages(["ffmpeg"]) == []
