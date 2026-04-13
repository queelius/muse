"""Tests for pip and system-package helpers."""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from muse.core.install import (
    install_pip_extras,
    check_system_packages,
)


class TestInstallPipExtras:
    @patch("muse.core.install.subprocess.run")
    def test_installs_missing_packages(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("muse.core.install.importlib.util.find_spec", return_value=None):
            install_pip_extras(["diffusers"])
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pip" in args and "install" in args and "diffusers" in args

    @patch("muse.core.install.subprocess.run")
    def test_skips_already_installed(self, mock_run):
        with patch("muse.core.install.importlib.util.find_spec", return_value=MagicMock()):
            install_pip_extras(["numpy"])
        mock_run.assert_not_called()

    @patch("muse.core.install.subprocess.run")
    def test_raises_on_pip_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["pip"])
        with patch("muse.core.install.importlib.util.find_spec", return_value=None):
            with pytest.raises(subprocess.CalledProcessError):
                install_pip_extras(["bogus-pkg"])


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
