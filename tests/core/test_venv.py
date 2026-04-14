"""Tests for venv creation + pip install helpers."""
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from muse.core.venv import (
    create_venv,
    install_into_venv,
    venv_python,
    find_free_port,
)


class TestVenvPython:
    def test_returns_bin_python_on_posix(self, tmp_path):
        # On POSIX venv layout, python is at <venv>/bin/python
        path = venv_python(tmp_path)
        assert path == tmp_path / "bin" / "python"


class TestCreateVenv:
    @patch("muse.core.venv.subprocess.run")
    def test_calls_python_venv_module(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        target = tmp_path / "myenv"
        create_venv(target)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        # Use sys.executable to guarantee we create the venv with the same
        # Python that muse is running on (matters for ABI compatibility)
        import sys
        assert args[0] == sys.executable
        assert "-m" in args and "venv" in args
        assert str(target) in args

    @patch("muse.core.venv.subprocess.run")
    def test_raises_on_venv_creation_failure(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, ["python", "-m", "venv"])
        with pytest.raises(subprocess.CalledProcessError):
            create_venv(tmp_path / "doomed")


class TestInstallIntoVenv:
    @patch("muse.core.venv.subprocess.run")
    def test_uses_venvs_pip_not_system_pip(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        # Simulate a venv layout
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        install_into_venv(tmp_path, ["numpy", "scipy"])
        args = mock_run.call_args[0][0]
        # Must be <venv>/bin/python -m pip install <pkgs>
        assert args[0] == str(tmp_path / "bin" / "python")
        assert args[1:4] == ["-m", "pip", "install"]
        assert "numpy" in args
        assert "scipy" in args

    @patch("muse.core.venv.subprocess.run")
    def test_empty_package_list_is_noop(self, mock_run, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        install_into_venv(tmp_path, [])
        mock_run.assert_not_called()

    @patch("muse.core.venv.subprocess.run")
    def test_raises_on_install_failure(self, mock_run, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        mock_run.side_effect = subprocess.CalledProcessError(1, ["pip"])
        with pytest.raises(subprocess.CalledProcessError):
            install_into_venv(tmp_path, ["bogus"])


class TestFindFreePort:
    def test_returns_an_int_in_range(self):
        p = find_free_port(start=9001, end=9999)
        assert 9001 <= p <= 9999

    def test_skips_bound_ports(self):
        import socket
        # Bind 9001 so find_free_port must skip it
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 9001))
            s.listen(1)
            p = find_free_port(start=9001, end=9003)
            assert p != 9001
        finally:
            s.close()

    def test_raises_when_no_free_port_in_range(self):
        import socket
        sockets = []
        try:
            # Bind every port in a tiny range
            for port in (19001, 19002, 19003):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                s.listen(1)
                sockets.append(s)
            with pytest.raises(RuntimeError, match="no free port"):
                find_free_port(start=19001, end=19003)
        finally:
            for s in sockets:
                s.close()
