"""Prebuilt CUDA llama-cpp-python wheels: install plan and runtime preload."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from muse.core import llama_cuda
from muse.core.venv import pip_install_commands

CU130, VERIFIED, RUNTIMES = llama_cuda.WHEELS["cu130"]


@pytest.mark.parametrize("req, expected", [
    ("llama-cpp-python==0.3.33", True),
    ("llama_cpp_python>=0.2.90", True),
    ("Llama-CPP-Python", True),
    ("llama-cpp-python[server]", True),
    ("llama-cpp-python-extra", False),
    ("numpy", False),
])
def test_recognizes_llama_cpp_requirements(req, expected):
    assert llama_cuda.is_llama_cpp_requirement(req) is expected


def test_plan_is_unchanged_without_the_setting_or_without_llama():
    assert llama_cuda.install_plan(["a", "llama-cpp-python==0.3.33"], None) == [
        ["a", "llama-cpp-python==0.3.33"],
    ]
    assert llama_cuda.install_plan(["a"], "cu130") == [["a"]]
    assert llama_cuda.install_plan([], "cu130") == []


def test_plan_installs_llama_binary_only_from_the_wheel_index():
    plan = llama_cuda.install_plan(["a", "llama-cpp-python==0.3.33"], "cu130")
    assert plan == [
        ["a", *RUNTIMES],
        ["--no-cache-dir", "--only-binary=llama-cpp-python", "--extra-index-url", CU130,
         f"llama-cpp-python==0.3.33,<={VERIFIED}"],
    ]


@pytest.mark.parametrize("req, capped", [
    ("llama-cpp-python", "llama-cpp-python<=V"),
    ("llama-cpp-python>=0.2.90", "llama-cpp-python>=0.2.90,<=V"),
    ("llama_cpp_python[server] >= 0.3", "llama_cpp_python[server]>= 0.3,<=V"),
    ('llama-cpp-python>=0.3; sys_platform == "linux"', 'llama-cpp-python>=0.3,<=V; sys_platform == "linux"'),
    ("llama-cpp-python @ https://example.invalid/x.whl", "llama-cpp-python @ https://example.invalid/x.whl"),
])
def test_cap_joins_the_requirement_so_pip_never_fetches_newer_wheels(req, capped):
    assert llama_cuda._capped(req, "V") == capped


def test_unknown_wheel_is_refused():
    with pytest.raises(ValueError):
        llama_cuda.install_plan(["llama-cpp-python"], "rocm")


def _commands(monkeypatch, packages, *, cuda_build, upgrade=False):
    monkeypatch.setenv("MUSE_LLAMA_CPP_WHEEL", "cu130")
    with patch("muse.core.venv._is_cuda_llama_build", return_value=cuda_build):
        return list(pip_install_commands("/v/bin/python", packages, upgrade=upgrade))


def test_commands_replace_a_cpu_build_without_touching_deps(monkeypatch):
    cmds = _commands(monkeypatch, ["llama-cpp-python==0.3.33"], cuda_build=False, upgrade=True)
    base = ["/v/bin/python", "-m", "pip", "install", "--upgrade"]
    wheel = ["--no-cache-dir", "--only-binary=llama-cpp-python", "--extra-index-url", CU130,
             f"llama-cpp-python==0.3.33,<={VERIFIED}"]
    assert cmds == [
        [*base, *RUNTIMES],
        [*base, *wheel],
        [*base, "--force-reinstall", "--no-deps", *wheel],
    ]


def test_install_checks_the_build_only_after_the_install_steps_ran(monkeypatch, tmp_path):
    from muse.core import venv as venv_module

    monkeypatch.setenv("MUSE_LLAMA_CPP_WHEEL", "cu130")
    events: list[str] = []
    monkeypatch.setattr(
        venv_module, "run_owned_command",
        lambda cmd, **kwargs: events.append(
            "reinstall" if "--force-reinstall" in cmd else "install"
        ),
    )

    def is_cuda_build(python):
        events.append("check")
        return False

    monkeypatch.setattr(venv_module, "_is_cuda_llama_build", is_cuda_build)
    venv_module.install_into_venv(tmp_path, ["llama-cpp-python==0.3.33"])
    assert events == ["install", "install", "check", "reinstall"]


def test_commands_stop_when_the_cuda_build_is_in_place(monkeypatch):
    cmds = _commands(monkeypatch, ["llama-cpp-python==0.3.33"], cuda_build=True)
    assert len(cmds) == 2


def test_commands_without_the_setting_are_one_pip_call(monkeypatch):
    monkeypatch.delenv("MUSE_LLAMA_CPP_WHEEL", raising=False)
    with patch("muse.core.config.get", return_value=None):
        cmds = list(pip_install_commands("/py", ["llama-cpp-python"], quiet=True))
    assert cmds == [["/py", "-m", "pip", "install", "-q", "llama-cpp-python"]]


def test_setting_accepts_only_known_wheels():
    from muse.core import config

    setting = config.SETTINGS_BY_KEY["install.llama_cpp_wheel"]
    assert setting.env == "MUSE_LLAMA_CPP_WHEEL" and setting.default is None
    assert set(setting.choices) == set(llama_cuda.WHEELS)


# --- runtime preload ---------------------------------------------------------


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A fake site-packages with a llama_cpp package and NVIDIA's pip libs."""
    package = tmp_path / "llama_cpp"
    (package / "lib").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    nvidia = tmp_path / "nvidia" / "cu13" / "lib"
    nvidia.mkdir(parents=True)
    for name in ("libcublas.so.13", "libcublasLt.so.13", "libcudart.so.13", "libcudart_static.a"):
        (nvidia / name).write_bytes(b"")
    spec = MagicMock(origin=str(package / "__init__.py"))
    monkeypatch.setattr(llama_cuda.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(llama_cuda, "_prepared", None)
    monkeypatch.delitem(llama_cuda.sys.modules, "llama_cpp", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return tmp_path


def test_cpu_build_loads_nothing(site):
    with patch.object(llama_cuda.ctypes, "CDLL") as cdll:
        assert llama_cuda.prepare_llama_cpp(cpu_only=True) == []
    cdll.assert_not_called()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_cuda_build_preloads_runtime_before_cublas(site):
    (site / "llama_cpp" / "lib" / "libggml-cuda.so").write_bytes(b"")
    with patch.object(llama_cuda.ctypes, "CDLL") as cdll:
        loaded = llama_cuda.prepare_llama_cpp()
        assert llama_cuda.prepare_llama_cpp() == loaded  # once per process
    names = [os.path.basename(p) for p in loaded]
    assert names == ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"]
    assert cdll.call_count == 3
    assert all(c.kwargs["mode"] == llama_cuda.ctypes.RTLD_GLOBAL for c in cdll.call_args_list)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_cpu_only_hides_cuda_devices_before_import(site):
    (site / "llama_cpp" / "lib" / "libggml-cuda.so").write_bytes(b"")
    with patch.object(llama_cuda.ctypes, "CDLL"):
        llama_cuda.prepare_llama_cpp(cpu_only=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_cpu_only_after_import_cannot_hide_devices(site, monkeypatch):
    (site / "llama_cpp" / "lib" / "libggml-cuda.so").write_bytes(b"")
    monkeypatch.setitem(llama_cuda.sys.modules, "llama_cpp", object())
    with patch.object(llama_cuda.ctypes, "CDLL"):
        llama_cuda.prepare_llama_cpp(cpu_only=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_missing_package_is_a_cpu_build(monkeypatch):
    monkeypatch.setattr(llama_cuda.importlib.util, "find_spec", lambda name: None)
    assert llama_cuda.is_cuda_build() is False


def test_capability_query_preloads_and_tracks_the_libraries():
    from muse.core.placement import _QUERY

    assert _QUERY.index("prepare_llama_cpp()") < _QUERY.index("import llama_cpp\n")
    assert "*preloaded" in _QUERY
