import io
import struct
import wave
from pathlib import Path

import pytest

from muse.mcp.files import (
    allowed_prefixes, as_data_url, media_facts, preview_png, read_input,
    write_output,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", raising=False)
    monkeypatch.delenv("MUSE_MCP_OUTPUT_DIR", raising=False)


def test_reads_file_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.png").write_bytes(b"abc")
    data, name, mime = read_input("a.png", field="image", max_bytes=100)
    assert (data, name, mime) == (b"abc", "a.png", "image/png")


def test_empty_setting_means_cwd_and_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = allowed_prefixes()
    assert str(tmp_path.resolve()) in got
    assert str((tmp_path / "muse-output").resolve()) in got


def test_explicit_prefixes(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(tmp_path))
    assert allowed_prefixes() == [str(tmp_path.resolve())]


def test_symlink_escape_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "link").symlink_to("/etc/hostname")
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        read_input("link", field="file", max_bytes=10**6)


def test_dotdot_escape_refused(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"x")
    monkeypatch.chdir(sub)
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        read_input("../secret.txt", field="file", max_bytes=100)


def test_size_cap_and_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big").write_bytes(b"x" * 50)
    with pytest.raises(ValueError, match="exceeds"):
        read_input("big", field="file", max_bytes=10)
    with pytest.raises(ValueError, match="not found"):
        read_input("nope", field="file", max_bytes=10)


def test_as_data_url():
    assert as_data_url(b"hi", "text/plain") == "data:text/plain;base64,aGk="


def test_write_output_content_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = write_output(b"RIFF....", "audio/wav", out_path=None)
    assert w.path.endswith(".wav") and Path(w.path).read_bytes() == b"RIFF...."
    assert Path(w.path).parent == (tmp_path / "muse-output").resolve()
    assert w.bytes == 8 and len(w.sha256) == 64 and w.mime == "audio/wav"
    assert Path(w.path).name.startswith(w.sha256[:16])
    assert [p.name for p in Path(w.path).parent.iterdir()] == [Path(w.path).name]


def test_write_output_out_path_creates_parents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    w = write_output(b"x", "image/png", out_path="deep/er/x.png")
    assert Path(w.path).read_bytes() == b"x"


def test_out_path_outside_prefixes_refused(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        write_output(b"x", "image/png", out_path=str(tmp_path / "out.png"))


def _png(w, h):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, "PNG")
    return buf.getvalue()


def test_preview_long_side_512():
    pytest.importorskip("PIL")
    from PIL import Image
    out = preview_png(_png(1024, 256))
    assert Image.open(io.BytesIO(out)).size == (512, 128)
    assert preview_png(b"not an image") is None


def test_media_facts_image():
    pytest.importorskip("PIL")
    assert media_facts(_png(30, 20), "image/png") == {"width": 30, "height": 20}


def test_wav_duration():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<h", 0) * 4000)
    assert media_facts(buf.getvalue(), "audio/wav")["duration_seconds"] == 0.5
    assert media_facts(b"junk", "audio/wav") == {}


def test_parent_swapped_to_symlink_before_replace_refused(tmp_path, monkeypatch):
    import tempfile as _tf
    from muse.mcp import files
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    monkeypatch.chdir(work)
    real_mkstemp = _tf.mkstemp

    def swapping(*a, **kw):
        res = real_mkstemp(*a, **kw)
        parent = work / "muse-output"
        parent.rename(work / "moved")
        parent.symlink_to(outside)
        return res

    monkeypatch.setattr(files.tempfile, "mkstemp", swapping)
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        write_output(b"payload", "image/png", out_path=None)
    assert list(outside.iterdir()) == []


def test_fifo_refused_without_hanging(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(ValueError, match="regular file"):
        read_input("pipe", field="file", max_bytes=10)


# ---- cwd `/` (Claude Desktop launches stdio servers there) ----

def test_cwd_root_drops_implicit_cwd_prefix(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir("/")
    assert allowed_prefixes() == [str((home / "muse-output").resolve())]
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        read_input("/etc/hostname", field="image", max_bytes=10_000)


def test_cwd_root_default_output_goes_under_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir("/")
    wf = write_output(b"\x89PNG\r\n\x1a\nxx", "image/png", out_path=None)
    assert Path(wf.path).parent == (home / "muse-output").resolve()
    assert Path(wf.path).read_bytes().startswith(b"\x89PNG")


def test_cwd_root_explicit_output_dir_kept(tmp_path, monkeypatch):
    monkeypatch.chdir("/")
    monkeypatch.setenv("MUSE_MCP_OUTPUT_DIR", str(tmp_path / "out"))
    assert allowed_prefixes() == [str((tmp_path / "out").resolve())]
    wf = write_output(b"x", "image/png", out_path=None)
    assert Path(wf.path).parent == (tmp_path / "out").resolve()


# ---- configured prefixes replace the read defaults; output dir stays writable ----

def test_configured_prefixes_still_write_default_output(tmp_path, monkeypatch):
    work, other = tmp_path / "work", tmp_path / "other"
    work.mkdir()
    other.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(other))
    # Replaces the defaults for reads: the cwd is no longer readable.
    (work / "a.png").write_bytes(b"abc")
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        read_input("a.png", field="image", max_bytes=100)
    # ...but mcp.output_dir is always writable.
    wf = write_output(b"x", "image/png", out_path=None)
    assert Path(wf.path).parent == (work / "muse-output").resolve()
    named = write_output(b"y", "image/png", out_path="muse-output/named.png")
    assert Path(named.path).read_bytes() == b"y"
    # Paths outside both the list and the output dir are still refused.
    with pytest.raises(ValueError, match="not within an allowed prefix"):
        write_output(b"z", "image/png", out_path=str(tmp_path / "x.png"))
