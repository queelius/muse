"""Seeding Muse's weights cache from the shared Hugging Face cache."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from muse.core.hf_seed import drop_unused_seeds, repo_folder, seed_repo

REPO = "org/model"
REV = "a" * 40


def _lfs(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _shared(tmp_path: Path, files: dict[str, tuple[bytes, str]]) -> Path:
    """files: {repo path: (content, blob name)} in one snapshot."""
    root = tmp_path / "hub" / repo_folder(REPO)
    (root / "blobs").mkdir(parents=True)
    snap = root / "snapshots" / REV
    for rel, (data, name) in files.items():
        (root / "blobs" / name).write_bytes(data)
        link = snap / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(root / "blobs" / name, link.parent))
    return tmp_path / "hub"


def _ino(p: Path) -> int:
    return os.stat(p).st_ino


def test_verified_blobs_are_hardlinked_into_the_muse_cache(tmp_path):
    weights = b"W" * 1000
    shared = _shared(tmp_path, {
        "model.safetensors": (weights, _lfs(weights)),
        "config.json": (b"{}", _git(b"{}")),
    })
    muse = tmp_path / "weights"
    seeded = seed_repo(muse, REPO, shared=shared)
    assert len(seeded) == 2
    for blob in seeded:
        assert _ino(blob) == _ino(shared / repo_folder(REPO) / "blobs" / blob.name)


def test_a_blob_whose_content_does_not_match_its_name_is_skipped(tmp_path):
    shared = _shared(tmp_path, {"model.safetensors": (b"corrupt", _lfs(b"original"))})
    assert seed_repo(tmp_path / "weights", REPO, shared=shared) == []


def test_allow_patterns_limit_what_is_seeded(tmp_path):
    a, b = b"A" * 10, b"B" * 10
    shared = _shared(tmp_path, {
        "unet/model.fp16.safetensors": (a, _lfs(a)),
        "unet/model.safetensors": (b, _lfs(b)),
    })
    seeded = seed_repo(tmp_path / "weights", REPO, shared=shared, allow_patterns=["*.fp16.*"])
    assert [p.name for p in seeded] == [_lfs(a)]


def test_existing_blobs_and_missing_shared_repos_are_left_alone(tmp_path):
    data = b"W" * 10
    shared = _shared(tmp_path, {"m.bin": (data, _lfs(data))})
    muse = tmp_path / "weights"
    (muse / repo_folder(REPO) / "blobs").mkdir(parents=True)
    (muse / repo_folder(REPO) / "blobs" / _lfs(data)).write_bytes(data)
    assert seed_repo(muse, REPO, shared=shared) == []
    assert seed_repo(muse, "other/repo", shared=shared) == []
    assert seed_repo(muse, REPO, shared=tmp_path / "absent") == []


def test_unused_seeds_are_dropped_after_the_download(tmp_path):
    used, unused = b"U" * 10, b"X" * 10
    shared = _shared(tmp_path, {"a.bin": (used, _lfs(used)), "b.bin": (unused, _lfs(unused))})
    muse = tmp_path / "weights"
    seeded = seed_repo(muse, REPO, shared=shared)
    # The download (huggingface_hub) links only the file this revision needs.
    snap = muse / repo_folder(REPO) / "snapshots" / REV
    snap.mkdir(parents=True)
    (snap / "a.bin").symlink_to(f"../../blobs/{_lfs(used)}")
    drop_unused_seeds(seeded)
    blobs = {p.name for p in (muse / repo_folder(REPO) / "blobs").iterdir()}
    assert blobs == {_lfs(used)}
    assert (shared / repo_folder(REPO) / "blobs" / _lfs(unused)).exists()  # shared copy kept


def test_seeded_blobs_lose_group_and_world_write(tmp_path):
    data = b"W" * 10
    shared = _shared(tmp_path, {"m.bin": (data, _lfs(data))})
    os.chmod(shared / repo_folder(REPO) / "blobs" / _lfs(data), 0o666)
    [blob] = seed_repo(tmp_path / "weights", REPO, shared=shared)
    assert os.stat(blob).st_mode & 0o777 == 0o644


def test_blobs_owned_by_another_user_are_not_seeded(tmp_path, monkeypatch):
    data = b"W" * 10
    shared = _shared(tmp_path, {"m.bin": (data, _lfs(data))})
    monkeypatch.setattr(os, "getuid", lambda: os.stat(shared).st_uid + 1)
    muse = tmp_path / "weights"
    assert seed_repo(muse, REPO, shared=shared) == []
    assert not list((muse / repo_folder(REPO) / "blobs").iterdir())


def test_the_linked_file_is_what_gets_verified(tmp_path, monkeypatch):
    import muse.core.hf_seed as hf_seed

    data = b"W" * 10
    shared = _shared(tmp_path, {"m.bin": (data, _lfs(data))})
    checked = []
    real = hf_seed._verified
    monkeypatch.setattr(hf_seed, "_verified", lambda p, n: checked.append(p) or real(p, n))
    seed_repo(tmp_path / "weights", REPO, shared=shared)
    assert checked and all(".muse-seed-" in p.name for p in checked)


def test_folder_patterns_and_revisions_follow_huggingface_hub(tmp_path):
    a, b = b"A" * 10, b"B" * 10
    shared = _shared(tmp_path, {"unet/w.safetensors": (a, _lfs(a))})
    repo = shared / repo_folder(REPO)
    (repo / "blobs" / _lfs(b)).write_bytes(b)
    other = repo / "snapshots" / ("b" * 40)
    other.mkdir()
    (other / "x.bin").symlink_to(os.path.relpath(repo / "blobs" / _lfs(b), other))
    seeded = seed_repo(tmp_path / "w1", REPO, shared=shared, allow_patterns=["unet/"])
    assert [p.name for p in seeded] == [_lfs(a)]
    seeded = seed_repo(tmp_path / "w2", REPO, shared=shared, revision=REV)
    assert [p.name for p in seeded] == [_lfs(a)]
    seeded = seed_repo(tmp_path / "w3", REPO, shared=shared, revision="c" * 40)
    assert sorted(p.name for p in seeded) == sorted([_lfs(a), _lfs(b)])
