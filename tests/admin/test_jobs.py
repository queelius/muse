"""Tests for the in-memory JobStore."""
from __future__ import annotations

import threading
import time

import pytest

from muse.admin.jobs import (
    Job,
    JobStore,
    get_default_store,
    reset_default_store,
)


class TestJobStore:
    def test_create_assigns_uuid_job_id_and_pending_state(self):
        store = JobStore()
        job = store.create(op="enable", model_id="soprano-80m")
        assert isinstance(job.job_id, str)
        assert len(job.job_id) == 32  # uuid4 hex
        assert job.op == "enable"
        assert job.model_id == "soprano-80m"
        assert job.state == "pending"
        assert job.started_at  # iso timestamp present
        assert job.finished_at is None

    def test_get_returns_job_by_id(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        fetched = store.get(job.job_id)
        assert fetched is job

    def test_get_returns_none_for_unknown_id(self):
        store = JobStore()
        assert store.get("nonexistent") is None

    def test_update_running_then_done_sets_finished_at(self):
        store = JobStore()
        job = store.create(op="enable", model_id="m")
        store.update(job.job_id, state="running")
        assert job.state == "running"
        assert job.finished_at is None
        store.update(job.job_id, state="done", result={"worker_port": 9001})
        assert job.state == "done"
        assert job.finished_at is not None
        assert job.result == {"worker_port": 9001}

    def test_update_failed_state_sets_finished_at_and_error(self):
        store = JobStore()
        job = store.create(op="pull", model_id="m")
        store.update(job.job_id, state="failed", error="subprocess crashed")
        assert job.state == "failed"
        assert job.error == "subprocess crashed"
        assert job.finished_at is not None

    def test_update_unknown_id_returns_none(self):
        store = JobStore()
        assert store.update("nope", state="done") is None

    def test_list_recent_returns_newest_first(self):
        store = JobStore()
        a = store.create(op="enable", model_id="a")
        b = store.create(op="enable", model_id="b")
        c = store.create(op="enable", model_id="c")
        listing = [j.job_id for j in store.list_recent()]
        assert listing == [c.job_id, b.job_id, a.job_id]

    def test_list_recent_caps_at_max_jobs(self):
        store = JobStore(max_jobs=3)
        for i in range(5):
            store.create(op="enable", model_id=f"m{i}")
        listing = store.list_recent()
        # The deque drops oldest job_ids beyond maxlen; only 3 stay
        # addressable via list_recent.
        assert len(listing) == 3

    def test_jobs_expire_after_retention(self):
        """Lazy reap on get/list_recent. Use a near-zero retention to test."""
        store = JobStore(retention_seconds=0.01)
        job = store.create(op="enable", model_id="m")
        store.update(job.job_id, state="done", result={"ok": True})
        time.sleep(0.05)
        # After retention, the job is reaped on next list call.
        assert store.list_recent() == []
        assert store.get(job.job_id) is None

    def test_pending_jobs_never_expire(self):
        """A pending job has finished_at_monotonic = None; reap skips it."""
        store = JobStore(retention_seconds=0.01)
        job = store.create(op="enable", model_id="m")
        time.sleep(0.05)
        assert store.get(job.job_id) is job

    def test_to_dict_excludes_thread_and_monotonic(self):
        store = JobStore()
        job = store.create(op="enable", model_id="m")
        job.thread = threading.Thread(target=lambda: None)
        store.update(job.job_id, state="done", result={"ok": True})
        d = job.to_dict()
        assert "thread" not in d
        assert "finished_at_monotonic" not in d
        assert d["state"] == "done"
        assert d["result"] == {"ok": True}
        assert d["job_id"] == job.job_id

    def test_shutdown_joins_threads(self):
        store = JobStore()
        ran = {"n": 0}
        def work():
            time.sleep(0.01)
            ran["n"] += 1
        job = store.create(op="enable", model_id="m")
        t = threading.Thread(target=work, daemon=True)
        job.thread = t
        t.start()
        store.shutdown(timeout=1.0)
        assert ran["n"] == 1


class TestDefaultStore:
    def test_get_default_store_returns_singleton(self):
        reset_default_store()
        s1 = get_default_store()
        s2 = get_default_store()
        assert s1 is s2

    def test_reset_default_store_creates_new_instance(self):
        reset_default_store()
        s1 = get_default_store()
        reset_default_store()
        s2 = get_default_store()
        assert s1 is not s2


@pytest.fixture(autouse=True)
def _reset_default_store():
    reset_default_store()
    yield
    reset_default_store()
