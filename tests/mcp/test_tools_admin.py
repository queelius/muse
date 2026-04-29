"""Tests for the 11 admin tools.

Each handler is invoked via MCPServer.call_handler with mocked
AdminClient methods. Tests assert the output content blocks structure
and that admin errors are translated to structured error blocks.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from muse.admin.client import AdminClientError
from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    client = MuseClient(server_url="http://test", admin_token="t")
    # Replace the AdminClient with a fully-mocked object so handlers
    # exercise dispatch + output shaping but never make HTTP calls.
    client.admin = MagicMock()
    return MCPServer(client=client, filter_kind="admin")


def _parse(blocks):
    """Return parsed JSON of the first text block, or raw text."""
    assert blocks
    assert blocks[0]["type"] == "text"
    return json.loads(blocks[0]["text"])


class TestRegistry:
    def test_eleven_admin_tools(self, server):
        names = {t.name for t in server.tools}
        expected = {
            "muse_list_models",
            "muse_get_model_info",
            "muse_search_models",
            "muse_pull_model",
            "muse_remove_model",
            "muse_enable_model",
            "muse_disable_model",
            "muse_probe_model",
            "muse_get_memory_status",
            "muse_get_workers",
            "muse_get_jobs",
        }
        assert names == expected
        assert len(server.tools) == 11

    def test_each_tool_has_description(self, server):
        for t in server.tools:
            assert t.description and len(t.description) >= 30
            assert "muse_" in t.name

    def test_each_tool_has_input_schema(self, server):
        for t in server.tools:
            assert t.inputSchema is not None
            assert t.inputSchema.get("type") == "object"


class TestListModels:
    def test_returns_data_field(self, server):
        server.client.list_models = MagicMock(
            return_value={"data": [
                {"id": "x", "modality": "chat/completion", "status": "enabled"},
                {"id": "y", "modality": "image/generation"},
            ]},
        )
        out = _parse(server.call_handler("muse_list_models", {}))
        assert out["count"] == 2
        assert len(out["data"]) == 2

    def test_filter_modality(self, server):
        server.client.list_models = MagicMock(
            return_value={"data": [
                {"id": "x", "modality": "chat/completion"},
                {"id": "y", "modality": "image/generation"},
            ]},
        )
        out = _parse(server.call_handler(
            "muse_list_models", {"filter_modality": "chat/completion"},
        ))
        assert out["count"] == 1
        assert out["data"][0]["id"] == "x"

    def test_filter_status(self, server):
        server.client.list_models = MagicMock(
            return_value={"data": [
                {"id": "x", "status": "enabled"},
                {"id": "y", "status": "disabled"},
            ]},
        )
        out = _parse(server.call_handler(
            "muse_list_models", {"filter_status": "disabled"},
        ))
        assert out["count"] == 1
        assert out["data"][0]["id"] == "y"


class TestGetModelInfo:
    def test_returns_status_record(self, server):
        server.client.admin.status.return_value = {
            "model_id": "x", "loaded": True, "worker_port": 9001,
        }
        out = _parse(server.call_handler(
            "muse_get_model_info", {"model_id": "x"},
        ))
        assert out["worker_port"] == 9001

    def test_admin_error_translated(self, server):
        server.client.admin.status.side_effect = AdminClientError(
            404, "model_not_found", "no such model", {},
        )
        out = _parse(server.call_handler(
            "muse_get_model_info", {"model_id": "x"},
        ))
        assert out["error"]["code"] == "model_not_found"
        assert out["error"]["status"] == 404


class TestSearchModels:
    def test_passes_through(self, server):
        server.client.search_models = MagicMock(
            return_value={"results": [
                {"uri": "hf://x/y@q4", "modality": "chat/completion"},
            ]},
        )
        out = _parse(server.call_handler(
            "muse_search_models",
            {"query": "qwen", "modality": "chat/completion"},
        ))
        assert out["count"] == 1
        assert out["results"][0]["uri"] == "hf://x/y@q4"

    def test_passes_max_size(self, server):
        server.client.search_models = MagicMock(return_value={"results": []})
        server.call_handler("muse_search_models", {
            "query": "q", "max_size_gb": 5.0, "limit": 10,
        })
        call = server.client.search_models.call_args
        assert call.kwargs["max_size_gb"] == 5.0
        assert call.kwargs["limit"] == 10


class TestPullModel:
    def test_returns_job_id_with_polling_hint(self, server):
        server.client.admin.pull.return_value = {
            "job_id": "abc", "status": "pending",
        }
        out = _parse(server.call_handler(
            "muse_pull_model", {"identifier": "hf://x/y@q4"},
        ))
        assert out["job_id"] == "abc"
        assert "muse_get_jobs" in out["poll_with"]
        assert out["estimated_time"]

    def test_admin_error_translated(self, server):
        server.client.admin.pull.side_effect = AdminClientError(
            503, "admin_disabled", "no token", {},
        )
        out = _parse(server.call_handler(
            "muse_pull_model", {"identifier": "x"},
        ))
        assert out["error"]["code"] == "admin_disabled"


class TestRemoveModel:
    def test_passes_purge_flag(self, server):
        server.client.admin.remove.return_value = {
            "model_id": "x", "removed": True, "purged": True,
        }
        out = _parse(server.call_handler(
            "muse_remove_model", {"model_id": "x", "purge": True},
        ))
        assert out["purged"] is True
        call = server.client.admin.remove.call_args
        assert call.kwargs["purge"] is True

    def test_default_no_purge(self, server):
        server.client.admin.remove.return_value = {
            "model_id": "x", "removed": True, "purged": False,
        }
        server.call_handler("muse_remove_model", {"model_id": "x"})
        call = server.client.admin.remove.call_args
        assert call.kwargs["purge"] is False


class TestEnableDisable:
    def test_enable_returns_job_id(self, server):
        server.client.admin.enable.return_value = {
            "job_id": "j", "status": "pending",
        }
        out = _parse(server.call_handler(
            "muse_enable_model", {"model_id": "x"},
        ))
        assert out["job_id"] == "j"

    def test_disable_is_sync(self, server):
        server.client.admin.disable.return_value = {
            "model_id": "x", "loaded": False, "worker_terminated": True,
        }
        out = _parse(server.call_handler(
            "muse_disable_model", {"model_id": "x"},
        ))
        assert out["loaded"] is False
        assert out["worker_terminated"] is True


class TestProbeModel:
    def test_passes_options(self, server):
        server.client.admin.probe.return_value = {
            "job_id": "j", "status": "pending",
        }
        server.call_handler("muse_probe_model", {
            "model_id": "x", "no_inference": True, "device": "cpu",
        })
        call = server.client.admin.probe.call_args
        assert call.kwargs["no_inference"] is True
        assert call.kwargs["device"] == "cpu"

    def test_returns_polling_hint(self, server):
        server.client.admin.probe.return_value = {"job_id": "j", "status": "pending"}
        out = _parse(server.call_handler("muse_probe_model", {"model_id": "x"}))
        assert "muse_get_jobs" in out["poll_with"]


class TestMemoryAndWorkers:
    def test_memory(self, server):
        server.client.admin.memory.return_value = {
            "gpu": {"used_gb": 12.4, "total_gb": 24.0},
            "cpu": {"used_gb": 6.1, "total_gb": 64.0},
        }
        out = _parse(server.call_handler("muse_get_memory_status", {}))
        assert out["gpu"]["used_gb"] == 12.4

    def test_workers(self, server):
        server.client.admin.workers.return_value = {
            "workers": [{"port": 9001, "models": ["x"]}],
        }
        out = _parse(server.call_handler("muse_get_workers", {}))
        assert out["workers"][0]["port"] == 9001


class TestGetJobs:
    def test_specific_job(self, server):
        server.client.admin.job.return_value = {
            "job_id": "j", "state": "done", "result": {"x": 1},
        }
        out = _parse(server.call_handler(
            "muse_get_jobs", {"job_id": "j"},
        ))
        assert out["state"] == "done"
        assert server.client.admin.job.called
        assert not server.client.admin.jobs.called

    def test_list_when_no_id(self, server):
        server.client.admin.jobs.return_value = {"jobs": []}
        out = _parse(server.call_handler("muse_get_jobs", {}))
        assert "jobs" in out
        assert server.client.admin.jobs.called
        assert not server.client.admin.job.called
