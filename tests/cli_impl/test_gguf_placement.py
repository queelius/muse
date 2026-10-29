"""Native GGUF placement must determine both loading and accounting."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from muse.admin.routes.memory import _per_model_breakdown
from muse.cli_impl import supervisor
from muse.cli_impl.load_director import LoadDirector, LoadEntry
from muse.cli_impl.probe_worker import _resolve_device
from muse.cli_impl.supervisor import SupervisorState, WorkerSpec, backfill_manifest_memory
from muse.core import catalog


MODEL = "placement-test"
BACKEND = "muse.modalities.chat_completion.runtimes.llama_cpp:LlamaCppModel"


class Probe:
    def cpu_free_gb(self):
        return 16.0

    def gpu_free_gb(self):
        return 12.0


@pytest.fixture
def native_entry(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    interpreter = tmp_path / "python"
    interpreter.symlink_to(sys.executable)
    native_file = tmp_path / "native-library"
    native_file.write_text("fixture library identity")
    manifest = {
        "model_id": MODEL, "modality": "chat/completion",
        "hf_repo": "test/placement", "backend_path": BACKEND,
        "capabilities": {"gguf_file": "test.gguf", "memory_gb": 1.0},
    }
    entry = {"enabled": True, "local_dir": str(weights),
             "python_path": str(interpreter), "manifest": manifest}
    catalog._write_catalog({MODEL: entry})
    observed = {"supported": False, "calls": 0}

    def native_query(command, **kwargs):
        assert command[0] == str(interpreter)
        assert kwargs["timeout"] == 15
        observed["calls"] += 1
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "gpu_offload": observed["supported"], "paths": [str(native_file)],
            "platform": observed.get("platform", "linux"),
            "system_info": observed.get("system_info", "CUDA : ARCHS = 800"),
        }), "")

    monkeypatch.setattr("muse.core.placement.run_owned_command", native_query)
    monkeypatch.setattr(supervisor, "_MemoryProbeAdapter", Probe)
    supervisor.clear_supervisor_state()
    yield entry, observed, native_file
    supervisor.clear_supervisor_state()


@pytest.mark.parametrize(
    ("supported", "requested", "layers", "want_device", "want_layers"),
    [(False, "auto", -1, "cpu", 0), (True, "auto", -1, "cuda", -1),
     (True, "cpu", -1, "cpu", 0), (True, "auto", 0, "cpu", 0)],
)
def test_backfill_resolves_native_placement(native_entry, supported, requested, layers, want_device, want_layers):
    entry, observed, _ = native_entry
    observed["supported"] = supported
    entry["manifest"]["capabilities"]["n_gpu_layers"] = layers
    catalog._write_catalog({MODEL: entry})
    prepared = backfill_manifest_memory(entry["manifest"], MODEL, supervisor_device=requested)
    assert prepared["capabilities"]["device"] == want_device
    assert prepared["capabilities"]["n_gpu_layers"] == want_layers
    if requested == "cpu" or layers == 0:
        assert observed["calls"] == 0


def test_unsupported_explicit_cuda_is_unservable(native_entry):
    entry, _, _ = native_entry
    entry["device_override"] = "cuda"
    catalog._write_catalog({MODEL: entry})
    state = SupervisorState(device="auto")
    supervisor.validate_catalog_at_boot(state)
    assert MODEL in state.unservable_reasons
    assert "offload" in state.unservable_reasons[MODEL]


def test_auto_override_uses_native_capability_not_supervisor_cuda(native_entry):
    entry, _, _ = native_entry
    entry["device_override"] = "auto"
    catalog._write_catalog({MODEL: entry})
    prepared = backfill_manifest_memory(entry["manifest"], MODEL, supervisor_device="cuda")
    assert prepared["capabilities"]["device"] == "cpu"


def test_capability_cache_refreshes_when_native_library_changes(native_entry):
    entry, observed, native_file = native_entry
    first = backfill_manifest_memory(entry["manifest"], MODEL)
    second = backfill_manifest_memory(entry["manifest"], MODEL)
    assert first["capabilities"]["device"] == second["capabilities"]["device"] == "cpu"
    assert observed["calls"] == 1
    observed["supported"] = True
    native_file.write_text("updated library identity with GPU support")
    assert backfill_manifest_memory(entry["manifest"], MODEL)["capabilities"]["device"] == "cuda"
    assert observed["calls"] == 2


def test_director_reserves_cpu_for_cpu_only_native_runtime(native_entry):
    entry, _, _ = native_entry
    prepared = backfill_manifest_memory(entry["manifest"], MODEL)
    director = LoadDirector(enable_fn=lambda _: 9001, disable_fn=lambda _: None, memory_probe=Probe())
    director.warmup(MODEL, manifest=prepared)
    assert director.loaded[MODEL].pool == "cpu"


def test_probe_resolves_native_runtime_instead_of_torch_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        llama_backend_init=lambda: None, llama_supports_gpu_offload=lambda: False,
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    entry = SimpleNamespace(backend_path=BACKEND, extra={"gguf_file": "test.gguf"})
    assert _resolve_device("auto", entry) == "cpu"


def test_memory_view_uses_resident_pool_not_manifest_guess(native_entry):
    _, _, _ = native_entry
    state = SupervisorState(device="auto")
    state.director = LoadDirector(enable_fn=lambda _: 9001, disable_fn=lambda _: None, memory_probe=Probe())
    now = time.monotonic()
    state.director.loaded[MODEL] = LoadEntry(MODEL, 9001, 1.0, 0, now, now, pool="cpu")
    state.workers.append(WorkerSpec(
        models=[MODEL], python_path=sys.executable, port=9001, status="running",
        process=SimpleNamespace(pid=424242, returncode=None, poll=lambda: None),
    ))
    supervisor.set_supervisor_state(state)
    assert [r["model_id"] for r in _per_model_breakdown("cpu", "cpu")] == [MODEL]
    assert _per_model_breakdown("cuda", "gpu") == []


def test_memory_view_excludes_retained_dead_worker(native_entry):
    state = SupervisorState(device="cpu")
    state.workers.append(WorkerSpec(models=[MODEL], python_path=sys.executable, port=9001, status="dead"))
    supervisor.set_supervisor_state(state)
    assert _per_model_breakdown("cpu", "cpu") == []


@pytest.mark.parametrize("device,layers", [("cpu", -1), ("auto", 0), ("cuda", 0)])
def test_native_cpu_placement_never_queries_support(device, layers, monkeypatch):
    def forbidden_query():
        pytest.fail("CPU or zero layers must not inspect GPU capability")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        llama_backend_init=forbidden_query,
    ))
    entry = SimpleNamespace(backend_path=BACKEND, extra={"n_gpu_layers": layers})
    assert _resolve_device(device, entry) == "cpu"


def test_director_keeps_resident_pool_after_preferences_change(native_entry):
    entry, observed, _ = native_entry
    director = LoadDirector(enable_fn=lambda _: 9001, disable_fn=lambda _: None, memory_probe=Probe())
    director.warmup(MODEL, manifest=entry["manifest"])
    observed["supported"] = True
    entry["device_override"] = "cuda"
    catalog._write_catalog({MODEL: entry})
    director.warmup(MODEL, manifest=entry["manifest"])
    assert director.loaded[MODEL].pool == "cpu"
    assert observed["calls"] == 1


def test_capability_cache_refreshes_when_interpreter_changes(native_entry):
    entry, observed, _ = native_entry
    first = backfill_manifest_memory(entry["manifest"], MODEL)
    assert first["capabilities"]["device"] == "cpu"
    observed["supported"] = True
    interpreter = entry["python_path"]
    # Replacing the venv executable (even with the same target) is a new identity.
    from pathlib import Path
    Path(interpreter).unlink()
    Path(interpreter).write_text("new interpreter")
    assert backfill_manifest_memory(entry["manifest"], MODEL)["capabilities"]["device"] == "cuda"
    assert observed["calls"] == 2


def test_memory_view_excludes_exited_running_worker(native_entry):
    state = SupervisorState(device="cpu")
    state.workers.append(WorkerSpec(
        models=[MODEL], python_path=sys.executable, port=9001, status="running",
        # A test double with a concrete pid: the memory view now observes it
        # through the supervisor's identity-checked state helper.
        process=SimpleNamespace(pid=4242, returncode=None, poll=lambda: 1),
    ))
    supervisor.set_supervisor_state(state)
    assert _per_model_breakdown("cpu", "cpu") == []


def test_zero_operator_gpu_layers_override_pools_on_cpu(native_entry):
    entry, observed, _ = native_entry
    observed["supported"] = True
    entry["gpu_layers_override"] = 0
    entry["manifest"]["capabilities"]["n_gpu_layers"] = -1
    catalog._write_catalog({MODEL: entry})
    prepared = backfill_manifest_memory(entry["manifest"], MODEL)
    assert prepared["capabilities"]["device"] == "cpu"
    assert prepared["capabilities"]["n_gpu_layers"] == 0
    assert observed["calls"] == 0


def test_non_native_resolver_retains_explicit_device():
    from muse.core.placement import resolve_model_placement
    placement = resolve_model_placement("generic", {"capabilities": {"device": "mps"}}, entry={})
    assert placement.device == "mps"
    assert placement.pool == "cpu"


def test_native_probe_respects_zero_operator_gpu_layers(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        llama_backend_init=lambda: pytest.fail("zero layers queried native support"),
    ))
    entry = SimpleNamespace(backend_path=BACKEND, extra={"n_gpu_layers": -1})
    assert _resolve_device("cuda", entry, gpu_layers_override=0) == "cpu"


def test_director_preserves_mps_measurement_bucket():
    class ChangingProbe:
        free = 16.0

        def cpu_free_gb(self):
            return self.free

        def gpu_free_gb(self):
            return None

    probe = ChangingProbe()

    def enable(_):
        probe.free = 15.0
        return 9001

    director = LoadDirector(enable_fn=enable, disable_fn=lambda _: None, memory_probe=probe)
    observations = []
    director.observed_peak = lambda model_id, **kwargs: observations.append(kwargs)
    director.warmup("mps-test", manifest={"capabilities": {"device": "mps", "memory_gb": 1}})
    assert director.loaded["mps-test"].pool == "cpu"
    assert observations[0]["device"] == "mps"


@pytest.mark.parametrize(
    "device,layers,supported,want",
    [("auto", -1, False, ("cpu", "cpu", 0)),
     ("auto", -1, True, ("cuda", "cuda", -1)),
     ("cpu", 12, True, ("cpu", "cpu", 0)),
     ("cuda", 0, False, ("cpu", "cpu", 0)),
     ("mps", 12, True, ("mps", "cpu", 12))],
)
def test_native_placement_contract(device, layers, supported, want):
    from muse.core.placement import native_placement
    placement = native_placement(
        device, layers, gpu_offload=supported, gpu_backend="metal" if device == "mps" else "cuda",
    )
    assert (placement.device, placement.pool, placement.n_gpu_layers) == want


@pytest.mark.slow
def test_owned_query_tracks_package_and_native_library(tmp_path, monkeypatch):
    """Exercise the query producer with a fake package, never host ML libraries."""
    from muse.core import placement
    from muse.core.owned_command import run_owned_command

    package = tmp_path / "llama_cpp"
    package.mkdir()
    library = package / "fake-native.so"
    library.write_text("cpu")
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "def llama_backend_init(): pass\n"
        "def llama_supports_gpu_offload():\n"
        "    return (Path(__file__).parent / 'fake-native.so').read_text() == 'gpu'\n"
    )
    interpreter = tmp_path / "python"
    interpreter.symlink_to(sys.executable)
    calls = []

    def isolated_query(command, **kwargs):
        calls.append(command[0])
        assert command[0] == str(interpreter)
        command = [*command[:2], f"import sys; sys.path.insert(0, {str(tmp_path)!r})\n" + command[2]]
        return run_owned_command(command, **kwargs)

    monkeypatch.setattr(placement, "run_owned_command", isolated_query)
    manifest = {"backend_path": BACKEND}
    entry = {"python_path": str(interpreter)}
    assert placement.resolve_model_placement(MODEL, manifest, entry=entry).device == "cpu"
    assert placement.resolve_model_placement(MODEL, manifest, entry=entry).device == "cpu"
    assert len(calls) == 1
    library.write_text("gpu")
    assert placement.resolve_model_placement(MODEL, manifest, entry=entry).device == "cuda"
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["timeout", "malformed"])
def test_unknown_native_capability_rejects_admission_and_retries(native_entry, monkeypatch, failure):
    from muse.core import placement
    entry, observed, _ = native_entry
    original_query = placement.run_owned_command

    def broken_query(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "not a capability record", "")

    monkeypatch.setattr(placement, "run_owned_command", broken_query)
    with pytest.raises(RuntimeError, match="query.*offload"):
        backfill_manifest_memory(entry["manifest"], MODEL)
    state = SupervisorState()
    supervisor.validate_catalog_at_boot(state)
    assert "query" in state.unservable_reasons[MODEL]
    monkeypatch.setattr(placement, "run_owned_command", original_query)
    assert backfill_manifest_memory(entry["manifest"], MODEL)["capabilities"]["device"] == "cpu"
    assert observed["calls"] == 1


def test_concurrent_failed_native_queries_share_one_owned_child(native_entry, monkeypatch):
    from muse.core import placement
    entry, _, _ = native_entry
    started = threading.Event()
    joined = threading.Event()
    release = threading.Event()
    queries = []

    class ObservedFuture(Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    def slow_failed_query(command, **kwargs):
        queries.append(command)
        started.set()
        assert release.wait(2)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(placement, "Future", ObservedFuture, raising=False)
    monkeypatch.setattr(placement, "run_owned_command", slow_failed_query)
    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(backfill_manifest_memory, entry["manifest"], MODEL)
        assert started.wait(2)
        waiter = executor.submit(backfill_manifest_memory, entry["manifest"], MODEL)
        try:
            assert joined.wait(1), "concurrent callers did not join the same capability query"
        finally:
            release.set()
        for result in (owner, waiter):
            with pytest.raises(RuntimeError, match="query.*offload"):
                result.result(timeout=2)
    assert len(queries) == 1


def test_resident_backfill_sizes_without_resolving_edited_placement(native_entry):
    entry, observed, _ = native_entry
    entry["device_override"] = "cuda"
    del entry["manifest"]["capabilities"]["memory_gb"]
    entry["measurements"] = {"cpu": {"peak_bytes": 2 * 1024**3}}
    catalog._write_catalog({MODEL: entry})
    prepared = backfill_manifest_memory(entry["manifest"], MODEL, resolve_placement=False)
    assert prepared["capabilities"]["memory_gb"] == 2.0
    assert prepared["capabilities"].get("device", "auto") == "auto"
    assert observed["calls"] == 0


def test_darwin_native_auto_uses_unified_memory(monkeypatch):
    from muse.core import placement
    monkeypatch.setattr(sys, "platform", "darwin")
    result = placement.native_placement("auto", -1, gpu_offload=True)
    assert (result.device, result.pool, result.n_gpu_layers) == ("mps", "cpu", -1)
    with pytest.raises(RuntimeError, match="offload"):
        placement.native_placement("cuda", -1, gpu_offload=True)


def test_cuda_build_cannot_claim_explicit_mps():
    from muse.core.placement import native_placement
    with pytest.raises(RuntimeError, match="offload"):
        native_placement("mps", -1, gpu_offload=True, gpu_backend="cuda")


def test_recorded_interpreter_metal_capability_pools_on_cpu(native_entry):
    entry, observed, _ = native_entry
    observed.update(supported=True, platform="darwin", system_info="Metal : EMBED_LIBRARY = 1")
    prepared = backfill_manifest_memory(entry["manifest"], MODEL)
    assert prepared["capabilities"]["device"] == "mps"
    assert prepared["capabilities"]["n_gpu_layers"] == -1
    director = LoadDirector(enable_fn=lambda _: 9001, disable_fn=lambda _: None, memory_probe=Probe())
    director.warmup(MODEL, manifest=prepared)
    assert director.loaded[MODEL].pool == "cpu"


def test_native_probe_uses_metal_not_torch_cuda(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        llama_backend_init=lambda: None, llama_supports_gpu_offload=lambda: True,
        llama_print_system_info=lambda: b"Metal : EMBED_LIBRARY = 1",
    ))
    entry = SimpleNamespace(backend_path=BACKEND, extra={})
    assert _resolve_device("auto", entry) == "mps"


@pytest.mark.parametrize("info", ["Vulkan : UMA = 1", "UNKNOWN : GPU = 1", "CUDA : ARCHS = 800 | Metal : 1"])
def test_identifiable_unsupported_native_backend_is_unservable(native_entry, info):
    entry, observed, _ = native_entry
    observed.update(supported=True, system_info=info)
    with pytest.raises(RuntimeError, match="offload"):
        backfill_manifest_memory(entry["manifest"], MODEL)


def test_native_query_wait_deadline_is_actionable(native_entry, monkeypatch):
    from muse.core import placement
    entry, _, _ = native_entry
    original_query = placement.run_owned_command
    started = threading.Event()
    release = threading.Event()

    class TimedOutWaiter(Future):
        def result(self, timeout=None):
            raise TimeoutError("owner is still cleaning its owned child")

    def slow_query(command, **kwargs):
        started.set()
        assert release.wait(2)
        return original_query(command, **kwargs)

    monkeypatch.setattr(placement, "Future", TimedOutWaiter)
    monkeypatch.setattr(placement, "run_owned_command", slow_query)
    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(backfill_manifest_memory, entry["manifest"], MODEL)
        assert started.wait(2)
        try:
            with pytest.raises(RuntimeError, match="query.*offload"):
                backfill_manifest_memory(entry["manifest"], MODEL)
        finally:
            release.set()
        assert owner.result(timeout=2)["capabilities"]["device"] == "cpu"


@pytest.mark.parametrize("change", ["blocked", "evicting", "resident", "resident_error", "claim", "claim_error", "manifest", "catalog", "budget"])
def test_native_admission_rechecks_after_unlocked_preparation(native_entry, monkeypatch, change):
    from muse.admin.operations import OperationError
    from muse.cli_impl.load_director import InFlightLoad
    from muse.core import placement

    entry, _, _ = native_entry
    manifest = entry["manifest"]
    director = LoadDirector(
        enable_fn=lambda _: 9001, disable_fn=lambda _: None, memory_probe=Probe(),
        cpu_headroom_gb=0,
    )
    started = threading.Event()
    release = threading.Event()

    def prepare(*args, **kwargs):
        started.set()
        assert release.wait(3)
        if change.endswith("_error"):
            raise RuntimeError("native capability became unavailable")
        return placement.Placement("cpu", "cpu", 0)

    monkeypatch.setattr(placement, "resolve_model_placement", prepare)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(director._decide, MODEL, manifest=manifest)
        assert started.wait(1)
        try:
            assert director.lock.acquire(timeout=1), "native query retained the admission lock"
            try:
                if change == "blocked":
                    director._blocked_models.add(MODEL)
                elif change == "evicting":
                    director.evicting.add(MODEL)
                elif change.startswith("resident"):
                    now = time.monotonic()
                    director.loaded[MODEL] = LoadEntry(MODEL, 9003, 1.0, 0, now, now, pool="cuda")
                elif change.startswith("claim"):
                    claim = InFlightLoad(threading.Event(), 1.0, "cuda")
                    director.in_flight_loads[MODEL] = claim
                elif change == "manifest":
                    manifest["capabilities"]["memory_gb"] = 2.0
                elif change == "catalog":
                    entry["device_override"] = "cpu"
                    catalog._write_catalog({MODEL: entry})
                elif change == "budget":
                    director.cpu_budget_gb = 0.25
            finally:
                director.lock.release()
        finally:
            release.set()

        if change.startswith(("resident", "claim")):
            decision = pending.result(timeout=2)
            if change.startswith("resident"):
                assert decision == ("hot", 9003)
                assert director.loaded[MODEL].pool == "cuda"
                assert director.loaded[MODEL].refcount == 1
            else:
                assert decision == ("wait", claim)
        else:
            expected = {
                "blocked": "model_disabled", "evicting": "model_eviction_in_progress",
                "manifest": "model_placement_changed", "catalog": "model_placement_changed",
                "budget": "model_too_large_for_device",
            }
            with pytest.raises(OperationError) as caught:
                pending.result(timeout=2)
            assert caught.value.code == expected[change]
            assert MODEL not in director.in_flight_loads
