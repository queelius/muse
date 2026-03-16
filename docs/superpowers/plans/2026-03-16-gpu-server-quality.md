# GPU Server, Quality Pipeline, and Benchmark Framework — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GPU acceleration via a FastAPI server, improve TTS quality with LLM rewriting and better failure detection, and build a benchmark framework for iterating on performance.

**Architecture:** Two parallel tracks — Infrastructure (GPU device, server, client, benchmarks) and Quality (clean_text fixes, garbled audio detection, LLM rewriting). The benchmark framework is built early since both tracks use it. Each task produces a working, testable increment.

**Tech Stack:** PyTorch (device management), FastAPI/uvicorn (optional server), requests (client/rewrite HTTP), pytest (testing)

**Spec:** `docs/superpowers/specs/2026-03-16-gpu-server-quality-design.md`

---

## File Map

**New files:**
- `narro/server.py` — FastAPI server with `/v1/audio/speech` and `/health`
- `narro/client.py` — HTTP client for remote inference
- `narro/bench.py` — Benchmark framework (replaces `benchmarks/bench.py`)
- `narro/rewrite.py` — LLM paragraph rewriting via `/v1/chat/completions`
- `tests/test_server.py` — Server endpoint tests
- `tests/test_client.py` — Client tests
- `tests/test_bench.py` — Benchmark framework tests
- `tests/test_rewrite.py` — Rewrite module tests
- `tests/test_quality_check.py` — Garbled audio detection tests

**Modified files:**
- `narro/tts.py` — Device parameter, quality_check, threading defaults, retries
- `narro/backends/transformers.py` — Device parameter
- `narro/backends/base.py` — Device-aware inference, tighter max_new_tokens
- `narro/decode_only.py` — Device-aware decoder loading
- `narro/cli.py` — `serve`, `bench` subcommands, `--server`/`--device` flags, update `_subcommands`
- `narro/hugo/cli.py` — Client integration, `--rewrite` flag, use `extract_paragraphs`
- `narro/hugo/extract.py` — `extract_paragraphs()` helper
- `narro/utils/text_normalizer.py` — clean_text improvements
- `pyproject.toml` — `[server]` optional extra

---

## Chunk 1: GPU Device Support

### Task 1: Thread device parameter through TransformersModel

**Files:**
- Modify: `narro/backends/transformers.py`
- Test: `tests/test_tts_coverage.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_tts_coverage.py`, add a test that `TransformersModel` accepts and stores a `device` parameter:

```python
class TestTransformersModelDevice:
    def test_init_accepts_device_parameter(self):
        """TransformersModel should accept a device parameter."""
        from unittest.mock import patch, MagicMock
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok:
            mock_model.from_pretrained.return_value = MagicMock()
            mock_tok.from_pretrained.return_value = MagicMock()
            model = TransformersModel(compile=False, quantize=False, device='cpu')
            assert model.device == 'cpu'

    def test_init_defaults_to_cpu(self):
        """Device should default to cpu."""
        from unittest.mock import patch, MagicMock
        with patch('narro.backends.transformers.AutoModelForCausalLM') as mock_model, \
             patch('narro.backends.transformers.AutoTokenizer') as mock_tok:
            mock_model.from_pretrained.return_value = MagicMock()
            mock_tok.from_pretrained.return_value = MagicMock()
            model = TransformersModel(compile=False, quantize=False)
            assert model.device == 'cpu'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_coverage.py::TestTransformersModelDevice -v`
Expected: FAIL — `TransformersModel.__init__()` got unexpected keyword argument 'device'

- [ ] **Step 3: Add device parameter to TransformersModel**

In `narro/backends/transformers.py`, update `__init__`:

```python
class TransformersModel(BaseModel):
    def __init__(self, model_path=None, compile=True, quantize=True, device='cpu'):
        model_name_or_path = model_path if model_path else 'ekwek/Soprano-1.1-80M'
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()

        # quantize_dynamic is CPU-only
        if quantize and device == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        if compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_coverage.py::TestTransformersModelDevice -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add narro/backends/transformers.py tests/test_tts_coverage.py
git commit -m "feat: add device parameter to TransformersModel"
```

### Task 2: Thread device through BaseModel.infer

**Files:**
- Modify: `narro/backends/base.py`
- Test: `tests/test_tts_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
class TestBaseModelDeviceInfer:
    def test_infer_moves_inputs_to_device(self):
        """Inputs should be moved to self.device before generation."""
        from unittest.mock import MagicMock

        model = TransformersModel.__new__(TransformersModel)
        model.device = 'cpu'
        mock_tokenizer_output = MagicMock()
        mock_tokenizer_output.to.return_value = mock_tokenizer_output
        mock_tokenizer_output.__getitem__ = MagicMock(return_value=MagicMock())
        model.tokenizer = MagicMock(return_value=mock_tokenizer_output)
        model.model = MagicMock()
        model.model.config.eos_token_id = 0
        model.model.config.hidden_size = 512
        mock_outputs = MagicMock()
        mock_outputs.hidden_states = []
        mock_outputs.sequences = torch.zeros(1, 1, dtype=torch.long)
        model.model.generate.return_value = mock_outputs

        model.infer(['test'])
        mock_tokenizer_output.to.assert_called_with('cpu')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_coverage.py::TestBaseModelDeviceInfer -v`
Expected: FAIL — `.to()` not called

- [ ] **Step 3: Add device-aware input handling in base.py**

In `narro/backends/base.py`, update `infer` method — add `.to(self.device)` after the tokenizer call:

```python
inputs = self.tokenizer(
    prompts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512,
).to(self.device)
```

Also update `stream_infer` to move inputs to device:

```python
inputs = self.tokenizer(prompt, return_tensors='pt')
input_ids = inputs['input_ids'].to(self.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_coverage.py::TestBaseModelDeviceInfer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add narro/backends/base.py tests/test_tts_coverage.py
git commit -m "feat: move tokenizer inputs to device in BaseModel.infer"
```

### Task 3: Thread device through decoder loading and Narro constructor

**Files:**
- Modify: `narro/decode_only.py`, `narro/tts.py`
- Test: `tests/test_tts_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
class TestDeviceAutoDetection:
    def test_narro_accepts_device_parameter(self):
        """Narro constructor should accept device='cpu' without error."""
        from unittest.mock import patch, MagicMock
        with patch('narro.tts.TransformersModel') as mock_tm, \
             patch('narro.tts.load_decoder') as mock_ld:
            mock_tm.return_value = MagicMock()
            mock_ld.return_value = MagicMock()
            tts = Narro(device='cpu', compile=False)
            assert tts.device == 'cpu'
            mock_tm.assert_called_once()
            assert mock_tm.call_args.kwargs.get('device') == 'cpu'

    def test_narro_auto_detects_cpu(self):
        """device='auto' should resolve to 'cpu' when no GPU available."""
        from unittest.mock import patch, MagicMock
        with patch('narro.tts.TransformersModel') as mock_tm, \
             patch('narro.tts.load_decoder') as mock_ld, \
             patch('torch.cuda.is_available', return_value=False):
            mock_tm.return_value = MagicMock()
            mock_ld.return_value = MagicMock()
            tts = Narro(device='auto', compile=False)
            assert tts.device == 'cpu'

    def test_load_decoder_accepts_device(self):
        """load_decoder should accept a device parameter."""
        from unittest.mock import patch, MagicMock
        from narro.decode_only import load_decoder
        from narro.vocos.decoder import SopranoDecoder
        with patch('narro.decode_only.hf_hub_download', return_value='/fake/path'), \
             patch('torch.load', return_value={}), \
             patch('narro.decode_only.load_with_migration', return_value={}), \
             patch.object(SopranoDecoder, 'load_state_dict'), \
             patch.object(SopranoDecoder, 'to', return_value=SopranoDecoder()) as mock_to:
            load_decoder(device='cpu', compile=False)
            mock_to.assert_called_with('cpu')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_coverage.py::TestDeviceAutoDetection -v`
Expected: FAIL — unexpected keyword argument 'device'

- [ ] **Step 3: Implement device support in Narro and load_decoder**

In `narro/tts.py`, update `Narro.__init__` to accept `device='auto'` parameter. Add auto-detection logic for cuda/mps/cpu. Pass `device` to `TransformersModel` and `load_decoder`. Create warmup tensors on the correct device.

In `narro/decode_only.py`, update `load_decoder` to accept `device='cpu'`. After `load_state_dict`, call `decoder.to(device)`. Update `decode()` to detect the device from the decoder's parameters and create `batch_hidden_states` on the correct device.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_coverage.py::TestDeviceAutoDetection -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add narro/tts.py narro/decode_only.py tests/test_tts_coverage.py
git commit -m "feat: add device auto-detection to Narro and load_decoder"
```

### Task 4: Add --device flag to CLI and update _subcommands

**Files:**
- Modify: `narro/cli.py`

- [ ] **Step 1: Add --device to _add_common_args**

Add `--device` with choices `['auto', 'cpu', 'cuda', 'mps']` defaulting to `'auto'`. Update `cmd_speak` and `cmd_encode` to pass `device=args.device` to `Narro()`.

- [ ] **Step 2: Update _subcommands set**

Change line 107 to: `_subcommands = {'speak', 'encode', 'decode', 'hugo', 'serve', 'bench'}`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add narro/cli.py
git commit -m "feat: add --device flag to CLI, update _subcommands"
```

---

## Chunk 2: Benchmark Framework

### Task 5: Create benchmark module

**Files:**
- Create: `narro/bench.py`
- Create: `tests/test_bench.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for narro.bench — benchmark framework."""
from narro.bench import BENCH_CORPUS, format_table


class TestBenchCorpus:
    def test_corpus_has_required_keys(self):
        assert 'short' in BENCH_CORPUS
        assert 'medium' in BENCH_CORPUS
        assert 'long' in BENCH_CORPUS
        assert 'blog' in BENCH_CORPUS

    def test_corpus_values_are_nonempty_strings(self):
        for key, text in BENCH_CORPUS.items():
            assert isinstance(text, str)
            assert len(text) > 0


class TestFormatTable:
    def test_format_table_produces_output(self):
        results = {
            "device": "cpu",
            "compile": True,
            "num_runs": 3,
            "texts": {
                "short": {
                    "input_chars": 22,
                    "tokens": 10,
                    "audio_sec": 0.64,
                    "encode_ms": 400.0,
                    "decode_ms": 20.0,
                    "total_ms": 420.0,
                    "rtf": 0.66,
                }
            }
        }
        output = format_table(results)
        assert "short" in output
        assert "cpu" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bench.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement bench module**

Create `narro/bench.py` with:
- `BENCH_CORPUS` dict (short, medium, long, blog texts)
- `run_benchmark(device, compile, quantize, num_runs, num_threads)` — measures startup, preprocessing, encode, decode, total, RTF per corpus text
- `format_table(results)` — human-readable table output

The blog corpus text should exercise difficult cases: years, dollar amounts, abbreviations, URLs, parenthetical content.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add narro/bench.py tests/test_bench.py
git commit -m "feat: add benchmark framework with corpus and table output"
```

### Task 6: Add `narro bench` CLI subcommand

**Files:**
- Modify: `narro/cli.py`

- [ ] **Step 1: Add bench subcommand**

Add `cmd_bench(args)` handler that calls `run_benchmark()` and prints table or JSON. Add `bench` subparser with `--runs`, `--json` args plus common args.

- [ ] **Step 2: Verify CLI help shows bench**

Run: `python -m narro.cli --help`
Expected: Shows `bench` in subcommands list

- [ ] **Step 3: Commit**

```bash
git add narro/cli.py
git commit -m "feat: add 'narro bench' CLI subcommand"
```

---

## Chunk 3: Server and Client

### Task 7: Create the FastAPI server

**Files:**
- Create: `narro/server.py`
- Create: `tests/test_server.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add server optional dependency to pyproject.toml**

Add `[project.optional-dependencies]` section with `server = ["fastapi", "uvicorn"]`.

- [ ] **Step 2: Write the failing test**

```python
"""Tests for narro.server — FastAPI TTS server."""
import json
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked Narro."""
    with patch('narro.server._get_tts') as mock_get:
        import torch
        mock_tts = MagicMock()
        mock_tts.device = 'cpu'
        mock_tts.model_id = 'test/model'
        mock_tts.infer.return_value = torch.randn(3200)
        mock_get.return_value = mock_tts

        from narro.server import app
        yield TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "device" in data


class TestSpeech:
    def test_speech_returns_audio(self, client):
        resp = client.post("/v1/audio/speech", json={"input": "Hello world"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 44

    def test_speech_rejects_empty_input(self, client):
        resp = client.post("/v1/audio/speech", json={"input": ""})
        assert resp.status_code == 400

    def test_speech_rejects_missing_input(self, client):
        resp = client.post("/v1/audio/speech", json={})
        assert resp.status_code in (400, 422)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_server.py -v`
Expected: FAIL — ImportError or skip if fastapi not installed

- [ ] **Step 4: Implement the server**

Create `narro/server.py` with:
- `create_app(device, model_path, compile, quantize)` — creates FastAPI app, loads model once
- `_get_tts()` — returns global Narro instance
- `GET /health` — returns status, device, model
- `POST /v1/audio/speech` — accepts `input`, `response_format`, `stream`, `align`
- Non-streaming: returns WAV/opus bytes, alignment in `X-Alignment` header
- Streaming: SSE with base64-encoded raw PCM chunks, format metadata in first event
- `asyncio.Lock` for single-request concurrency
- `serve(host, port, device, ...)` — convenience function for `narro serve`

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_server.py -v`
Expected: PASS (or skip if fastapi not installed)

- [ ] **Step 6: Commit**

```bash
git add narro/server.py tests/test_server.py pyproject.toml
git commit -m "feat: add FastAPI TTS server with /v1/audio/speech endpoint"
```

### Task 8: Add `narro serve` CLI subcommand

**Files:**
- Modify: `narro/cli.py`

- [ ] **Step 1: Add serve command handler and subparser**

Add `cmd_serve(args)` that calls `narro.server.serve()`. Add `serve` subparser with `--host`, `--port` args plus common args.

- [ ] **Step 2: Commit**

```bash
git add narro/cli.py
git commit -m "feat: add 'narro serve' CLI subcommand"
```

### Task 9: Create the HTTP client

**Files:**
- Create: `narro/client.py`
- Create: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for narro.client — HTTP client for remote narro server."""
import json
from unittest.mock import patch, MagicMock

import pytest

from narro.client import NarroClient


class TestNarroClient:
    def test_health_returns_dict(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "device": "cuda"}

        with patch('narro.client.requests.get', return_value=mock_resp):
            client = NarroClient("http://localhost:8000")
            result = client.health()
            assert result["status"] == "ok"

    def test_infer_returns_audio_bytes(self):
        fake_wav = b"RIFF" + b"\x00" * 100
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav
        mock_resp.headers = {"content-type": "audio/wav"}

        with patch('narro.client.requests.post', return_value=mock_resp):
            client = NarroClient("http://localhost:8000")
            audio = client.infer("Hello world")
            assert audio == fake_wav

    def test_generate_with_alignment(self):
        fake_wav = b"RIFF" + b"\x00" * 100
        alignment = [{"paragraph": 0, "start": 0.0, "end": 1.5}]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav
        mock_resp.headers = {
            "content-type": "audio/wav",
            "x-alignment": json.dumps(alignment),
        }

        with patch('narro.client.requests.post', return_value=mock_resp):
            client = NarroClient("http://localhost:8000")
            audio, align = client.generate_with_alignment(
                ["Paragraph one.", "Paragraph two."],
                out_path="/dev/null",
            )
            assert align == alignment

    def test_server_unreachable_raises(self):
        import requests as req_lib
        with patch('narro.client.requests.get',
                   side_effect=req_lib.ConnectionError("refused")):
            client = NarroClient("http://localhost:9999")
            with pytest.raises(ConnectionError):
                client.health()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement the client**

Create `narro/client.py` with `NarroClient` class:
- `health()` — GET `/health`, raise `ConnectionError` on failure
- `infer(text, out_path, response_format)` — POST to `/v1/audio/speech`
- `generate_with_alignment(paragraphs, out_path, response_format)` — POST with `align=True`, parse `X-Alignment` header

Uses `requests` (transitive dep). No new dependencies.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add narro/client.py tests/test_client.py
git commit -m "feat: add HTTP client for remote narro server"
```

### Task 10: Wire client into CLI and Hugo

**Files:**
- Modify: `narro/cli.py`, `narro/hugo/cli.py`

- [ ] **Step 1: Add --server flag to common args**

In `_add_common_args`, add `--server`/`-s` parameter. Update `cmd_speak` to check `args.server` or `NARRO_SERVER` env var and use `NarroClient` instead of local `Narro` when set.

- [ ] **Step 2: Update Hugo CLI for client support**

In `narro/hugo/cli.py`, update `_lazy_import` to check `NARRO_SERVER` env var. When set, use `NarroClient` instead of local `Narro`.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add narro/cli.py narro/hugo/cli.py
git commit -m "feat: wire client into CLI (--server) and Hugo integration"
```

---

## Chunk 4: Quality — clean_text and Garbled Audio Detection

### Task 11: Fix the worst clean_text transforms

**Files:**
- Modify: `narro/utils/text_normalizer.py`
- Test: `tests/test_tts_coverage.py`

- [ ] **Step 1: Write tests for the fixes**

```python
class TestCleanTextFixes:
    def test_colons_not_converted_to_periods(self):
        """Colons should not create artificial sentence boundaries."""
        from narro.utils.text_normalizer import clean_text
        result = clean_text("Note: this is important")
        # The old code converted ':' to '.' creating "note. this is important"
        assert ". this" not in result
        assert "note" in result
        assert "important" in result

    def test_urls_dropped_entirely(self):
        """URLs should be removed, not spelled out."""
        from narro.utils.text_normalizer import clean_text
        result = clean_text("Visit https://example.com for details")
        assert "h t t p" not in result
        assert "colon" not in result
        assert "slash" not in result

    def test_email_dropped(self):
        """Email addresses should be removed."""
        from narro.utils.text_normalizer import clean_text
        result = clean_text("Contact user@example.com for help")
        assert "example" not in result

    def test_normalize_newlines_no_period_insertion(self):
        """Newlines should not get periods added."""
        from narro.utils.text_normalizer import normalize_newlines
        result = normalize_newlines("first line\nsecond line")
        assert result.count('.') == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tts_coverage.py::TestCleanTextFixes -v`
Expected: FAIL

- [ ] **Step 3: Apply fixes**

In `narro/utils/text_normalizer.py`:

1. Change colon handling from `(':', '.')` to `(':', ',')` in `_special_characters`
2. Add `remove_urls()` function that strips `https?://\S+` and `\S+@\S+\.\S+` patterns
3. Call `remove_urls` in `clean_text()` after `convert_to_ascii`, before `normalize_newlines`
4. Simplify `normalize_newlines()` to just join lines with spaces (no period insertion)
5. (Spec 6c) Audit parentheses handling — verify `(text)` → `, text,` produces acceptable TTS output. If not, simplify to just strip parens.
6. (Spec 6e) Evaluate CamelCase splitting — consider removing `normalize_mixedcase` from the pipeline since code identifiers should already be stripped by `extract_prose()`. If kept, at minimum fix the "McDonald" → "Mc Donald" false positive.
7. (Spec 6f) Run the BENCH_CORPUS through the updated `clean_text()` and manually inspect output. Verify no remaining transforms produce text worse for TTS than the original.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tts_coverage.py::TestCleanTextFixes -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: All pass (fix any tests that relied on old colon or newline behavior)

- [ ] **Step 6: Commit**

```bash
git add narro/utils/text_normalizer.py tests/test_tts_coverage.py
git commit -m "fix: stop colons->periods, drop URLs, fix newline period insertion"
```

### Task 12: Replace hallucination_detector with quality_check

**Files:**
- Modify: `narro/tts.py`
- Create: `tests/test_quality_check.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for quality_check — multi-signal failure detection."""
import torch
import pytest
from unittest.mock import patch

from narro.tts import Narro


@pytest.fixture
def tts():
    with patch('narro.tts.TransformersModel'), \
         patch('narro.tts.load_decoder'):
        n = Narro.__new__(Narro)
        n.device = 'cpu'
        return n


class TestQualityCheck:
    def test_passes_good_output(self, tts):
        response = {
            'hidden_state': torch.randn(50, 512),
            'token_entropy': torch.full((50,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A normal sentence.") is None

    def test_detects_repetition(self, tts):
        response = {
            'hidden_state': torch.zeros(50, 512),
            'token_entropy': torch.full((50,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A sentence.") == 'repetition'

    def test_detects_high_entropy(self, tts):
        response = {
            'hidden_state': torch.randn(50, 512),
            'token_entropy': torch.full((50,), 20.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A sentence.") == 'garbled'

    def test_detects_truncation(self, tts):
        response = {
            'hidden_state': torch.randn(50, 512),
            'token_entropy': torch.full((50,), 2.0),
            'finish_reason': 'length',
        }
        assert tts.quality_check(response, "A sentence.") == 'truncated'

    def test_detects_too_many_tokens(self, tts):
        response = {
            'hidden_state': torch.randn(200, 512),
            'token_entropy': torch.full((200,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "Short.") == 'length_anomaly'

    def test_detects_too_few_tokens(self, tts):
        response = {
            'hidden_state': torch.randn(1, 512),
            'token_entropy': torch.full((1,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A" * 100) == 'length_anomaly'


class TestRetriesDefault:
    def test_encode_batch_defaults_to_retries_1(self):
        """encode_batch should default to retries=1 (not 0)."""
        import inspect
        from narro.tts import Narro
        sig = inspect.signature(Narro.encode_batch)
        assert sig.parameters['retries'].default == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_quality_check.py -v`
Expected: FAIL — `Narro` has no attribute `quality_check`

- [ ] **Step 3: Implement quality_check**

In `narro/tts.py`:

1. Add threshold constants: `ENTROPY_THRESHOLD = 8.0`, `MAX_TOKEN_RATIO = 15.0`, `MIN_TOKEN_RATIO = 0.05`
2. Add `quality_check(self, response, input_text)` method that checks: repetition, entropy, length ratio, finish_reason
3. Extract existing repetition logic into `_detect_repetition(self, hidden_states)`
4. Update `encode_batch` to call `quality_check` instead of `hallucination_detector`
5. Change `retries` default from 0 to 1

- [ ] **Step 4: Run quality_check tests**

Run: `pytest tests/test_quality_check.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add narro/tts.py tests/test_quality_check.py
git commit -m "feat: replace hallucination_detector with multi-signal quality_check"
```

---

## Chunk 5: LLM Rewriting Layer

### Task 13: Create rewrite module

**Files:**
- Create: `narro/rewrite.py`
- Create: `tests/test_rewrite.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for narro.rewrite — LLM paragraph rewriting."""
from unittest.mock import patch, MagicMock

from narro.rewrite import rewrite_paragraphs


class TestRewriteParagraphs:
    def test_returns_same_number_of_paragraphs(self):
        paragraphs = ["Paragraph one.", "Paragraph two."]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Rewritten paragraph."}}]
        }

        with patch('narro.rewrite.requests.post', return_value=mock_resp):
            result = rewrite_paragraphs(paragraphs, api_url="http://localhost/v1")
            assert len(result) == 2

    def test_empty_input_returns_empty(self):
        result = rewrite_paragraphs([], api_url="http://localhost/v1")
        assert result == []

    def test_passes_api_key_in_header(self):
        paragraphs = ["Hello."]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hi."}}]
        }

        with patch('narro.rewrite.requests.post', return_value=mock_resp) as mock_post:
            rewrite_paragraphs(paragraphs, api_url="http://x/v1", api_key="sk-test")
            headers = mock_post.call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer sk-test"

    def test_falls_back_on_error(self):
        """If the LLM call fails, return the original paragraph."""
        paragraphs = ["Original text."]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Server error")

        with patch('narro.rewrite.requests.post', return_value=mock_resp):
            result = rewrite_paragraphs(paragraphs, api_url="http://localhost/v1")
            assert result == ["Original text."]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rewrite.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement rewrite module**

Create `narro/rewrite.py` with:
- `SYSTEM_PROMPT` — instructs LLM to rewrite for TTS narration
- `rewrite_paragraphs(paragraphs, api_url, api_key, model)` — calls `/v1/chat/completions` per paragraph, falls back to original on error

Uses `requests` (transitive dep via `huggingface_hub`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rewrite.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add narro/rewrite.py tests/test_rewrite.py
git commit -m "feat: add LLM paragraph rewriting module"
```

### Task 14: Add extract_paragraphs helper and wire --rewrite into Hugo CLI

**Files:**
- Modify: `narro/hugo/extract.py`, `narro/hugo/cli.py`, `narro/cli.py`
- Test: `tests/test_hugo_extract.py`

- [ ] **Step 1: Write the failing test for extract_paragraphs**

```python
from narro.hugo.extract import extract_paragraphs

class TestExtractParagraphs:
    def test_splits_on_double_newline(self):
        body = "First paragraph.\n\nSecond paragraph."
        result = extract_paragraphs(body)
        assert len(result) == 2
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."

    def test_strips_markdown(self):
        body = "## Heading\n\nParagraph text.\n\n```python\ncode\n```\n\nMore text."
        result = extract_paragraphs(body)
        assert all("##" not in p for p in result)
        assert all("```" not in p for p in result)

    def test_empty_body(self):
        assert extract_paragraphs("") == []

    def test_skips_empty_paragraphs(self):
        body = "Para one.\n\n\n\nPara two."
        result = extract_paragraphs(body)
        assert len(result) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hugo_extract.py::TestExtractParagraphs -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement extract_paragraphs in hugo/extract.py**

```python
def extract_paragraphs(body):
    """Extract speakable paragraphs from a markdown body.

    Calls extract_prose() then splits on paragraph boundaries.
    """
    prose = extract_prose(body)
    if not prose.strip():
        return []
    return [p.strip() for p in prose.split('\n\n') if p.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hugo_extract.py::TestExtractParagraphs -v`
Expected: PASS

- [ ] **Step 5: Add --rewrite flag to Hugo generate**

In `narro/cli.py`, add `--rewrite` to the Hugo generate subparser. In `narro/hugo/cli.py`, update `cmd_hugo_generate` to accept `rewrite` parameter. When enabled and `NARRO_LLM_URL` is set, call `rewrite_paragraphs` before `encode_batch`. Replace inline paragraph splitting with `extract_paragraphs`.

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add narro/hugo/extract.py narro/hugo/cli.py narro/cli.py \
        tests/test_hugo_extract.py
git commit -m "feat: add extract_paragraphs helper and --rewrite flag for Hugo"
```

---

## Chunk 6: CPU Micro-optimizations and Final Integration

### Task 15: CPU micro-optimizations

**Files:**
- Modify: `narro/tts.py`, `narro/backends/base.py`

- [ ] **Step 1: Update threading defaults in tts.py (spec 5a)**

Auto-tune `torch.set_num_threads` to `min(os.cpu_count() or 4, 8)` when `num_threads` is not explicitly set.

- [ ] **Step 2: Audit inference_mode usage (spec 5b)**

Verify all production code uses `torch.inference_mode()` not `torch.no_grad()`. Check `base.py` (infer, stream_infer), `decode_only.py` (decode), and `tts.py`. Convert any remaining `no_grad` to `inference_mode`.

- [ ] **Step 3: Add tighter max_new_tokens in base.py (spec 5d)**

In `BaseModel.infer`, after tokenization, compute: `prompt_token_count = inputs['input_ids'].shape[1]` and use `max_new_tokens = min(512, prompt_token_count * 8)` instead of hardcoded 512.

- [ ] **Step 4: Note torch.compile mode tuning for benchmark iteration (spec 5c, 5e)**

KV-cache prefix reuse (spec 5c) and torch.compile mode comparison (spec 5e) require running the benchmark framework on real hardware to evaluate. Add a comment in `narro/bench.py` documenting these as future optimization experiments once baseline benchmarks are collected. Do not implement speculatively.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add narro/tts.py narro/backends/base.py narro/bench.py
git commit -m "perf: auto-tune threading, tighter max_new_tokens, inference_mode audit"
```

### Task 15b: Remove old benchmarks/bench.py

**Files:**
- Remove: `benchmarks/bench.py`

- [ ] **Step 1: Delete the old benchmark file**

```bash
git rm benchmarks/bench.py
```

If the `benchmarks/` directory is now empty, remove it too.

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove old benchmarks/bench.py (replaced by narro/bench.py)"
```

### Task 16: Final integration test

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 2: Verify CLI help for all new subcommands**

Run: `python -m narro.cli --help`
Expected: Shows `speak`, `encode`, `decode`, `hugo`, `serve`, `bench`

Run: `python -m narro.cli serve --help`
Expected: Shows `--host`, `--port`, `--device`

Run: `python -m narro.cli bench --help`
Expected: Shows `--runs`, `--json`, `--device`

- [ ] **Step 3: Run test coverage**

Run: `pytest tests/ --cov=narro --cov-report=term-missing`
Review coverage for new files: `server.py`, `client.py`, `bench.py`, `rewrite.py`

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: final integration cleanup"
```
