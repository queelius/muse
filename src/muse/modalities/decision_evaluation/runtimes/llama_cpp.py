"""DecisionLlamaCppModel: Decision-0.8B GGUF candidate scorer.

Each question is one forward pass over the published prompt format (see
`templates.py`). Instead of generating, the scorer reads the final
position's logits for every listed option letter through llama.cpp's
`llama_get_logits_ith(-1)` and softmaxes over exactly those candidates.
`Llama.scores` is not used: with `logits_all=False`, llama-cpp-python
0.3.33 never fills it.

Native state is reset (token count, KV and recurrent memory; the model is
a Qwen3.5 hybrid) before every question, so no question or request sees
another's context. A native failure marks the instance unusable.
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

from muse.core.placement import native_gpu_backend, native_gpu_offload, native_placement
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionInputError,
    DecisionRequest,
    PreparedBatch,
    PreparedQuestion,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.templates import (
    LABELS,
    candidate_token_ids,
    render_decision_tokens,
)


logger = logging.getLogger(__name__)

# The model's trained context; also the per-question input cap.
MAX_CONTEXT = 2048

# Module-level sentinel; real import deferred so discovery tolerates
# missing deps. Tests patch this attribute directly via the module path.
Llama: Any = None


def _ensure_deps() -> None:
    global Llama
    if Llama is not None:
        return
    try:
        from llama_cpp import Llama as _Llama
        Llama = _Llama
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-cpp-python unavailable: %s", e)


def final_candidate_logits(
    backend: object, candidate_token_ids: tuple[int, ...],
) -> tuple[float, ...]:
    """Copy the last evaluated position's logits for `candidate_token_ids`.

    backend is a llama_cpp.Llama that has just run `eval`. The token IDs
    name the candidates to read, not the prompt. Values are copied out
    before the caller resets native state.
    """
    n_vocab = backend.n_vocab()
    if any(not 0 <= t < n_vocab for t in candidate_token_ids):
        raise DecisionBackendError("invalid_candidates")
    row = backend._ctx.get_logits_ith(-1)
    if not row:
        raise DecisionBackendError("missing_logits")
    return tuple(float(row[t]) for t in candidate_token_ids)


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    if not all(math.isfinite(x) for x in logits):
        raise DecisionBackendError("invalid_distribution")
    top = max(logits)
    weights = [math.exp(x - top) for x in logits]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)


class _Prompt:
    """Adapter-private payload; `owner` ties a batch to its instance."""

    __slots__ = ("owner", "tokens")

    def __init__(self, owner: "DecisionLlamaCppModel", tokens: tuple[int, ...]) -> None:
        self.owner = owner
        self.tokens = tokens


class DecisionLlamaCppModel:
    """Decision GGUF backend implementing `DecisionModel`.

    Constructor kwargs mirror manifest capabilities (they arrive through
    load_backend's capabilities merge): gguf_file, device, n_gpu_layers,
    context_length (at most 2048), n_threads. Placement follows the same
    native rules as chat GGUFs: an explicit GPU device the native build
    cannot serve is an error, never a silent CPU fallback.
    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str,
        gguf_file: str,
        device: str = "auto",
        n_gpu_layers: int = -1,
        context_length: int = MAX_CONTEXT,
        n_threads: int = 2,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed; run `muse pull` or "
                "install `llama-cpp-python` into this venv"
            )
        if not 1 <= int(context_length) <= MAX_CONTEXT:
            raise ValueError(f"context_length must be 1-{MAX_CONTEXT}")
        if not local_dir:
            raise RuntimeError("local_dir is required; the GGUF file must be on disk")
        gguf_path = Path(local_dir) / gguf_file
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_file}")

        self.model_id = model_id
        self.hf_repo = hf_repo
        self.context_length = int(context_length)
        offload = native_gpu_offload() if device != "cpu" and n_gpu_layers != 0 else False
        placement = native_placement(
            device, n_gpu_layers, gpu_offload=offload,
            gpu_backend=native_gpu_backend() if offload else None,
        )
        self.device = placement.device
        self._lock = threading.Lock()
        self._unusable = False
        self._llama = Llama(
            model_path=str(gguf_path),
            n_ctx=self.context_length,
            n_gpu_layers=placement.n_gpu_layers,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            logits_all=False,
            verbose=False,
        )
        try:
            self._label_ids = candidate_token_ids(self._llama, len(LABELS))
        except Exception:
            self.close()
            raise

    def prepare(self, request: DecisionRequest) -> PreparedBatch:
        if request.model != self.model_id:
            raise DecisionBackendError("model_mismatch")
        self._check_usable()
        questions = []
        for i, (qid, question) in enumerate(request.questions):
            tokens = render_decision_tokens(request, qid, self._llama)
            if len(tokens) > self.context_length:
                raise DecisionInputError(
                    "input_too_long",
                    f"questions[{i}] renders to {len(tokens)} tokens; "
                    f"this model accepts at most {self.context_length}",
                )
            questions.append(PreparedQuestion(
                question_id=qid,
                keys=question.keys,
                payload=_Prompt(self, tokens),
                input_tokens=len(tokens),
            ))
        return PreparedBatch(
            model_id=self.model_id,
            questions=tuple(questions),
            input_tokens=sum(q.input_tokens for q in questions),
        )

    def evaluate(self, batch: PreparedBatch) -> DecisionBatch:
        if batch.model_id != self.model_id or any(
            not isinstance(q.payload, _Prompt) or q.payload.owner is not self
            for q in batch.questions
        ):
            raise DecisionBackendError("foreign_batch")
        with self._lock:
            self._check_usable()
            answers = []
            try:
                for q in batch.questions:
                    candidates = self._label_ids[: len(q.keys)]
                    self._reset()
                    self._llama.eval(q.payload.tokens)
                    logits = final_candidate_logits(self._llama, candidates)
                    answers.append(QuestionDistribution(
                        question_id=q.question_id,
                        probabilities=dict(zip(q.keys, _softmax(logits))),
                    ))
                self._reset()
            except DecisionBackendError as exc:
                # Non-finite logits are a bad answer, not bad native state
                # (every question starts from a reset); anything else is.
                if exc.code != "invalid_distribution":
                    self._unusable = True
                raise
            except Exception:
                self._unusable = True
                raise DecisionBackendError("inference_failed") from None
        return DecisionBatch(answers=tuple(answers), input_tokens=batch.input_tokens)

    def close(self) -> None:
        llama, self._llama = getattr(self, "_llama", None), None
        self._unusable = True
        if llama is not None and hasattr(llama, "close"):
            llama.close()

    def _reset(self) -> None:
        self._llama.reset()
        self._llama._ctx.kv_cache_clear()

    @property
    def usable(self) -> bool:
        """False once native state is uncertain; the worker's /health
        then reports 503 so the supervisor replaces the worker."""
        return not self._unusable

    def _check_usable(self) -> None:
        if self._unusable or self._llama is None:
            raise DecisionBackendError("backend_unusable")
