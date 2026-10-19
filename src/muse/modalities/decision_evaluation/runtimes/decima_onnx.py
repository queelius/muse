"""DecimaOnnxModel: Decima-base INT8 ONNX decision scorer (CPU only).

Decima is a late-interaction scorer: the state (with the question) is
encoded once, each option is encoded separately, and a small scorer
returns one logit per option plus per-option features for the ordinal
Score head. One fitted temperature divides every logit.

The pinned `decima-ai` 2.2.0 runtime supplies the ONNX sessions, the
tokenizer, the export config and the ordinal head. Muse does NOT call its
`decide*` / System One entry points, because they:
  - tokenize with `truncation=True`, silently cutting long input;
  - shorten the question copy inside each option ("fit" rendering);
  - may download a snapshot or chunk long documents.
Instead Muse renders the same text the SDK's System One mapping would
(see `_render`), tokenizes it itself, rejects anything over the export's
limits (512 state+question tokens, 64 tokens per rendered option) before
any inference, and feeds exactly those tokens to the sessions.

Only local files are read: `<local_dir>/onnx/int8/`.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionInputError,
    DecisionQuestion,
    DecisionRequest,
    PreparedBatch,
    PreparedQuestion,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.validation import NOUL_DEFAULTS


logger = logging.getLogger(__name__)

EXPORT_SUBDIR = ("onnx", "int8")
EXPORT_FILES = ("decima.json", "encoder.onnx", "scorer.onnx", "tokenizer")
# How Muse question types map onto Decima's heads (decima.systemone).
_HEAD = {"choice": "choose", "noul": "verify", "score": "score"}
# The verify head scores the options in this order; P(yes) is P(true).
_VERIFY_OPTIONS = ("yes", "no")

# Module-level sentinels; real imports deferred so discovery tolerates
# missing deps. Tests patch these attributes directly via the module path.
DecimaOnnx: Any = None
normalize: Any = None
np: Any = None


def _ensure_deps() -> None:
    global DecimaOnnx, normalize, np
    if DecimaOnnx is not None and normalize is not None and np is not None:
        return
    try:
        import numpy as _np
        from decima.normalize import normalize as _normalize
        from decima.runtime import DecimaOnnx as _DecimaOnnx
        DecimaOnnx, normalize, np = _DecimaOnnx, _normalize, _np
    except Exception as e:  # noqa: BLE001
        logger.debug("decima-ai unavailable: %s", e)


def _state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _render(question: DecisionQuestion) -> tuple[str, list[str]]:
    """(question text, option texts) as decima.systemone maps them.

    Choice options are "key: description" (the key alone without one).
    Noul is a statement plus "True if / False if" lines when the caller
    gave descriptions, scored over yes/no. Score options are the levels.
    """
    if question.kind == "choice":
        options = [f"{k}: {d}" if d and d.strip() else k for k, d in question.criteria]
        return question.instructions, options
    if question.kind == "score":
        return question.instructions, [d for _, d in question.criteria]
    given = dict(question.criteria)
    defaults = dict(NOUL_DEFAULTS)
    tr, fa = (
        "" if given[k] == defaults[k] else given[k].strip() for k in ("true", "false")
    )
    text = question.instructions
    if tr or fa:
        text = f"{text}\nTrue if: {tr or '—'}\nFalse if: {fa or '—'}"
    return text, list(_VERIFY_OPTIONS)


class _Prepared:
    """Adapter-private payload; `owner` ties a batch to its instance."""

    __slots__ = ("owner", "head", "state_ids", "choice_rows")

    def __init__(self, owner, head, state_ids, choice_rows):
        self.owner = owner
        self.head = head
        self.state_ids = state_ids
        self.choice_rows = choice_rows


class DecimaOnnxModel:
    """Decima backend implementing `DecisionModel`.

    Constructor kwargs mirror manifest capabilities: device (cpu only),
    n_threads (default 2), option_cache_size (default 256 option sets),
    temperature (the expected fitted value; a mismatching export is
    refused rather than served with another calibration).
    """

    model_id: str

    def __init__(
        self,
        *,
        model_id: str,
        hf_repo: str,
        local_dir: str,
        device: str = "cpu",
        n_threads: int = 2,
        option_cache_size: int = 256,
        temperature: float | None = None,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if DecimaOnnx is None:
            raise RuntimeError(
                "decima-ai is not installed; run `muse pull` or install "
                "`decima-ai` and `onnxruntime` into this venv"
            )
        if str(device or "cpu").lower() not in ("cpu", "auto"):
            raise RuntimeError("Decima's INT8 ONNX export runs on CPU only")
        if int(n_threads) < 1 or int(option_cache_size) < 1:
            raise ValueError("n_threads and option_cache_size must be positive")
        if not local_dir:
            raise RuntimeError("local_dir is required; the ONNX export must be on disk")
        export = Path(local_dir).joinpath(*EXPORT_SUBDIR)
        missing = [name for name in EXPORT_FILES if not (export / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Decima INT8 export is incomplete (missing {', '.join(missing)}); re-pull"
            )

        self.model_id = model_id
        self.hf_repo = hf_repo
        self.device = "cpu"
        self._lock = threading.Lock()
        self._unusable = False
        self._cache: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
        self._cache_size = int(option_cache_size)
        self._rt = DecimaOnnx(export, threads=int(n_threads))
        try:
            cfg = self._rt.cfg
            self.temperature = float(cfg["temperature"])
            if not (math.isfinite(self.temperature) and self.temperature > 0):
                raise ValueError("export temperature is invalid")
            if temperature is not None and self.temperature != float(temperature):
                raise ValueError("export temperature does not match the pinned calibration")
            self.max_state_tokens = int(cfg["max_state_tokens"])
            self.max_choice_tokens = int(cfg["max_choice_tokens"])
            self._question_in_state = bool(cfg.get("question_in_state", False))
            self._question_in_choices = bool(cfg.get("question_in_choices", True))
            self._state_prefix = str(cfg.get("state_prefix", "query: "))
            self._choice_prefix = str(cfg.get("choice_prefix", "passage: "))
            self._pad = self._rt.tok.pad_token_id or 0
        except Exception:
            self.close()
            raise

    # -- preparation (tokenization only) -----------------------------------

    def _ids(self, text: str) -> tuple[int, ...]:
        return tuple(int(t) for t in self._rt.tok(text)["input_ids"])

    def prepare(self, request: DecisionRequest) -> PreparedBatch:
        if request.model != self.model_id:
            raise DecisionBackendError("model_mismatch")
        self._check_usable()
        state = _state_text(request.state)
        questions = []
        for i, (qid, question) in enumerate(request.questions):
            text, options = _render(question)
            body = f"{text}\n{state}" if self._question_in_state else state
            state_ids = self._ids(self._state_prefix + normalize(body, "en"))
            if len(state_ids) > self.max_state_tokens:
                raise DecisionInputError(
                    "input_too_long",
                    f"questions[{i}] state and question render to {len(state_ids)} "
                    f"tokens; this model accepts at most {self.max_state_tokens}",
                )
            head_text = text if self._question_in_choices else ""
            rows = []
            for j, option in enumerate(options):
                ids = self._ids(
                    self._choice_prefix + normalize(f"{head_text} {option}".strip(), "en")
                )
                if len(ids) > self.max_choice_tokens:
                    raise DecisionInputError(
                        "input_too_long",
                        f"questions[{i}] option {j} renders to {len(ids)} tokens "
                        f"with its question; this model accepts at most "
                        f"{self.max_choice_tokens}",
                    )
                rows.append(ids)
            questions.append(PreparedQuestion(
                question_id=qid,
                keys=question.keys,
                payload=_Prepared(self, _HEAD[question.kind], state_ids, tuple(rows)),
                input_tokens=len(state_ids) + sum(len(r) for r in rows),
            ))
        return PreparedBatch(
            model_id=self.model_id,
            questions=tuple(questions),
            input_tokens=sum(q.input_tokens for q in questions),
        )

    # -- inference ---------------------------------------------------------

    def _encode(self, rows: tuple[tuple[int, ...], ...]):
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), self._pad, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for k, row in enumerate(rows):
            ids[k, : len(row)] = row
            mask[k, : len(row)] = 1
        hidden = self._rt.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        return hidden, mask

    def _options(self, rows: tuple[tuple[int, ...], ...]):
        cached = self._cache.get(rows)
        if cached is not None:
            self._cache.move_to_end(rows)
            return cached
        encoded = self._encode(rows)
        self._cache[rows] = encoded
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return encoded

    def _probabilities(self, p: _Prepared) -> list[float]:
        h, m = self._encode((p.state_ids,))
        ch, cm = self._options(p.choice_rows)
        s, z = self._rt.sc.run(
            None, {"state_h": h, "state_mask": m, "choice_h": ch, "choice_mask": cm},
        )
        s = np.asarray(s, dtype=np.float64).reshape(-1) / self.temperature
        if s.shape[0] != len(p.choice_rows):
            raise DecisionBackendError("invalid_distribution")
        if p.head == "score":
            logp = np.asarray(self._rt.ordinal(s, z), dtype=np.float64).reshape(-1)
        else:
            logp = s
        if logp.shape[0] != len(p.choice_rows) or not np.all(np.isfinite(logp)):
            raise DecisionBackendError("invalid_distribution")
        e = np.exp(logp - logp.max())
        return [float(x) for x in e / e.sum()]

    def evaluate(self, batch: PreparedBatch) -> DecisionBatch:
        if batch.model_id != self.model_id or any(
            not isinstance(q.payload, _Prepared) or q.payload.owner is not self
            for q in batch.questions
        ):
            raise DecisionBackendError("foreign_batch")
        with self._lock:
            self._check_usable()
            answers = []
            try:
                for q in batch.questions:
                    probs = self._probabilities(q.payload)
                    answers.append(QuestionDistribution(
                        question_id=q.question_id,
                        probabilities=dict(zip(q.keys, probs)),
                    ))
            except DecisionBackendError as exc:
                if exc.code != "invalid_distribution":
                    self._unusable = True
                raise
            except Exception:
                self._unusable = True
                raise DecisionBackendError("inference_failed") from None
        return DecisionBatch(answers=tuple(answers), input_tokens=batch.input_tokens)

    def close(self) -> None:
        self._rt = None
        self._cache.clear()
        self._unusable = True

    def _check_usable(self) -> None:
        if self._unusable or self._rt is None:
            raise DecisionBackendError("backend_unusable")
