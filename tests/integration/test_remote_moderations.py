"""End-to-end /v1/moderations against a running muse server. Opt-in.

Requires MUSE_REMOTE_SERVER set + the target server has the
text_moderation_model loaded (default text-moderation).
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.slow


def test_protocol_classifies_safe_text(openai_client, text_moderation_model):
    """A neutral input must come back not-flagged.

    Asserts the actual contract, not just response shape. The previous
    version checked `hasattr(res, "flagged")` — true even when flagged
    was True, which masked the v0.14.0 KoalaAI/OK-label bug. The bug
    pattern: model assigns >0.99 confidence to "OK" on benign input,
    codec's argmax-above-threshold rule promoted it to flagged=True.
    The safe_labels capability now suppresses that. This test is the
    hard regression watchdog; without `assert flagged is False`, the
    bug could silently re-emerge.
    """
    r = openai_client.moderations.create(
        model=text_moderation_model,
        input="hello world, this is a friendly message",
    )
    assert len(r.results) == 1
    res = r.results[0]
    assert hasattr(res, "flagged")
    assert hasattr(res, "category_scores")
    assert res.flagged is False, (
        f"benign text was flagged: categories={res.categories}, "
        f"scores={res.category_scores}"
    )


def test_protocol_classifies_batch_returns_ordered_results(
    openai_client, text_moderation_model,
):
    r = openai_client.moderations.create(
        model=text_moderation_model,
        input=["hello world", "another safe text"],
    )
    assert len(r.results) == 2
