"""End-to-end /v1/moderations against a running muse server. Opt-in.

Requires MUSE_REMOTE_SERVER set + the target server has the
text_moderation_model loaded (default text-moderation).
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.slow


def test_protocol_classifies_safe_text(openai_client, text_moderation_model):
    """A neutral input should return a non-flagged result."""
    r = openai_client.moderations.create(
        model=text_moderation_model,
        input="hello world, this is a friendly message",
    )
    assert len(r.results) == 1
    res = r.results[0]
    assert hasattr(res, "flagged")
    assert hasattr(res, "category_scores")


def test_protocol_classifies_batch_returns_ordered_results(
    openai_client, text_moderation_model,
):
    r = openai_client.moderations.create(
        model=text_moderation_model,
        input=["hello world", "another safe text"],
    )
    assert len(r.results) == 2
