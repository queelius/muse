"""Validate every curated entry that declares image_processor_overrides.

Walks the curated catalog, finds entries with the override block, and
constructs DerivedImageProcessor against each. Fails fast if a curated
entry has malformed overrides (e.g., image_mean length mismatch).
"""
import pytest


def _curated_entries_with_overrides():
    from muse.core.curated import all_curated
    out = []
    for entry in all_curated():
        caps = entry.capabilities or {}
        overrides = caps.get("image_processor_overrides")
        if overrides:
            out.append((entry.id, overrides))
    return out


@pytest.mark.parametrize(
    "entry_id,overrides",
    _curated_entries_with_overrides(),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_curated_image_processor_overrides_construct(entry_id, overrides):
    """DerivedImageProcessor(**overrides) must construct without error
    for every curated entry that declares image_processor_overrides."""
    from muse.core.image_preprocessing import DerivedImageProcessor
    DerivedImageProcessor(
        num_channels=int(overrides.get("num_channels", 3)),
        image_size=overrides.get("image_size", 224),
        image_mean=overrides.get("image_mean"),
        image_std=overrides.get("image_std"),
    )


def test_texteller_curated_has_overrides():
    """TexTeller specifically should have image_processor_overrides
    declared with num_channels=1 and image_size=448 (its known shape)."""
    from muse.core.curated import find_curated
    entry = find_curated("texteller")
    assert entry is not None
    caps = entry.capabilities or {}
    overrides = caps.get("image_processor_overrides")
    assert overrides is not None
    assert overrides["num_channels"] == 1
    assert overrides["image_size"] == 448


def test_all_curated_includes_texteller():
    """all_curated() exposes the texteller entry (smoke-tests the new public API)."""
    from muse.core.curated import all_curated
    ids = {e.id for e in all_curated()}
    assert "texteller" in ids
