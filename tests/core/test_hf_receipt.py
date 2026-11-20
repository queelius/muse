"""One HF artifact-receipt item validator shared by resolve, pull and catalog."""
import pytest

from muse.core.hf_receipt import ReceiptError, validate_patterns, validate_receipt_item

REV = "a" * 40


def _item(**over):
    item = {"repo_id": "org/name", "revision": REV, "subdir": "."}
    item.update(over)
    return item


@pytest.mark.parametrize("subdir", ["", ".", "transformer", "vae_1.0"])
def test_accepts_known_subdirs(subdir):
    assert validate_receipt_item(_item(subdir=subdir), label="x")["subdir"] == subdir


@pytest.mark.parametrize("subdir", ["..", "a/b", "a\\b", ".hidden", "-x", "has space", 3])
def test_rejects_unsafe_subdir(subdir):
    with pytest.raises(ReceiptError, match="subdir"):
        validate_receipt_item(_item(subdir=subdir), label="x")


def test_dot_subdir_can_be_disallowed():
    with pytest.raises(ReceiptError, match="subdir"):
        validate_receipt_item(_item(subdir="."), label="x", allow_root=False)


@pytest.mark.parametrize("repo", ["noslash", "a/b/c", "a /b", None])
def test_rejects_bad_repo(repo):
    with pytest.raises(ReceiptError, match="repo_id"):
        validate_receipt_item(_item(repo_id=repo), label="x")


@pytest.mark.parametrize("rev", ["main", "A" * 40, REV + "\n", None])
def test_rejects_mutable_revision(rev):
    with pytest.raises(ReceiptError, match="revision"):
        validate_receipt_item(_item(revision=rev), label="x")


@pytest.mark.parametrize("field", ["allow_patterns", "required_patterns"])
@pytest.mark.parametrize("bad", [[], "x", [""], [1]])
def test_rejects_bad_patterns(field, bad):
    with pytest.raises(ReceiptError, match=field):
        validate_receipt_item(_item(**{field: bad}), label="x")


def test_null_patterns_are_absent():
    out = validate_receipt_item(_item(allow_patterns=None), label="x")
    assert "allow_patterns" not in out


def test_required_patterns_must_be_safe_relative():
    with pytest.raises(ReceiptError, match="unsafe relative pattern"):
        validate_receipt_item(_item(required_patterns=["../w"]), label="x")


def test_unknown_keys_rejected_only_on_request():
    validate_receipt_item(_item(extra=1), label="x")
    with pytest.raises(ReceiptError, match="unknown keys"):
        validate_receipt_item(_item(extra=1), label="x", reject_unknown=True)


def test_default_subdir():
    item = {"repo_id": "org/name", "revision": REV}
    with pytest.raises(ReceiptError, match="subdir"):
        validate_receipt_item(item, label="x")
    assert validate_receipt_item(item, label="x", default_subdir=".")["subdir"] == "."


def test_validate_patterns():
    assert validate_patterns(None, label="p") is None
    assert validate_patterns(["*.json"], label="p") == ("*.json",)
    with pytest.raises(ReceiptError):
        validate_patterns([], label="p")


def test_resolver_and_catalog_agree_on_empty_allow_patterns():
    """Regression: the resolver used to accept allow_patterns [] that the
    catalog then rejected at write time, after the download had run."""
    from muse.core import catalog_store
    from muse.core.resolvers import ResolvedModel, ResolverError
    from muse.core.resolvers_hf import _validated_hf_result

    resolved = ResolvedModel(
        manifest={"model_id": "m", "modality": "embedding/text",
                  "hf_repo": "org/name", "revision": REV},
        backend_path="x:Y", download=lambda cache_root: None,
        artifact_provenance=({"repo_id": "org/name", "revision": REV,
                              "subdir": ".", "allow_patterns": []},),
    )
    with pytest.raises(ResolverError, match="allow_patterns"):
        _validated_hf_result(resolved, repo_id="org/name", revision=REV)
    with pytest.raises(catalog_store._CatalogSchemaError, match="allow_patterns"):
        catalog_store._validate_artifact_receipt(
            [{"repo_id": "org/name", "revision": REV, "subdir": ".",
              "allow_patterns": []}], path="e")
