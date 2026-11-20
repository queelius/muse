from muse.core.hf_repo import catalog_entries_for_repo, repo_key


def test_repo_key_normalizes():
    assert repo_key("hf://Org/Name@abc") == "org/name"
    assert repo_key(" Org/Name/ ") == "org/name"
    assert repo_key("bare-id") is None
    assert repo_key(None) is None
    assert repo_key(42) is None


def test_catalog_entries_for_repo_matches_top_level_and_manifest():
    catalog = {
        "b": {"manifest": {"hf_repo": "Org/Name"}},
        "a": {"hf_repo": "org/name@main"},
        "c": {"hf_repo": "org/other"},
        "d": "not-a-dict",
    }
    assert [mid for mid, _ in catalog_entries_for_repo("ORG/NAME", catalog)] == ["a", "b"]
    assert catalog_entries_for_repo("bare", catalog) == []
