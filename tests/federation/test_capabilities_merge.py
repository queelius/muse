from muse.federation.capabilities import merge_capabilities


def doc_with(mid, rev="1" * 40, loaded=False, depth=0, modality="audio/speech"):
    return {
        "object": "muse.capabilities", "server": {},
        "modalities": [{"modality": modality, "summary": "s", "models": [mid]}],
        "models": [{"id": mid, "modality": modality, "revision": rev,
                    "loaded": loaded, "queue_depth": depth}],
    }


def test_merge_unions_models_with_nodes():
    doc = merge_capabilities([
        ("a", doc_with("m", loaded=True, depth=1)),
        ("b", doc_with("m", depth=2)),
        ("c", doc_with("n")),
        ("d", None),
    ])
    assert [m["id"] for m in doc["models"]] == ["m", "n"]
    m = doc["models"][0]
    assert m["loaded"] is True and m["revision"] == "1" * 40
    assert [n["name"] for n in m["nodes"]] == ["a", "b"]
    assert doc["modalities"][0]["models"] == ["m", "n"]
    assert doc["unreachable_nodes"] == ["d"]


def test_disagreeing_revisions_both_shown():
    doc = merge_capabilities([("a", doc_with("m", rev="1" * 40)),
                              ("b", doc_with("m", rev="2" * 40))])
    m = doc["models"][0]
    assert m["revision"] is None
    assert {n["revision"] for n in m["nodes"]} == {"1" * 40, "2" * 40}


def test_garbage_entries_ignored():
    d = {"models": [1, {"id": 3}], "modalities": ["x"]}
    assert merge_capabilities([("a", d)])["models"] == []
