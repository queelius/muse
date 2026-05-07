"""Curated VLM entries: ids, shapes, and capability flags."""


def test_curated_smolvlm_instruct_resolves():
    from muse.core.curated import find_curated
    entry = find_curated("smolvlm-instruct")
    assert entry is not None
    assert entry.uri == "hf://HuggingFaceTB/SmolVLM-Instruct"
    assert entry.modality == "chat/completion"
    caps = entry.capabilities
    assert caps["supports_vision"] is True
    assert caps["supports_multi_image"] is True


def test_curated_qwen2_vl_2b_resolves():
    from muse.core.curated import find_curated
    entry = find_curated("qwen2-vl-2b-instruct")
    assert entry is not None
    assert entry.uri == "hf://Qwen/Qwen2-VL-2B-Instruct"
    assert entry.capabilities["supports_multi_image"] is True


def test_curated_llava_15_7b_single_image():
    from muse.core.curated import find_curated
    entry = find_curated("llava-1.5-7b")
    assert entry is not None
    assert entry.capabilities["supports_multi_image"] is False
    assert entry.capabilities["device"] == "cuda"


def test_curated_qwen2_vl_7b_resolves():
    from muse.core.curated import find_curated
    entry = find_curated("qwen2-vl-7b-instruct")
    assert entry is not None
    assert entry.capabilities["device"] == "cuda"
