"""muse.core.gguf: names of every shard in a split GGUF."""
import pytest

from muse.core.gguf import gguf_shard_files


def test_single_file_is_its_own_set():
    assert gguf_shard_files("model-q4_k_m.gguf") == ("model-q4_k_m.gguf",)


def test_any_shard_names_the_whole_set_in_order():
    assert gguf_shard_files("Q8_0/M-Q8_0-00002-of-00003.gguf") == (
        "Q8_0/M-Q8_0-00001-of-00003.gguf",
        "Q8_0/M-Q8_0-00002-of-00003.gguf",
        "Q8_0/M-Q8_0-00003-of-00003.gguf",
    )


@pytest.mark.parametrize("name", [
    "M-00000-of-00002.gguf",
    "M-00003-of-00002.gguf",
    "M-00001-of-00000.gguf",
])
def test_impossible_shard_numbers_are_rejected(name):
    with pytest.raises(ValueError):
        gguf_shard_files(name)
