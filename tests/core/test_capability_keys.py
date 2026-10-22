import ast
from pathlib import Path

from muse.core.capability_keys import classify, public_capabilities
from muse.core.catalog_models import get_manifest, known_models
from muse.core.curated import load_curated


def test_public_and_hidden():
    assert public_capabilities({"supports_vision": True, "gguf_file": "x.gguf",
                                "voices": ["a"], "max_concurrency": 1,
                                "mystery": 1}) == \
        {"supports_vision": True, "voices": ["a"]}


def test_classify():
    assert classify("max_concurrency") == "hidden"
    assert classify("max_context_length") == "public"
    assert classify("nope") is None


def _is_caps_name(n):
    return isinstance(n, ast.Name) and ("capabilit" in n.id or n.id == "caps")


def _dict_keys(node):
    return {k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _resolver_keys() -> set[str]:
    """Statically collect capability keys the HF plugins synthesize."""
    root = Path(__file__).resolve().parents[2] / "src/muse/modalities"
    keys: set[str] = set()
    for path in root.glob("*/hf.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                if any(_is_caps_name(t) for t in node.targets):
                    keys |= _dict_keys(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Dict):
                if _is_caps_name(node.target):
                    keys |= _dict_keys(node.value)
            elif isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if (isinstance(k, ast.Constant) and k.value == "capabilities"
                            and isinstance(v, ast.Dict)):
                        keys |= _dict_keys(v)
            elif isinstance(node, ast.Subscript) and _is_caps_name(node.value):
                if (isinstance(node.ctx, ast.Store)
                        and isinstance(node.slice, ast.Constant)
                        and isinstance(node.slice.value, str)):
                    keys.add(node.slice.value)
            elif isinstance(node, ast.FunctionDef) and (
                    "capabilit" in node.name or "infer" in node.name):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                        keys |= _dict_keys(sub.value)
    return keys


def test_every_known_capability_key_is_classified():
    keys = set()
    for model_id, entry in known_models().items():
        if entry.bundled:
            keys |= set(get_manifest(model_id).get("capabilities") or {})
    for c in load_curated():
        keys |= set(c.capabilities)
    keys |= _resolver_keys()
    keys |= {"chat_format", "gguf_file", "supports_tools", "context_length",
             "max_context_length", "supports_vision", "supports_multi_image"}
    unclassified = sorted(k for k in keys if classify(k) is None)
    assert not unclassified, f"classify these in capability_keys.py: {unclassified}"
