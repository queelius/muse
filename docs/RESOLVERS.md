# Resolvers

A **resolver** translates a URI like `hf://Qwen/Qwen3-8B-GGUF@q4_k_m`
into a `ResolvedModel` (synthesized MANIFEST + backend class path +
downloader function) so muse can pull the model without a hand-written
`.py` script. Resolvers also expose `search()` for discovering models.

This is the second extension surface in muse, alongside model scripts
(`docs/MODEL_SCRIPTS.md`):

- **Model scripts** are best when a model needs custom code (NV-Embed's
  custom encode method, Soprano's Narro engine).
- **Resolvers** are best when a class of models shares a uniform runtime
  (every GGUF file works with `LlamaCppModel`; every
  sentence-transformers repo works with `SentenceTransformerModel`).

## URI format

```
<scheme>://<source-specific-id>[@<variant>]
```

- `scheme`: registered resolver name (e.g. `hf`)
- `source-specific-id`: whatever the resolver expects (e.g. `org/repo` for HF)
- `@variant`: optional, used by resolvers that have multiple files per source
  (e.g. GGUF quantizations: `q4_k_m`, `q8_0`, `f16`)

Examples:

```
hf://Qwen/Qwen3-8B-GGUF@q4_k_m         # GGUF chat model
hf://Qwen/Qwen3-Embedding-0.6B          # sentence-transformers embedder
hf://sentence-transformers/all-MiniLM-L6-v2
```

## Currently registered: `hf`

The HuggingFace resolver lives at `muse.core.resolvers_hf`. It sniffs
each repo to decide how to handle it:

| Repo signal | Inferred modality | Runtime class |
|---|---|---|
| any `*.gguf` sibling | `chat/completion` | `LlamaCppModel` |
| `sentence-transformers` tag | `embedding/text` | `SentenceTransformerModel` |
| `sentence_transformers_config.json` sibling | `embedding/text` | `SentenceTransformerModel` |
| neither | (raises with tag list) | n/a |

For GGUF, `@variant` is **required**. There is no magic default
because picking one quantization for the user is rude (Q8_0 vs Q4_K_M
trades 4x VRAM against quality). Pulling without a variant raises with
the available list:

```
$ muse pull hf://Qwen/Qwen3-8B-GGUF
error: variant required for GGUF repo Qwen/Qwen3-8B-GGUF; available: ['q4_k_m', 'q5_k_m', 'q6_k', 'q8_0']
```

For sentence-transformers, repos are monolithic so `@variant` is not used.

### Capability sniffing

When resolving a GGUF, the resolver tries to sniff
`tokenizer_config.json` for tool-calling templates and writes
`capabilities.supports_tools: true|false|null` into the synthesized
manifest. Surfaced via `/v1/models`:

```json
{
  "id": "qwen3-8b-gguf-q4-k-m",
  "modality": "chat/completion",
  "supports_tools": true,
  "gguf_file": "qwen3-8b-q4_k_m.gguf",
  ...
}
```

This is a hint, not a gate: tool requests still flow to the model
regardless of the flag, because llama-cpp-python may handle some
tool-template variants the resolver's regex misses.

## Searching

```
muse search <query> [--modality {chat/completion, embedding/text}]
                    [--limit N] [--sort {downloads, lastModified, likes}]
                    [--max-size-gb F] [--backend B]
```

Queries the HF Hub via `HfApi.list_models` with the appropriate filter,
prints a compact aligned table:

```
$ muse search qwen3 --modality chat/completion --max-size-gb 10 --limit 5
  hf://Qwen/Qwen3-8B-GGUF@q4_k_m                              4.5 GB  dl=    1,234,567  apache-2.0       Qwen/Qwen3-8B-GGUF (q4_k_m)
  hf://Qwen/Qwen3-8B-GGUF@q5_k_m                              5.6 GB  dl=    1,234,567  apache-2.0       Qwen/Qwen3-8B-GGUF (q5_k_m)
  ...
```

For GGUF results, **each variant in a repo is a separate row** so the
user can see sizes and pick by hand. For sentence-transformers, one
row per repo (no variants).

## Pulling

```
muse pull <uri>
```

Routes the URI through the matching resolver:

1. `resolve(uri)` returns a `ResolvedModel` with the synthesized MANIFEST.
2. Per-model venv created at `~/.muse/venvs/<model_id>/`.
3. `pip install muse[server]` (editable) + `pip install <manifest.pip_extras>`.
4. `resolved.download(weights_cache)` fetches the weights (snapshot_download
   with appropriate `allow_patterns` for GGUF; full repo for sentence-transformers).
5. Synthesized MANIFEST + `source: <uri>` persisted in `~/.muse/catalog.json`.
6. The `known_models` cache is invalidated so the new entry is visible
   on the next call (no restart needed for that part; `muse serve`
   still needs a restart to load the new model into a worker).

Bundled scripts always win on `model_id` collision: a malicious or
careless resolver pull cannot silently shadow a bundled script that
ships with muse.

## Writing a new resolver

A resolver is a class subclassing `muse.core.resolvers.Resolver`:

```python
# muse/core/resolvers_civitai.py (example, not real)
from muse.core.resolvers import (
    Resolver, ResolvedModel, SearchResult, register_resolver,
)


class CivitaiResolver(Resolver):
    scheme = "civitai"

    def resolve(self, uri):
        # parse uri, hit civitai API, synthesize MANIFEST
        return ResolvedModel(
            manifest={...},
            backend_path="muse.modalities.image_generation.runtimes.diffusers:DiffusersModel",
            download=lambda cache: snapshot_download_civitai(...),
        )

    def search(self, query, **filters):
        for hit in civitai_api.search(query):
            yield SearchResult(
                uri=f"civitai://{hit.id}",
                model_id=...,
                modality="image/generation",
                size_gb=hit.size_gb,
                downloads=hit.downloads,
                license=hit.license,
                description=hit.name,
            )


register_resolver(CivitaiResolver())
```

Wire it into the CLI by adding a one-line import in `muse/cli.py`'s
`_cmd_pull` (for the URI dispatch) and `_cmd_search` (for the search
backend choice). Future work may auto-discover resolvers via a
`$MUSE_RESOLVERS_DIR` env var or an entry-point.

## Implementation notes

- Resolvers are instantiated and registered eagerly at import time.
  `from muse.core import resolvers_hf` is enough to wire `hf://`.
- The CLI lazy-imports each resolver only when needed so `muse --help`
  stays fast.
- `register_resolver()` overwrites the prior registration for the same
  scheme. Useful for tests; in production the import order determines
  which resolver wins (last import for a given scheme is used).
- `parse_uri()` handles the `@variant` split using `rsplit`, so URIs
  with `@` in the path (rare) are tolerated as long as the variant is
  the final `@`-segment.
