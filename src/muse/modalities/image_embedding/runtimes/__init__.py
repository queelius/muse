"""Runtimes for image/embedding (generic, not bundled).

The single runtime here, `ImageEmbeddingRuntime`, wraps any HF
image-feature-extraction repo via `transformers.AutoModel` +
`AutoProcessor`. Resolver-pulled models point their `backend_path` at
this class; bundled scripts wrap transformers directly.
"""
