"""Image encoding for the image/upscale modality.

Re-exports the same to_bytes / to_data_url helpers used by
image/generation. Kept as a thin module rather than a duplicate so
the modality is self-describing without code duplication. If a future
upscale-specific encoder lands (e.g. a "preserve original alpha"
mode), it slots in here.
"""
from muse.modalities.image_generation.codec import to_bytes, to_data_url

__all__ = ["to_bytes", "to_data_url"]
