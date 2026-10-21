"""Helpers shared by several modality packages.

Not a modality: discovery skips `_`-prefixed directories. Submodules
are imported explicitly (`muse.modalities._common.uploads`, ...) so
this package import stays dependency-free.
"""
