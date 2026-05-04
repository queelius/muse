"""Generic runtimes for image/ocr.

One runtime so far:

  HFVision2SeqRuntime
    Wraps transformers.AutoModelForVision2Seq + AutoProcessor for any
    vision-encoder + text-decoder OCR model (TrOCR, Nougat, TexTeller,
    GOT-OCR). Resolved by the HF plugin when the repo's tags include
    `image-to-text`.

Future runtimes (if Donut JSON parsing or DocVQA modalities land) live
beside this one. The HF plugin's _resolve dispatches per-architecture.
"""
from muse.modalities.image_ocr.runtimes.hf_vision2seq import (
    HFVision2SeqRuntime,
)


__all__ = ["HFVision2SeqRuntime"]
