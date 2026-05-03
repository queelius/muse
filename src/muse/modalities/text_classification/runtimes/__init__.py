"""Generic runtimes for text/classification.

Two runtimes coexist on this modality:

  HFTextClassifier
    Wraps AutoModelForSequenceClassification + AutoTokenizer for any
    fine-tuned classifier head (sentiment, intent, toxicity, etc.).
    Resolved by the HF plugin when the repo's tags include
    `text-classification` (catch-all).

  HFZeroShotPipeline
    Wraps `transformers.pipeline("zero-shot-classification")` for any
    NLI head. Resolved by the HF plugin when the repo's tags include
    `zero-shot-classification` (or the repo name suggests it).

Both return list[ClassificationResult] so the codec layer treats them
uniformly. Capability flags on the manifest (supports_classification,
supports_zero_shot) gate which route methods accept the model.
"""
from muse.modalities.text_classification.runtimes.hf_text_classifier import (
    HFTextClassifier,
)
from muse.modalities.text_classification.runtimes.hf_zero_shot import (
    HFZeroShotPipeline,
)


__all__ = ["HFTextClassifier", "HFZeroShotPipeline"]
