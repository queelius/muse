"""Ask a muse decision model typed questions about one piece of text.

    muse pull decima-base          # or decision-0.8b-q8
    muse serve
    python examples/decisions.py decima-base

Prints each answer and its full probability distribution. Confidence
measures how concentrated a distribution is, not whether it is right.
"""
from __future__ import annotations

import json
import sys

from muse.modalities.decision_evaluation import DecisionClient


def main(model: str) -> None:
    out = DecisionClient().decide(
        model=model,
        state="Customer: my card was charged twice for this month's plan. Please fix it today.",
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges and refunds", "technical": "Software faults",
                             "sales": "New purchases and upgrades"},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is the request?",
                "criteria": ["Routine", "Needs a prompt reply", "Immediate action"],
            },
            "duplicate": {"type": "noul", "instructions": "The customer reports a duplicate charge."},
        },
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "decima-base")
