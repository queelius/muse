"""Paragraph-level alignment extraction.

Each sentence in an EncodedSpeech has a known token count, giving exact
duration (tokens * TOKEN_SIZE / SAMPLE_RATE). Paragraphs are emitted as
alignment entries with cumulative timestamps. No approximation needed.
"""


def extract_paragraph_alignment(encoded):
    """Extract paragraph-level alignment from an EncodedSpeech.

    Groups sentences by text_index (paragraph), computing cumulative
    timestamps across all sentences. Each paragraph gets one entry with
    the combined duration of its sentences.

    Args:
        encoded: EncodedSpeech instance where text_index identifies the
            source paragraph for each sentence.

    Returns:
        List of {"paragraph": int, "start": float, "end": float} with
        one entry per paragraph that has non-empty text, ordered
        chronologically.
    """
    from .tts import TOKEN_SIZE, SAMPLE_RATE

    token_duration = TOKEN_SIZE / SAMPLE_RATE
    cumulative_offset = 0.0
    paragraphs = {}  # text_index -> {"start": float, "end": float}

    for sentence in encoded.sentences:
        T = len(sentence.hidden_states)
        sentence_duration = T * token_duration
        ti = sentence.text_index

        if ti not in paragraphs:
            paragraphs[ti] = {
                'paragraph': ti,
                'start': round(cumulative_offset, 3),
                'end': None,
            }

        cumulative_offset += sentence_duration
        paragraphs[ti]['end'] = round(cumulative_offset, 3)

    return list(paragraphs.values())
