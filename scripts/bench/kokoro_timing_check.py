#!/usr/bin/env python3
"""One-off narration gate: Kokoro 0.9.4 timings vs Muse's forced aligner.

Run with ~/.muse/venvs/kokoro-82m/bin/python, --beats FILE --out SCRATCH_DIR.
Reads ~/.muse/catalog.json and local weights/voices only. No admin calls.
Writes WAVs, raw aligner responses, native tokens/chunks and results.json.
--analyze-only regenerates results from those saved takes without inference.
Heavy imports are confined to measurement; text matching/statistics are standalone.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import unicodedata


SAMPLE_RATE = 24000


def normalize(word: str) -> str:
    """Ignore case, accents and punctuation; preserve letters and digits."""
    return ''.join(c for c in unicodedata.normalize('NFKD', word.casefold())
                   if c.isalnum())


def compare(take: dict) -> dict:
    """Pair equal normalized words monotonically, reporting every unmatched word."""
    native = [w for w in take['tokens'] if normalize(w['word'])]
    aligned = [w for w in take['alignment']['words'] if normalize(w['word'])]
    matcher = SequenceMatcher(None, [normalize(w['word']) for w in native],
                              [normalize(w['word']) for w in aligned], autojunk=False)
    pairs, unmatched = [], []
    for op, i, j, a, b in matcher.get_opcodes():
        if op == 'equal':
            for k, q in zip(range(i, j), range(a, b)):
                n, r = native[k], aligned[q]
                pairs.append(dict(word=n['word'], kokoro=n['start'], aligner=r['start'],
                                  delta=n['start'] - r['start'], chunk=n['chunk'],
                                  chunk_word=n['chunk_word']))
        else:
            unmatched.extend(dict(source='kokoro', **w) for w in native[i:j])
            unmatched.extend(dict(source='aligner', **w) for w in aligned[a:b])
    # Every unmatched word on either side counts as a non-agreement, never as
    # evidence in favour of the gate. Matched-only coverage is also reported.
    return dict(id=take['beat']['id'], voice=take['beat']['voice'], wav=take['wav'],
                chunks=take['chunks'], pairs=pairs, unmatched=unmatched)


def stats(deltas: list[float], denominator: int | None = None) -> dict:
    """Linear-interpolated p95; signed and absolute errors are both explicit."""
    if not deltas:
        return dict(matched=0, denominator=denominator or 0, within_100ms=0.0,
                    median_signed=None, p95_signed=None, median_absolute=None,
                    p95_absolute=None, mean_signed=None)

    def p95(values: list[float]) -> float:
        values = sorted(values)
        index = (len(values) - 1) * .95
        lo = math.floor(index)
        hi = math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    absolute = [abs(d) for d in deltas]
    count = len(deltas) if denominator is None else denominator
    return dict(matched=len(deltas), denominator=count,
                # Subtracting decimal timestamps can put exact 100ms errors
                # a few floating-point ulps above the inclusive threshold.
                within_100ms=sum(d <= .1 + 1e-9 for d in absolute) / count,
                median_signed=statistics.median(deltas), p95_signed=p95(deltas),
                median_absolute=statistics.median(absolute), p95_absolute=p95(absolute),
                mean_signed=statistics.mean(deltas))


def analyze(takes: list[dict]) -> dict:
    comparisons = [compare(t) for t in takes]
    pairs = [p for c in comparisons for p in c['pairs']]
    deltas = [p['delta'] for p in pairs]
    denominator = len(pairs) + sum(len(c['unmatched']) for c in comparisons)
    raw = stats(deltas, denominator)
    mean = raw['mean_signed']
    if mean is None:
        raise ValueError('No matched words; cannot measure timing agreement')
    corrected = stats([d - mean for d in deltas], denominator)
    decision = ('PASS_NATIVE' if raw['within_100ms'] >= .95 else
                'PASS_CONSTANT' if corrected['within_100ms'] >= .95 else 'STOP')
    groups = {
        'first_chunk': [p for p in pairs if p['chunk'] == 0],
        'later_chunks': [p for p in pairs if p['chunk'] > 0],
        'first_three_words_after_boundary': [p for p in pairs
                                             if p['chunk'] > 0 and p['chunk_word'] < 3],
    }
    return dict(decision=decision,
                offset_seconds=0.0 if decision == 'PASS_NATIVE' else
                -mean if decision == 'PASS_CONSTANT' else None,
                constant_candidate_seconds=-mean, raw=raw, corrected=corrected,
                matched_only_raw=stats(deltas),
                matched_only_corrected=stats([d - mean for d in deltas]),
                boundary_groups={k: dict(raw=stats([p['delta'] for p in v]),
                                        corrected=stats([p['delta'] - mean for p in v]))
                                 for k, v in groups.items()}, comparisons=comparisons)


def measure(beats: list[dict], out: Path, server: str, device: str) -> list[dict]:
    """Synthesize locally, accumulating sample offsets, then align each WAV."""
    # Prevent Hub/network fallbacks; all model artifacts must already exist.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import httpx
    import numpy as np
    import soundfile as sf
    from kokoro import KModel, KPipeline

    catalog = json.loads(Path('~/.muse/catalog.json').expanduser().read_text())
    root = Path(catalog['kokoro-82m']['local_dir']).expanduser()
    for path in [root / 'config.json', root / 'kokoro-v1_0.pth',
                 *(root / 'voices' / f"{b['voice']}.pt" for b in beats)]:
        if not path.is_file():
            raise FileNotFoundError(path)
    model = KModel(config=str(root / 'config.json'),
                   model=str(root / 'kokoro-v1_0.pth')).to(device).eval()
    pipeline = KPipeline(lang_code='a', model=model, device=device)
    versions = {name: importlib.metadata.version(name) for name in ('kokoro', 'misaki')}
    takes = []
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        for index, beat in enumerate(beats):
            stem = out / f"{index:03d}-{beat['voice']}"
            audio, tokens, chunks = [], [], []
            samples = 0
            voice = str(root / 'voices' / f"{beat['voice']}.pt")
            for chunk_index, result in enumerate(pipeline(beat['text'], voice=voice)):
                if result.audio is None:
                    raise ValueError(f"Missing audio: {beat['id']}, chunk {chunk_index}")
                chunk = result.audio.detach().cpu().numpy()
                offset = samples / SAMPLE_RATE
                chunks.append(dict(index=chunk_index, offset=offset, samples=len(chunk),
                                   text=result.graphemes))
                lexical_index = 0
                for token in result.tokens or []:
                    if token.start_ts is not None:
                        tokens.append(dict(word=token.text, start=token.start_ts + offset,
                                           end=None if token.end_ts is None else token.end_ts + offset,
                                           chunk=chunk_index, chunk_word=lexical_index))
                        if normalize(token.text):
                            lexical_index += 1
                audio.append(chunk)
                samples += len(chunk)
            wav = stem.with_suffix('.wav')
            sf.write(wav, np.concatenate(audio), SAMPLE_RATE, subtype='PCM_16')
            with wav.open('rb') as f:
                response = client.post(server.rstrip('/') + '/v1/audio/alignments',
                                       files={'file': (wav.name, f, 'audio/wav')},
                                       data={'text': beat['text'], 'language': 'English',
                                             'model': 'qwen3-forced-aligner-0.6b'})
            stem.with_suffix('.alignment.json').write_text(response.text)
            response.raise_for_status()
            take = dict(beat=beat, wav=str(wav), sample_rate=SAMPLE_RATE,
                        duration_seconds=samples / SAMPLE_RATE, chunks=chunks,
                        tokens=tokens, alignment=response.json(), versions=versions,
                        local_dir=str(root), device=device, server=server,
                        lang_code='a', speed=1.0)
            stem.with_suffix('.take.json').write_text(json.dumps(take, indent=2))
            takes.append(take)
            print(f"{index + 1}/{len(beats)} {beat['id']} {beat['voice']}: "
                  f"{len(chunks)} chunks, {samples / SAMPLE_RATE:.2f}s", flush=True)
    return takes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--beats', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server', default=os.environ.get('MUSE_SERVER', 'http://localhost:8000'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    import yaml

    beats = yaml.safe_load(args.beats.read_text())
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        takes = [json.loads((out / f"{i:03d}-{b['voice']}.take.json").read_text())
                 for i, b in enumerate(beats)]
        if [t['beat'] for t in takes] != beats:
            raise ValueError('Saved takes do not match the frozen beat list')
    else:
        takes = measure(beats, out, args.server, args.device)
    result = analyze(takes)
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'comparisons'}, indent=2))


if __name__ == '__main__':
    main()
