KOKORO_TIMING_OFFSET_SECONDS = 0.0
Decision: PASS_NATIVE; raw 99.122%, constant-corrected 99.785% within 0.1 s.
Per-chunk correction needed: no. Uncorrected later chunks pass at 98.921%; the first three words after each boundary pass at 100.000%.
Measured constant candidate: +0.022999338 s added to Kokoro starts; unused because uncorrected agreement already exceeds 95%.
The aligner reports starts on an 80 ms grid, so this result shows no systematic offset or chunk drift rather than sub-0.1 s accuracy; the conclusion assumes plain concatenation with no inter-chunk silence, speed 1.0, lang_code "a".

# Kokoro timing agreement — 2026-10-24

This is Task 1’s one-off gate, with no speech API changes. Disagreement does not identify which model is wrong. No listening adjudication or timing repair was performed. Do not start Task 2 on a STOP result.

## Sample and reproduction

Measured 50 takes, 40 distinct beats from 8 reels, 6038 whitespace-separated words including repeat takes, 1660.025 seconds of audio. 37 takes split into multiple chunks (90 chunks total).

The committed [beat list](2026-10-24-kokoro-timing-beats.yaml) freezes `{id, text, voice}` in measurement order. IDs are content-relative reel directories plus zero-based beat indices. The first 40 beats use `voices[voice]` from their own `reel.yaml`; the last ten repeat the first five beats with `af_heart` and `am_puck`. Selection used the first five markup-free beats from each of eight reels, fixed before seeing timing results. All text is the actual `beats[].text` narrated by `engine/companion/reel.py` and `prosody.Beat`; these beats have no speech-only override. No markup links or inline prosody tags were stripped or inserted. Long text was left intact to trigger natural chunking. No lexicon, trimming, inserted silence, encoding delay correction, or speed change was applied.

Runtime: `{'kokoro': '0.9.4', 'misaki': '0.9.4'}`, device `cuda`, `KPipeline(lang_code="a")` for every voice, speed 1.0, mono 24,000 Hz PCM16 WAV. British voice packs are deliberately run through the required American English pipeline. Local weights: `/home/spinoza/.muse/weights/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987`. Kokoro model and voice packs were loaded by local filename with Hub offline mode; no weights were downloaded. Aligner: live Muse release 0.68.0 at `http://localhost:8000`, `qwen3-forced-aligner-0.6b`, language `English`; only inference POSTs to `/v1/audio/alignments`.

```bash
/home/spinoza/.muse/venvs/kokoro-82m/bin/python scripts/bench/kokoro_timing_check.py \
  --beats docs/benchmarks/2026-10-24-kokoro-timing-beats.yaml \
  --out .superpowers/sdd/2026-10-24-narration/kokoro-check
# Recompute from saved raw takes without any model or server access:
/home/spinoza/.muse/venvs/kokoro-82m/bin/python scripts/bench/kokoro_timing_check.py \
  --beats docs/benchmarks/2026-10-24-kokoro-timing-beats.yaml \
  --out .superpowers/sdd/2026-10-24-narration/kokoro-check --analyze-only
```

Scratch directory: `/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check`. Each numbered take has `.wav`, raw `.alignment.json`, and `.take.json` with native tokens, actual chunk sample lengths/offsets, text, voice and runtime metadata. `results.json` contains every pair and unmatched word. These artifacts are intentionally uncommitted; audio links below are local to this machine.

## Matching and decision

Δ = Kokoro start − aligner start, in seconds. Each native token with non-None `start_ts` is offset by the exact number of samples already concatenated / 24,000. Normalization uses Unicode NFKD, case folding, and retains only letters/digits. Punctuation-only tokens are excluded. `SequenceMatcher(autojunk=False)` pairs equal normalized words monotonically; substitutions, insertions and deletions are reported, never paired by guessed time. Tokenization differences are not silently merged. Repeated words can remain ambiguous after an insertion/deletion; raw lists are saved for inspection.

Matched pairs: 6038; unmatched native words: 0; unmatched aligner words: 0. Primary denominator = matched pairs + all unmatched lexical tokens on both sides = 6038. This conservative rule gives no agreement credit to unmatched words and can count a substitution twice. Matched-only rates are shown separately; neither changes the decision. Statistics of Δ use matched pairs only. Inclusive 0.1 s comparisons allow 1e-9 s of floating-point roundoff tolerance; reported timestamps are not rounded before analysis. p95 uses linear interpolation at `(n−1) × 0.95`.

Correction subtracts the global mean signed Δ from native times, equivalently adds its negative. It is fitted and evaluated on this same sample; a passing fitted result would still require later regression validation. The brief’s decision uses the uncorrected rate first, then this one constant. Per-chunk fits below are diagnostic and do not override STOP.

| Treatment | Within 0.1 s (primary) | Matched only | Median Δ s | p95 Δ s | Mean Δ s | Median absolute s | p95 absolute s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 99.122% | 99.122% | -0.022500 | 0.032500 | -0.022999 | 0.030000 | 0.077500 |
| Subtract mean Δ | 99.785% | 99.785% | 0.000499 | 0.055499 | -0.000000 | 0.022999 | 0.065499 |

Uncorrected agreement is 5,985 / 6,038 words, leaving 53 disagreements. After subtracting the mean, agreement is 6,025 / 6,038, leaving 13 disagreements. The listening table includes the union of both sets (56 occurrences).

## Voice and chunk behaviour

Rates in this section are matched-only; sample sizes are shown. A chunk boundary is a new Kokoro result joined directly after the preceding result’s audio, with no added silence.

| Voice | Takes | Pairs | Raw within 0.1 s | Corrected within 0.1 s | Mean raw Δ s | p95 absolute raw s |
|---|---:|---:|---:|---:|---:|---:|
| af_bella | 5 | 670 | 98.806% | 99.851% | -0.026489 | 0.082500 |
| af_heart | 5 | 677 | 98.818% | 99.852% | -0.023600 | 0.078000 |
| af_kore | 5 | 589 | 99.491% | 100.000% | -0.021099 | 0.077500 |
| am_liam | 5 | 701 | 99.001% | 99.857% | -0.023206 | 0.072500 |
| am_michael | 10 | 1094 | 98.903% | 99.360% | -0.018003 | 0.077500 |
| am_onyx | 5 | 438 | 99.087% | 100.000% | -0.026016 | 0.077500 |
| am_puck | 5 | 677 | 99.705% | 100.000% | -0.024158 | 0.077500 |
| bf_emma | 10 | 1192 | 99.245% | 99.748% | -0.024333 | 0.075000 |

| Chunk group | Pairs | Raw within 0.1 s | Corrected within 0.1 s | Mean raw Δ s | p95 absolute raw s |
|---|---:|---:|---:|---:|---:|
| first_chunk | 4000 | 99.225% | 99.725% | -0.022674 | 0.077500 |
| later_chunks | 2038 | 98.921% | 99.902% | -0.023638 | 0.080000 |
| first_three_words_after_boundary | 120 | 100.000% | 100.000% | -0.021667 | 0.075000 |

Even fitting a separate mean to each chunk on this same sample gives 99.785% primary agreement (99.785% matched-only). This is an optimistic diagnostic, not a production calibration. Per-chunk correction cannot be justified merely by accumulated chunk indices; sample-based concatenation already removes that bookkeeping drift.

All multi-chunk takes, with exact boundary offsets and chunk-specific signed means:

| Beat | Voice | Chunk (zero-based) | Audio offset s | Samples | Matched words | Mean raw Δ s | First matched Δ s |
|---|---|---:|---:|---:|---:|---:|---:|
| measure/01-the-descent/0 | bf_emma | 0 | 0.000000 | 461400 | 72 | -0.025139 | -0.070000 |
| measure/01-the-descent/0 | bf_emma | 1 | 19.225000 | 266400 | 42 | -0.034881 | -0.045000 |
| measure/01-the-descent/1 | bf_emma | 0 | 0.000000 | 626400 | 88 | -0.022614 | -0.070000 |
| measure/01-the-descent/1 | bf_emma | 1 | 26.100000 | 260400 | 40 | -0.029687 | -0.025000 |
| measure/01-the-descent/2 | bf_emma | 0 | 0.000000 | 604800 | 101 | -0.027970 | -0.045000 |
| measure/01-the-descent/2 | bf_emma | 1 | 25.200000 | 183600 | 24 | -0.032813 | -0.070000 |
| measure/01-the-descent/3 | bf_emma | 0 | 0.000000 | 546000 | 89 | -0.020506 | 0.035000 |
| measure/01-the-descent/3 | bf_emma | 1 | 22.750000 | 364200 | 55 | -0.027773 | -0.015000 |
| measure/01-the-descent/5 | bf_emma | 0 | 0.000000 | 547800 | 87 | -0.021724 | 0.035000 |
| measure/01-the-descent/5 | bf_emma | 1 | 22.825000 | 505200 | 79 | -0.028861 | 0.005000 |
| measure/03-the-tells/2 | bf_emma | 0 | 0.000000 | 544200 | 82 | -0.019543 | -0.070000 |
| measure/03-the-tells/2 | bf_emma | 1 | 22.675000 | 414600 | 57 | -0.018114 | -0.035000 |
| measure/03-the-tells/5 | bf_emma | 0 | 0.000000 | 598800 | 87 | -0.017500 | -0.020000 |
| measure/03-the-tells/5 | bf_emma | 1 | 24.950000 | 378600 | 53 | -0.017830 | 0.050000 |
| the-unbegotten/01-the-silence-and-the-waking/2 | am_michael | 0 | 0.000000 | 446400 | 55 | -0.016500 | 0.050000 |
| the-unbegotten/01-the-silence-and-the-waking/2 | am_michael | 1 | 18.600000 | 364800 | 49 | -0.024745 | -0.065000 |
| the-unbegotten/01-the-silence-and-the-waking/3 | am_michael | 0 | 0.000000 | 453600 | 58 | -0.009138 | -0.050000 |
| the-unbegotten/01-the-silence-and-the-waking/3 | am_michael | 1 | 18.900000 | 420000 | 56 | -0.014866 | 0.045000 |
| the-unbegotten/02-of-the-waking-one/1 | am_michael | 0 | 0.000000 | 711600 | 101 | -0.016089 | -0.050000 |
| the-unbegotten/02-of-the-waking-one/1 | am_michael | 1 | 29.650000 | 210000 | 27 | -0.016019 | -0.005000 |
| the-unbegotten/02-of-the-waking-one/2 | am_michael | 0 | 0.000000 | 663600 | 102 | -0.013750 | -0.025000 |
| the-unbegotten/02-of-the-waking-one/2 | am_michael | 1 | 27.650000 | 187800 | 25 | -0.009700 | 0.000000 |
| the-unbegotten/02-of-the-waking-one/3 | am_michael | 0 | 0.000000 | 627600 | 81 | -0.023272 | 0.000000 |
| the-unbegotten/02-of-the-waking-one/3 | am_michael | 1 | 26.150000 | 417600 | 56 | -0.015402 | -0.010000 |
| the-unbegotten/02-of-the-waking-one/4 | am_michael | 0 | 0.000000 | 532200 | 70 | -0.017929 | 0.025000 |
| the-unbegotten/02-of-the-waking-one/4 | am_michael | 1 | 22.175000 | 524400 | 63 | -0.018135 | -0.010000 |
| _archive/the-remainder/16-hazard-pay/0 | af_bella | 0 | 0.000000 | 646800 | 100 | -0.029275 | -0.050000 |
| _archive/the-remainder/16-hazard-pay/0 | af_bella | 1 | 26.950000 | 57600 | 5 | -0.048500 | -0.035000 |
| _archive/the-remainder/16-hazard-pay/1 | af_bella | 0 | 0.000000 | 642000 | 93 | -0.027016 | -0.055000 |
| _archive/the-remainder/16-hazard-pay/1 | af_bella | 1 | 26.750000 | 156000 | 23 | -0.026739 | -0.020000 |
| _archive/the-remainder/16-hazard-pay/2 | af_bella | 0 | 0.000000 | 589200 | 88 | -0.018665 | -0.050000 |
| _archive/the-remainder/16-hazard-pay/2 | af_bella | 1 | 24.550000 | 587400 | 81 | -0.022747 | -0.035000 |
| _archive/the-remainder/16-hazard-pay/3 | af_bella | 0 | 0.000000 | 303000 | 40 | -0.024188 | -0.025000 |
| _archive/the-remainder/16-hazard-pay/3 | af_bella | 1 | 12.625000 | 620400 | 89 | -0.033539 | -0.040000 |
| _archive/the-remainder/16-hazard-pay/3 | af_bella | 2 | 38.475000 | 225000 | 30 | -0.031250 | -0.030000 |
| _archive/the-remainder/16-hazard-pay/4 | af_bella | 0 | 0.000000 | 603000 | 83 | -0.025301 | 0.000000 |
| _archive/the-remainder/16-hazard-pay/4 | af_bella | 1 | 25.125000 | 281400 | 38 | -0.025658 | -0.070000 |
| _archive/the-remainder/19-on-the-record/0 | am_liam | 0 | 0.000000 | 434400 | 71 | -0.016162 | -0.015000 |
| _archive/the-remainder/19-on-the-record/0 | am_liam | 1 | 18.100000 | 496200 | 97 | -0.028943 | -0.020000 |
| _archive/the-remainder/19-on-the-record/1 | am_liam | 0 | 0.000000 | 238200 | 40 | -0.016938 | -0.015000 |
| _archive/the-remainder/19-on-the-record/1 | am_liam | 1 | 9.925000 | 139800 | 21 | -0.022976 | -0.010000 |
| _archive/the-remainder/19-on-the-record/1 | am_liam | 2 | 15.750000 | 480000 | 90 | -0.020722 | 0.000000 |
| _archive/the-remainder/19-on-the-record/3 | am_liam | 0 | 0.000000 | 531000 | 81 | -0.023951 | -0.015000 |
| _archive/the-remainder/19-on-the-record/3 | am_liam | 1 | 22.125000 | 318600 | 42 | -0.020238 | -0.075000 |
| _archive/the-remainder/19-on-the-record/4 | am_liam | 0 | 0.000000 | 324000 | 47 | -0.016862 | -0.015000 |
| _archive/the-remainder/19-on-the-record/4 | am_liam | 1 | 13.500000 | 450000 | 72 | -0.028924 | -0.035000 |
| _archive/the-remainder/19-on-the-record/4 | am_liam | 2 | 32.250000 | 304200 | 47 | -0.022340 | -0.005000 |
| _archive/the-remainder/25-a-hands-breadth/0 | af_kore | 0 | 0.000000 | 419400 | 62 | -0.015081 | -0.015000 |
| _archive/the-remainder/25-a-hands-breadth/0 | af_kore | 1 | 17.475000 | 448200 | 79 | -0.021234 | 0.045000 |
| _archive/the-remainder/25-a-hands-breadth/1 | af_kore | 0 | 0.000000 | 427800 | 70 | -0.022071 | -0.040000 |
| _archive/the-remainder/25-a-hands-breadth/1 | af_kore | 1 | 17.825000 | 447600 | 65 | -0.022615 | -0.005000 |
| _archive/the-remainder/25-a-hands-breadth/2 | af_kore | 0 | 0.000000 | 483600 | 92 | -0.025734 | -0.065000 |
| _archive/the-remainder/25-a-hands-breadth/2 | af_kore | 1 | 20.150000 | 450000 | 69 | -0.020833 | 0.000000 |
| _archive/the-remainder/25-a-hands-breadth/4 | af_kore | 0 | 0.000000 | 400200 | 51 | -0.020294 | 0.010000 |
| _archive/the-remainder/25-a-hands-breadth/4 | af_kore | 1 | 16.675000 | 408600 | 64 | -0.020547 | 0.070000 |
| _archive/the-remainder/26-the-reversible-observer/2 | am_onyx | 0 | 0.000000 | 546000 | 100 | -0.027825 | -0.095000 |
| _archive/the-remainder/26-the-reversible-observer/2 | am_onyx | 1 | 22.750000 | 139200 | 20 | -0.024125 | -0.065000 |
| measure/01-the-descent/0 | af_heart | 0 | 0.000000 | 489000 | 72 | -0.018368 | -0.070000 |
| measure/01-the-descent/0 | af_heart | 1 | 20.375000 | 283800 | 42 | -0.025476 | -0.015000 |
| measure/01-the-descent/1 | af_heart | 0 | 0.000000 | 669600 | 88 | -0.024006 | -0.070000 |
| measure/01-the-descent/1 | af_heart | 1 | 27.900000 | 271800 | 40 | -0.016688 | 0.015000 |
| measure/01-the-descent/2 | af_heart | 0 | 0.000000 | 654000 | 101 | -0.026139 | -0.045000 |
| measure/01-the-descent/2 | af_heart | 1 | 27.250000 | 186600 | 24 | -0.030208 | -0.020000 |
| measure/01-the-descent/3 | af_heart | 0 | 0.000000 | 585600 | 89 | -0.020730 | -0.020000 |
| measure/01-the-descent/3 | af_heart | 1 | 24.400000 | 388200 | 55 | -0.027864 | 0.060000 |
| measure/01-the-descent/5 | af_heart | 0 | 0.000000 | 589200 | 87 | -0.022241 | -0.020000 |
| measure/01-the-descent/5 | af_heart | 1 | 24.550000 | 546000 | 79 | -0.026930 | 0.050000 |
| measure/01-the-descent/0 | am_puck | 0 | 0.000000 | 432000 | 72 | -0.021354 | -0.065000 |
| measure/01-the-descent/0 | am_puck | 1 | 18.000000 | 276600 | 42 | -0.025536 | -0.015000 |
| measure/01-the-descent/1 | am_puck | 0 | 0.000000 | 519600 | 88 | -0.025085 | -0.065000 |
| measure/01-the-descent/1 | am_puck | 1 | 21.650000 | 263400 | 40 | -0.014438 | 0.060000 |
| measure/01-the-descent/2 | am_puck | 0 | 0.000000 | 480000 | 101 | -0.029802 | -0.065000 |
| measure/01-the-descent/2 | am_puck | 1 | 20.000000 | 184200 | 24 | -0.018333 | -0.065000 |
| measure/01-the-descent/3 | am_puck | 0 | 0.000000 | 463800 | 89 | -0.025112 | -0.040000 |
| measure/01-the-descent/3 | am_puck | 1 | 19.325000 | 372000 | 55 | -0.015636 | 0.055000 |
| measure/01-the-descent/5 | am_puck | 0 | 0.000000 | 454800 | 87 | -0.027356 | 0.040000 |
| measure/01-the-descent/5 | am_puck | 1 | 18.950000 | 455400 | 79 | -0.025759 | 0.055000 |

## Disagreements for listening

Every matched word exceeding 0.1 s either before or after the global correction is listed below; the raw and corrected deltas distinguish the two sets. Occurrence is the zero-based matched-pair index within that take. No outliers were discarded, and the aligner may itself be wrong.

| Beat | Voice | Occurrence | Word | Kokoro start s | Aligner start s | Raw Δ s | Corrected Δ s | WAV |
|---|---|---:|---|---:|---:|---:|---:|---|
| measure/01-the-descent/0 | bf_emma | 49 | her | 12.537500 | 12.640000 | -0.102500 | -0.079501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/000-bf_emma.wav) |
| measure/01-the-descent/0 | bf_emma | 112 | to | 29.075000 | 29.200000 | -0.125000 | -0.102001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/000-bf_emma.wav) |
| measure/01-the-descent/1 | bf_emma | 29 | the | 8.487500 | 8.640000 | -0.152500 | -0.129501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/001-bf_emma.wav) |
| measure/01-the-descent/2 | bf_emma | 6 | To | 1.950000 | 1.600000 | 0.350000 | 0.372999 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/002-bf_emma.wav) |
| measure/01-the-descent/2 | bf_emma | 51 | to | 12.850000 | 12.960000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/002-bf_emma.wav) |
| measure/01-the-descent/2 | bf_emma | 63 | to | 15.400000 | 15.520000 | -0.120000 | -0.097001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/002-bf_emma.wav) |
| measure/01-the-descent/2 | bf_emma | 82 | to | 19.800000 | 19.920000 | -0.120000 | -0.097001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/002-bf_emma.wav) |
| measure/01-the-descent/5 | bf_emma | 58 | the | 14.450000 | 14.560000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/004-bf_emma.wav) |
| measure/03-the-tells/6 | bf_emma | 39 | to | 10.600000 | 10.720000 | -0.120000 | -0.097001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/009-bf_emma.wav) |
| the-unbegotten/01-the-silence-and-the-waking/0 | am_michael | 27 | world | 7.837500 | 7.760000 | 0.077500 | 0.100499 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/010-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/0 | am_michael | 46 | hush | 14.187500 | 14.320000 | -0.132500 | -0.109501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/010-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/2 | am_michael | 38 | holding | 13.087500 | 13.200000 | -0.112500 | -0.089501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/012-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/2 | am_michael | 72 | whole | 23.850000 | 24.000000 | -0.150000 | -0.127001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/012-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/4 | am_michael | 10 | It | 3.250000 | 3.360000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/014-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/4 | am_michael | 32 | smooth | 10.000000 | 9.920000 | 0.080000 | 0.102999 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/014-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/4 | am_michael | 51 | It | 15.725000 | 15.840000 | -0.115000 | -0.092001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/014-am_michael.wav) |
| the-unbegotten/01-the-silence-and-the-waking/4 | am_michael | 55 | to | 16.775000 | 16.880000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/014-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/0 | am_michael | 53 | that | 13.775000 | 13.920000 | -0.145000 | -0.122001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/015-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/2 | am_michael | 64 | word | 16.887500 | 16.800000 | 0.087500 | 0.110499 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/017-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/3 | am_michael | 8 | earnest | 2.537500 | 2.640000 | -0.102500 | -0.079501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/018-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/3 | am_michael | 22 | how | 7.575000 | 7.680000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/018-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/3 | am_michael | 59 | whole | 18.275000 | 18.400000 | -0.125000 | -0.102001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/018-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/3 | am_michael | 88 | whole | 28.762500 | 28.880000 | -0.117500 | -0.094501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/018-am_michael.wav) |
| the-unbegotten/02-of-the-waking-one/4 | am_michael | 94 | to | 29.975000 | 30.080000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/019-am_michael.wav) |
| _archive/the-remainder/16-hazard-pay/0 | af_bella | 65 | and | 16.762500 | 16.880000 | -0.117500 | -0.094501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/020-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/0 | af_bella | 91 | height | 23.550000 | 23.680000 | -0.130000 | -0.107001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/020-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/1 | af_bella | 109 | have | 31.250000 | 31.360000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/021-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/2 | af_bella | 141 | but | 39.237500 | 39.360000 | -0.122500 | -0.099501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/022-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/3 | af_bella | 78 | only | 23.412500 | 23.520000 | -0.107500 | -0.084501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/023-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/3 | af_bella | 103 | high | 30.362500 | 30.480000 | -0.117500 | -0.094501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/023-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/3 | af_bella | 139 | three | 41.575000 | 41.680000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/023-af_bella.wav) |
| _archive/the-remainder/16-hazard-pay/4 | af_bella | 113 | that | 33.637500 | 33.760000 | -0.122500 | -0.099501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/024-af_bella.wav) |
| _archive/the-remainder/19-on-the-record/1 | am_liam | 28 | do | 6.875000 | 6.720000 | 0.155000 | 0.177999 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/026-am_liam.wav) |
| _archive/the-remainder/19-on-the-record/1 | am_liam | 59 | to | 14.600000 | 14.720000 | -0.120000 | -0.097001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/026-am_liam.wav) |
| _archive/the-remainder/19-on-the-record/3 | am_liam | 31 | hold | 7.175000 | 7.280000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/028-am_liam.wav) |
| _archive/the-remainder/19-on-the-record/3 | am_liam | 47 | the | 12.375000 | 12.480000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/028-am_liam.wav) |
| _archive/the-remainder/19-on-the-record/4 | am_liam | 28 | who | 7.400000 | 7.520000 | -0.120000 | -0.097001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/029-am_liam.wav) |
| _archive/the-remainder/19-on-the-record/4 | am_liam | 87 | who | 23.575000 | 23.680000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/029-am_liam.wav) |
| _archive/the-remainder/19-on-the-record/4 | am_liam | 125 | to | 33.575000 | 33.680000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/029-am_liam.wav) |
| _archive/the-remainder/25-a-hands-breadth/0 | af_kore | 67 | hundred | 18.600000 | 18.720000 | -0.120000 | -0.097001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/030-af_kore.wav) |
| _archive/the-remainder/25-a-hands-breadth/1 | af_kore | 127 | who | 33.650000 | 33.760000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/031-af_kore.wav) |
| _archive/the-remainder/25-a-hands-breadth/2 | af_kore | 110 | to | 25.250000 | 25.360000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/032-af_kore.wav) |
| _archive/the-remainder/26-the-reversible-observer/0 | am_onyx | 14 | holds | 3.575000 | 3.680000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/035-am_onyx.wav) |
| _archive/the-remainder/26-the-reversible-observer/0 | am_onyx | 20 | holds | 4.525000 | 4.640000 | -0.115000 | -0.092001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/035-am_onyx.wav) |
| _archive/the-remainder/26-the-reversible-observer/2 | am_onyx | 51 | It | 11.337500 | 11.440000 | -0.102500 | -0.079501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/037-am_onyx.wav) |
| _archive/the-remainder/26-the-reversible-observer/3 | am_onyx | 31 | they | 7.575000 | 7.680000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/038-am_onyx.wav) |
| measure/01-the-descent/0 | af_heart | 111 | hired | 30.762500 | 30.880000 | -0.117500 | -0.094501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/040-af_heart.wav) |
| measure/01-the-descent/0 | af_heart | 112 | to | 31.175000 | 31.280000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/040-af_heart.wav) |
| measure/01-the-descent/1 | af_heart | 78 | the | 24.375000 | 24.480000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/041-af_heart.wav) |
| measure/01-the-descent/2 | af_heart | 107 | have | 28.925000 | 29.040000 | -0.115000 | -0.092001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/042-af_heart.wav) |
| measure/01-the-descent/3 | af_heart | 14 | her | 3.975000 | 4.080000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/043-af_heart.wav) |
| measure/01-the-descent/3 | af_heart | 127 | had | 35.737500 | 35.840000 | -0.102500 | -0.079501 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/043-af_heart.wav) |
| measure/01-the-descent/5 | af_heart | 68 | to | 18.275000 | 18.400000 | -0.125000 | -0.102001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/044-af_heart.wav) |
| measure/01-the-descent/5 | af_heart | 106 | to | 29.650000 | 29.760000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/044-af_heart.wav) |
| measure/01-the-descent/0 | am_puck | 112 | to | 28.050000 | 28.160000 | -0.110000 | -0.087001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/045-am_puck.wav) |
| measure/01-the-descent/2 | am_puck | 23 | to | 5.175000 | 5.280000 | -0.105000 | -0.082001 | [WAV](/home/spinoza/github/muse-narration/.superpowers/sdd/2026-10-24-narration/kokoro-check/047-am_puck.wav) |

## Unmatched lexical words

These words are not counted as agreement. Starts are in seconds from the beginning of the WAV; no guessed counterpart is assigned.

None: zero unmatched native words and zero unmatched aligner words.
