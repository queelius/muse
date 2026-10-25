# Narration acceptance, 2026-10-24

Isolated server on :8010 (catalog outside ~/.muse), kokoro-82m, qwen3-asr-1.7b (listener, no prompt), qwen3-forced-aligner-0.6b and utmos on an RTX 3060, driven by `scripts/bench/speech_compare.py`. Scratch audio and results.json are not committed.

## Word-timing agreement (Kokoro native vs forced aligner)

- Takes: 100 (the 50 frozen beats in docs/benchmarks/2026-10-24-kokoro-timing-beats.yaml, each under both targets), 0 errors.
- Words compared: 12092; within 0.1 s: 11952; **agreement 98.84%** (bar 95%: met). Worst single take: 95.45%.
- Each beat's own voice overrides the `af_heart` / `am_puck` target, so the two targets repeat the same voices; the figure is over every take.

### Every disagreement

Timing rows give Kokoro seconds, aligner seconds and delta. `unmatched` rows are words one side lacks after ordered pairing (the listed start/end are that side's).

| beat | target | word | kind | kokoro / side | aligner | delta |
|---|---|---|---|---|---|---|
| measure/01-the-descent/0 | kokoro-82m:af_heart | her | timing | 12.537 | 12.64 | -0.103 |
| measure/01-the-descent/0 | kokoro-82m:af_heart | to | timing | 29.075 | 29.2 | -0.125 |
| measure/01-the-descent/1 | kokoro-82m:af_heart | the | timing | 8.488 | 8.64 | -0.152 |
| measure/01-the-descent/2 | kokoro-82m:af_heart | To | timing | 1.95 | 1.6 | +0.350 |
| measure/01-the-descent/2 | kokoro-82m:af_heart | to | timing | 12.85 | 12.96 | -0.110 |
| measure/01-the-descent/2 | kokoro-82m:af_heart | to | timing | 15.4 | 15.52 | -0.120 |
| measure/01-the-descent/5 | kokoro-82m:af_heart | the | timing | 14.45 | 14.56 | -0.110 |
| measure/03-the-tells/3 | kokoro-82m:af_heart | half-seen | unmatched (kokoro) | 5.825-6.275 | - | - |
| measure/03-the-tells/3 | kokoro-82m:af_heart | halfseen | unmatched (aligner) | 5.92-6.24 | - | - |
| measure/03-the-tells/3 | kokoro-82m:af_heart | time-stamped, | unmatched (kokoro) | 20.5-21.225 | - | - |
| measure/03-the-tells/3 | kokoro-82m:af_heart | timestamped | unmatched (aligner) | 20.48-21.28 | - | - |
| measure/03-the-tells/6 | kokoro-82m:af_heart | to | timing | 10.6 | 10.72 | -0.120 |
| measure/03-the-tells/6 | kokoro-82m:af_heart | boil-off, | unmatched (kokoro) | 16.075-16.55 | - | - |
| measure/03-the-tells/6 | kokoro-82m:af_heart | boiloff | unmatched (aligner) | 16.08-16.64 | - | - |
| the-unbegotten/01-the-silence-and-the-waking/0 | kokoro-82m:af_heart | hush | timing | 14.188 | 14.32 | -0.132 |
| the-unbegotten/01-the-silence-and-the-waking/2 | kokoro-82m:af_heart | holding | timing | 13.088 | 13.2 | -0.112 |
| the-unbegotten/01-the-silence-and-the-waking/2 | kokoro-82m:af_heart | whole | timing | 23.85 | 24.0 | -0.150 |
| the-unbegotten/01-the-silence-and-the-waking/4 | kokoro-82m:af_heart | It | timing | 3.25 | 3.36 | -0.110 |
| the-unbegotten/01-the-silence-and-the-waking/4 | kokoro-82m:af_heart | to | timing | 16.775 | 16.88 | -0.105 |
| the-unbegotten/02-of-the-waking-one/0 | kokoro-82m:af_heart | that | timing | 13.775 | 13.92 | -0.145 |
| the-unbegotten/02-of-the-waking-one/0 | kokoro-82m:af_heart | long-held | unmatched (kokoro) | 27.637-28.262 | - | - |
| the-unbegotten/02-of-the-waking-one/0 | kokoro-82m:af_heart | longheld | unmatched (aligner) | 27.6-28.32 | - | - |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:af_heart | earnest, | timing | 2.538 | 2.64 | -0.102 |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:af_heart | how | timing | 7.575 | 7.68 | -0.105 |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:af_heart | whole | timing | 18.275 | 18.4 | -0.125 |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:af_heart | whole | timing | 28.762 | 28.88 | -0.118 |
| the-unbegotten/02-of-the-waking-one/4 | kokoro-82m:af_heart | to | timing | 29.975 | 30.08 | -0.105 |
| _archive/the-remainder/16-hazard-pay/0 | kokoro-82m:af_heart | and | timing | 16.762 | 16.88 | -0.118 |
| _archive/the-remainder/16-hazard-pay/2 | kokoro-82m:af_heart | day | timing | 0.425 | 0.64 | -0.215 |
| _archive/the-remainder/16-hazard-pay/2 | kokoro-82m:af_heart | but | timing | 39.237 | 39.36 | -0.123 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:af_heart | only | timing | 23.413 | 23.52 | -0.107 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:af_heart | high | timing | 30.363 | 30.48 | -0.117 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:af_heart | three | timing | 41.575 | 41.68 | -0.105 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:af_heart | not-funny | unmatched (kokoro) | 44.825-45.5 | - | - |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:af_heart | notfunny | unmatched (aligner) | 44.8-45.52 | - | - |
| _archive/the-remainder/16-hazard-pay/4 | kokoro-82m:af_heart | that | timing | 33.638 | 33.76 | -0.122 |
| _archive/the-remainder/19-on-the-record/0 | kokoro-82m:af_heart | seventy-two | unmatched (kokoro) | 32.875-33.4 | - | - |
| _archive/the-remainder/19-on-the-record/0 | kokoro-82m:af_heart | seventytwo | unmatched (aligner) | 32.96-33.44 | - | - |
| _archive/the-remainder/19-on-the-record/1 | kokoro-82m:af_heart | do | timing | 6.875 | 6.72 | +0.155 |
| _archive/the-remainder/19-on-the-record/1 | kokoro-82m:af_heart | to | timing | 14.6 | 14.72 | -0.120 |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:af_heart | load-bearing | unmatched (kokoro) | 4.325-4.825 | - | - |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:af_heart | loadbearing | unmatched (aligner) | 4.32-4.88 | - | - |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:af_heart | hold | timing | 7.175 | 7.28 | -0.105 |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:af_heart | the | timing | 12.375 | 12.48 | -0.105 |
| _archive/the-remainder/19-on-the-record/4 | kokoro-82m:af_heart | who | timing | 7.4 | 7.52 | -0.120 |
| _archive/the-remainder/19-on-the-record/4 | kokoro-82m:af_heart | who | timing | 23.575 | 23.68 | -0.105 |
| _archive/the-remainder/19-on-the-record/4 | kokoro-82m:af_heart | to | timing | 33.575 | 33.68 | -0.105 |
| _archive/the-remainder/25-a-hands-breadth/0 | kokoro-82m:af_heart | hundred | timing | 18.6 | 18.72 | -0.120 |
| _archive/the-remainder/25-a-hands-breadth/1 | kokoro-82m:af_heart | who | timing | 33.65 | 33.76 | -0.110 |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:af_heart | she | timing | 9.412 | 9.52 | -0.108 |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:af_heart | ninety-nine | unmatched (kokoro) | 11.963-12.425 | - | - |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:af_heart | ninetynine | unmatched (aligner) | 12.0-12.48 | - | - |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:af_heart | to | timing | 25.25 | 25.36 | -0.110 |
| _archive/the-remainder/26-the-reversible-observer/0 | kokoro-82m:af_heart | that | timing | 1.012 | 1.12 | -0.108 |
| _archive/the-remainder/26-the-reversible-observer/0 | kokoro-82m:af_heart | holds, | timing | 3.575 | 3.68 | -0.105 |
| _archive/the-remainder/26-the-reversible-observer/2 | kokoro-82m:af_heart | It | timing | 11.338 | 11.44 | -0.102 |
| _archive/the-remainder/26-the-reversible-observer/3 | kokoro-82m:af_heart | they | timing | 7.575 | 7.68 | -0.105 |
| measure/01-the-descent/0 | kokoro-82m:af_heart | hired | timing | 30.762 | 30.88 | -0.118 |
| measure/01-the-descent/0 | kokoro-82m:af_heart | to | timing | 31.175 | 31.28 | -0.105 |
| measure/01-the-descent/1 | kokoro-82m:af_heart | the | timing | 24.375 | 24.48 | -0.105 |
| measure/01-the-descent/2 | kokoro-82m:af_heart | have, | timing | 28.925 | 29.04 | -0.115 |
| measure/01-the-descent/3 | kokoro-82m:af_heart | her | timing | 3.975 | 4.08 | -0.105 |
| measure/01-the-descent/3 | kokoro-82m:af_heart | had | timing | 35.737 | 35.84 | -0.103 |
| measure/01-the-descent/5 | kokoro-82m:af_heart | to | timing | 18.275 | 18.4 | -0.125 |
| measure/01-the-descent/5 | kokoro-82m:af_heart | to | timing | 29.65 | 29.76 | -0.110 |
| measure/01-the-descent/0 | kokoro-82m:af_heart | to | timing | 28.05 | 28.16 | -0.110 |
| measure/01-the-descent/2 | kokoro-82m:af_heart | to | timing | 5.175 | 5.28 | -0.105 |
| measure/01-the-descent/5 | kokoro-82m:af_heart | to | timing | 23.875 | 24.0 | -0.125 |
| measure/01-the-descent/0 | kokoro-82m:am_puck | her | timing | 12.537 | 12.64 | -0.103 |
| measure/01-the-descent/0 | kokoro-82m:am_puck | to | timing | 29.075 | 29.2 | -0.125 |
| measure/01-the-descent/1 | kokoro-82m:am_puck | the | timing | 8.488 | 8.64 | -0.152 |
| measure/01-the-descent/2 | kokoro-82m:am_puck | To | timing | 1.95 | 1.6 | +0.350 |
| measure/01-the-descent/2 | kokoro-82m:am_puck | to | timing | 12.85 | 12.96 | -0.110 |
| measure/01-the-descent/2 | kokoro-82m:am_puck | to | timing | 15.4 | 15.52 | -0.120 |
| measure/01-the-descent/5 | kokoro-82m:am_puck | the | timing | 14.45 | 14.56 | -0.110 |
| measure/03-the-tells/3 | kokoro-82m:am_puck | half-seen | unmatched (kokoro) | 5.825-6.275 | - | - |
| measure/03-the-tells/3 | kokoro-82m:am_puck | halfseen | unmatched (aligner) | 5.92-6.24 | - | - |
| measure/03-the-tells/3 | kokoro-82m:am_puck | time-stamped, | unmatched (kokoro) | 20.5-21.225 | - | - |
| measure/03-the-tells/3 | kokoro-82m:am_puck | timestamped | unmatched (aligner) | 20.48-21.28 | - | - |
| measure/03-the-tells/6 | kokoro-82m:am_puck | to | timing | 10.6 | 10.72 | -0.120 |
| measure/03-the-tells/6 | kokoro-82m:am_puck | boil-off, | unmatched (kokoro) | 16.075-16.55 | - | - |
| measure/03-the-tells/6 | kokoro-82m:am_puck | boiloff | unmatched (aligner) | 16.08-16.64 | - | - |
| the-unbegotten/01-the-silence-and-the-waking/0 | kokoro-82m:am_puck | hush | timing | 14.188 | 14.32 | -0.132 |
| the-unbegotten/01-the-silence-and-the-waking/2 | kokoro-82m:am_puck | holding | timing | 13.088 | 13.2 | -0.112 |
| the-unbegotten/01-the-silence-and-the-waking/2 | kokoro-82m:am_puck | whole | timing | 23.85 | 24.0 | -0.150 |
| the-unbegotten/01-the-silence-and-the-waking/4 | kokoro-82m:am_puck | It | timing | 3.25 | 3.36 | -0.110 |
| the-unbegotten/01-the-silence-and-the-waking/4 | kokoro-82m:am_puck | It | timing | 15.725 | 15.84 | -0.115 |
| the-unbegotten/01-the-silence-and-the-waking/4 | kokoro-82m:am_puck | to | timing | 16.775 | 16.88 | -0.105 |
| the-unbegotten/02-of-the-waking-one/0 | kokoro-82m:am_puck | that | timing | 13.775 | 13.92 | -0.145 |
| the-unbegotten/02-of-the-waking-one/0 | kokoro-82m:am_puck | long-held | unmatched (kokoro) | 27.637-28.262 | - | - |
| the-unbegotten/02-of-the-waking-one/0 | kokoro-82m:am_puck | longheld | unmatched (aligner) | 27.6-28.32 | - | - |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:am_puck | earnest, | timing | 2.538 | 2.64 | -0.102 |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:am_puck | how | timing | 7.575 | 7.68 | -0.105 |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:am_puck | whole | timing | 18.275 | 18.4 | -0.125 |
| the-unbegotten/02-of-the-waking-one/3 | kokoro-82m:am_puck | whole | timing | 28.762 | 28.88 | -0.118 |
| the-unbegotten/02-of-the-waking-one/4 | kokoro-82m:am_puck | to | timing | 29.975 | 30.08 | -0.105 |
| _archive/the-remainder/16-hazard-pay/0 | kokoro-82m:am_puck | hazard | timing | 2.75 | 2.88 | -0.130 |
| _archive/the-remainder/16-hazard-pay/0 | kokoro-82m:am_puck | and | timing | 16.762 | 16.88 | -0.118 |
| _archive/the-remainder/16-hazard-pay/0 | kokoro-82m:am_puck | height. | timing | 23.55 | 23.68 | -0.130 |
| _archive/the-remainder/16-hazard-pay/1 | kokoro-82m:am_puck | have | timing | 31.25 | 31.36 | -0.110 |
| _archive/the-remainder/16-hazard-pay/2 | kokoro-82m:am_puck | day | timing | 0.425 | 0.64 | -0.215 |
| _archive/the-remainder/16-hazard-pay/2 | kokoro-82m:am_puck | but | timing | 39.237 | 39.36 | -0.123 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:am_puck | high | timing | 30.363 | 30.48 | -0.117 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:am_puck | three | timing | 41.575 | 41.68 | -0.105 |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:am_puck | not-funny | unmatched (kokoro) | 44.825-45.5 | - | - |
| _archive/the-remainder/16-hazard-pay/3 | kokoro-82m:am_puck | notfunny | unmatched (aligner) | 44.8-45.52 | - | - |
| _archive/the-remainder/16-hazard-pay/4 | kokoro-82m:am_puck | that | timing | 33.638 | 33.76 | -0.122 |
| _archive/the-remainder/19-on-the-record/0 | kokoro-82m:am_puck | seventy-two | unmatched (kokoro) | 32.875-33.4 | - | - |
| _archive/the-remainder/19-on-the-record/0 | kokoro-82m:am_puck | seventytwo | unmatched (aligner) | 32.96-33.44 | - | - |
| _archive/the-remainder/19-on-the-record/1 | kokoro-82m:am_puck | do | timing | 6.875 | 6.72 | +0.155 |
| _archive/the-remainder/19-on-the-record/1 | kokoro-82m:am_puck | to | timing | 14.6 | 14.72 | -0.120 |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:am_puck | load-bearing | unmatched (kokoro) | 4.325-4.825 | - | - |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:am_puck | loadbearing | unmatched (aligner) | 4.32-4.88 | - | - |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:am_puck | hold | timing | 7.175 | 7.28 | -0.105 |
| _archive/the-remainder/19-on-the-record/3 | kokoro-82m:am_puck | the | timing | 12.375 | 12.48 | -0.105 |
| _archive/the-remainder/19-on-the-record/4 | kokoro-82m:am_puck | who | timing | 7.4 | 7.52 | -0.120 |
| _archive/the-remainder/19-on-the-record/4 | kokoro-82m:am_puck | who | timing | 23.575 | 23.68 | -0.105 |
| _archive/the-remainder/19-on-the-record/4 | kokoro-82m:am_puck | to | timing | 33.575 | 33.68 | -0.105 |
| _archive/the-remainder/25-a-hands-breadth/0 | kokoro-82m:am_puck | hundred | timing | 18.6 | 18.72 | -0.120 |
| _archive/the-remainder/25-a-hands-breadth/1 | kokoro-82m:am_puck | who | timing | 33.65 | 33.76 | -0.110 |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:am_puck | ninety-nine | unmatched (kokoro) | 11.963-12.425 | - | - |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:am_puck | ninetynine | unmatched (aligner) | 12.0-12.48 | - | - |
| _archive/the-remainder/25-a-hands-breadth/2 | kokoro-82m:am_puck | to | timing | 25.25 | 25.36 | -0.110 |
| _archive/the-remainder/26-the-reversible-observer/0 | kokoro-82m:am_puck | that | timing | 1.012 | 1.12 | -0.108 |
| _archive/the-remainder/26-the-reversible-observer/0 | kokoro-82m:am_puck | holds, | timing | 3.575 | 3.68 | -0.105 |
| _archive/the-remainder/26-the-reversible-observer/0 | kokoro-82m:am_puck | holds, | timing | 4.525 | 4.64 | -0.115 |
| _archive/the-remainder/26-the-reversible-observer/2 | kokoro-82m:am_puck | It | timing | 11.338 | 11.44 | -0.102 |
| _archive/the-remainder/26-the-reversible-observer/3 | kokoro-82m:am_puck | to | timing | 7.075 | 7.2 | -0.125 |
| _archive/the-remainder/26-the-reversible-observer/3 | kokoro-82m:am_puck | they | timing | 7.575 | 7.68 | -0.105 |
| measure/01-the-descent/0 | kokoro-82m:am_puck | hired | timing | 30.762 | 30.88 | -0.118 |
| measure/01-the-descent/0 | kokoro-82m:am_puck | to | timing | 31.175 | 31.28 | -0.105 |
| measure/01-the-descent/1 | kokoro-82m:am_puck | the | timing | 24.375 | 24.48 | -0.105 |
| measure/01-the-descent/2 | kokoro-82m:am_puck | have, | timing | 28.925 | 29.04 | -0.115 |
| measure/01-the-descent/3 | kokoro-82m:am_puck | her | timing | 3.975 | 4.08 | -0.105 |
| measure/01-the-descent/3 | kokoro-82m:am_puck | had | timing | 35.737 | 35.84 | -0.103 |
| measure/01-the-descent/5 | kokoro-82m:am_puck | to | timing | 18.275 | 18.4 | -0.125 |
| measure/01-the-descent/5 | kokoro-82m:am_puck | to | timing | 29.65 | 29.76 | -0.110 |
| measure/01-the-descent/0 | kokoro-82m:am_puck | to | timing | 28.05 | 28.16 | -0.110 |
| measure/01-the-descent/2 | kokoro-82m:am_puck | to | timing | 5.175 | 5.28 | -0.105 |
| measure/01-the-descent/5 | kokoro-82m:am_puck | to | timing | 23.875 | 24.0 | -0.125 |

140 disagreements in total.

## Pronunciation lexicon (heard by Qwen3-ASR)

| entry | af_heart | am_puck |
|---|---|---|
| monoid | heard (monoid: y; "Here is the word "monoid" in a sentence.") | heard (monoid: y; "Here is the word monoid in a sentence.") |
| monoids | heard (monoids: y; "Here is the word monoids in a sentence.") | heard (monoids: y; "Here is the word monoids in a sentence.") |
| superposition | heard (superposition: y; "Here is the word superposition in a sentence.") | heard (superposition: y; "Here is the word superposition in a sentence.") |
| superpositions | heard (superpositions: y; "Here is the word superpositions in a sentence.") | heard (superpositions: y; "Here is the word of superpositions in a sentence.") |
| poincare | heard (poincare: y; "Here is the word "Poincaré" in a sentence.") | heard (poincare: y; "Here is the word "Poincaré" in a sentence.") |
| poincaré | heard (poincaré: y; "Here is the word "Poincaré" in a sentence.") | heard (poincaré: y; "Here is the word "Poincaré" in a sentence.") |
| tanh | heard (tanh: y; "Here is the word hyperbolic tangent in a sentence.") | heard (tanh: y; "Here is the word hyperbolic tangent in a sentence.") |
| json | heard (json: y; "Here is the word JSON in a sentence.") | heard (json: y; "Here is the word JSON in a sentence.") |
| jq | heard (jq: y; "Here is the word JQ in a sentence.") | heard (jq: y; "Here is the word JQ in a sentence.") |
| exclusive or | heard (exclusive or: y; "Here is the word "exclusive," or in a sentence.") | heard (exclusive or: y; "Here is the word "exclusive" or in a sentence.") |
| unbegotten | heard (unbegotten: y; "Here is the word "unbegotten" in a sentence.") | heard (unbegotten: y; "Here is the word "unbegotten" in a sentence.") |
| runge kutta | NOT heard (runge kutta: n; "Here is the word "rungakuta" in a sentence.") | NOT heard (runge kutta: n; "Here is the word "rungakuta" in a sentence.") |
| lipschitz | NOT heard (lipschitz: n; "Here is the word "elipsits" in a sentence.") | NOT heard (lipschitz: n; "Here is the word "alipshits" in a sentence.") |
| coulomb | NOT heard (coulomb: n; "Here is the word kulam in a sentence.") | NOT heard (coulomb: n; "Here is the word "koulom" in a sentence.") |
| blaise pascal | NOT heard (blaise pascal: n; "Here is the word "blaze Pascal" in a sentence.") | NOT heard (blaise pascal: n; "Here is the word "blaze" Pascal in a sentence.") |
| albert camus | heard (albert camus: y; "Here is the word Albert Camus in a sentence.") | heard (albert camus: y; "Here is the word Albert Camus in a sentence.") |

**af_heart: 12/16 heard**; **am_puck: 12/16 heard**

## Request example

```bash
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' -d '{"model":"kokoro-82m","voice":"af_heart","input":"Coulomb measured charge.","timestamps":true,"pronunciations":[{"term":"Coulomb","ipa":"kuːˈlɒm"}]}'
```

Live result: Coulomb 0.275-0.600, measured 0.600-1.012, charge. 1.012-1.650; `timing.source: native`, 3 measured, 0 estimated; `pronunciations.applied[0].kind: ipa`.

## Live checks

- Through the gateway, X-Muse-Voice, X-Muse-Speed, X-Muse-Audio-SHA256, X-Muse-Pronunciations-Digest, X-Muse-Model and X-Muse-Model-Revision all arrive. X-Muse-Audio-SHA256 equals sha256 of the WAV body, and for `timestamps: true` of the decoded audio.
- Qwen3-ASR on real weights: transcribes without `language`, with `language=English`, and with a one-off `prompt`; `language=Klingon` gives 400 `invalid_language` (fixed in this task); `decode(return_format='parsed')` exists at transformers 5.20.2.
- Probed peak for qwen3-asr-1.7b: 3.84 GB on cuda (weights 3.80 GB); curated `memory_gb` set to 4.0.

## Misses (reported, bar not loosened)

- Lexicon: runge kutta, lipschitz, coulomb and blaise pascal were not heard for either voice (12 of 16 heard per voice). The first three are proper names for which the lexicon asks for a hyphenated respelling or an unadorned name; the listener's guess ("rungakuta", "elipsits", "kulam") differs from the written term, and "blaze Pascal" is a plausible hearing of a Blaise respelling. Whether these are speech or listening errors needs a human ear.
- Timing agreement met the bar; 32 of the 140 disagreements are unmatched words (aligner spells numbers as one word, for example "ninetynine").
