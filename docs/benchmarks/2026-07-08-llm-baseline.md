# LLM head-to-head: muse vs ollama

## short

| server | median elapsed s | tok/s (wall) | tok/s (gen-only, self-reported) |
|---|---|---|---|
| muse | 7.26 | 4.51 | - |
| ollama | 3.92 | 7.94 | 9.0 |

## ttft

| server | median TTFT s | median inter-token ms |
|---|---|---|
| muse | 0.336 | 103.8 |
| ollama | 0.294 | 79.0 |

## long

| server | prompt tokens | median elapsed s | prompt-eval s (self-reported) |
|---|---|---|---|
| muse | 1418 | 2.0 | - |
| ollama | 1439 | 4.28 | 0.06 |

## turns

| server | turn 1 s | turn 2 s | turn 3 s | turn 4 s |
|---|---|---|---|---|
| muse | 0.58 | 2.98 | 1.44 | 3.13 |
| ollama | 1.08 | 3.1 | 1.75 | 3.26 |

## split

| server | median elapsed s | tok/s (wall) | tok/s (gen-only, self-reported) |
|---|---|---|---|
| muse | 1.89 | 16.99 | - |
| muse | 1.97 | 17.11 | - |
