# LLM head-to-head: muse vs ollama

## short

| server | median elapsed s | tok/s (wall) | tok/s (gen-only, self-reported) |
|---|---|---|---|
| muse | 3.04 | 10.86 | - |
| ollama | 4.88 | 7.79 | 8.6 |

## ttft

| server | median TTFT s | median inter-token ms |
|---|---|---|
| muse | 0.476 | 202.1 |
| ollama | 0.315 | 82.6 |

## split

| server | median elapsed s | tok/s (wall) | tok/s (gen-only, self-reported) |
|---|---|---|---|
| muse | 2.21 | 15.42 | - |
| muse | 2.1 | 15.67 | - |
