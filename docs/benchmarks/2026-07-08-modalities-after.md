# Modality hot latencies: http://192.168.0.204:8000

## modalities

| case | model | median s |
|---|---|---|
| image_gen_sd_turbo | sd-turbo | 0.903 |
| image_gen_pixel_lora | pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5 | 4.714 |
| tts_kokoro | kokoro-82m | 0.402 |
| tts_supertonic | supertonic-3 | SKIPPED (not enabled) |
| sfx_stable_audio | stable-audio-open-1.0 | 17.684 |
| transcribe_whisper-base | whisper-base | ERROR HTTPStatusError |
| embed_qwen3-embedding-0.6b | qwen3-embedding-0.6b | 0.634 |
| chat_llama-3.2-3b-q4 | llama-3.2-3b-q4 | 2.816 |

## whisper-base -> cuda: resolved after CT2 CUDA libs

The first after-run's transcribe row ERRORed: CTranslate2's GPU path
needs libcublas.so.12 (+cuDNN) and frodo has no system CUDA toolkit.
Fixed with the same recipe as the llama.cpp cu-wheel (nvidia-cublas-cu12
+ nvidia-cudnn-cu12 pip wheels + venv sitecustomize RTLD_GLOBAL preload),
worker bounced, re-measured:

| case | model | before (cpu) | after (cuda) | speedup |
|---|---|---|---|---|
| transcribe_whisper-base | whisper-base | 8.27s | 0.72s | 11.5x |

## Before/after summary (frodo)

| case | baseline | after | note |
|---|---|---|---|
| image_gen_sd_turbo | 0.97s | 0.90s | noise |
| image_gen_pixel_lora | 7.75s | 4.71s | window variance; steps knob documented |
| tts_kokoro | 0.75s | 0.40s | noise/warm |
| sfx_stable_audio | 19.25s | 17.68s | 50-step default; steps knob documented |
| transcribe_whisper-base | 8.27s | 0.72s | set-device cuda + CT2 CUDA libs: 11.5x |
| embed_qwen3-embedding-0.6b | 1.12s | 0.63s | noise/warm |
| chat_llama-3.2-3b-q4 | 2.31s | 2.82s | noise |
