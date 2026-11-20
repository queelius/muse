"""PCM16 / WAV / Opus encoding shared by audio/speech and audio/generation.

Audio is float32 in [-1, 1] at the protocol boundary: `(samples,)` mono or
`(samples, channels)` interleaved frames.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from muse.core.venv import run_owned_command


class AudioFormatError(ValueError):
    """Raised when audio data cannot be encoded to the requested format."""


class FfmpegUnavailableError(AudioFormatError):
    """ffmpeg is not on PATH, so the format cannot be produced here."""


_FFMPEG_TIMEOUT_SECONDS = 60.0
_OPUS_OUTPUT_SLACK_BYTES = 1024 * 1024


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to an int16 PCM array.

    The one float -> int16 conversion for WAV encoding and the speech SSE
    stream. Scale by 32768 so -1.0 -> -32768 and +1.0 -> +32768, then clip
    to [-32768, 32767] before casting; a 32767 multiplier would leave
    -32768 unreachable.
    """
    scaled = np.clip(audio, -1.0, 1.0) * 32768.0
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def pcm16_wav_bytes(audio: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """A 16-bit PCM WAV of `audio` (C-contiguous frames, row-major)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(float_to_pcm16(audio).tobytes())
    return buf.getvalue()


def wav_bytes_to_opus(wav_data: bytes) -> bytes:
    """Transcode WAV -> Opus (OGG) with an owned, time-bounded ffmpeg.

    Raises FfmpegUnavailableError when ffmpeg is missing and
    AudioFormatError when the conversion fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegUnavailableError("ffmpeg not found; cannot encode opus")
    with tempfile.TemporaryDirectory(prefix="muse-ffmpeg-") as raw_dir:
        work_dir = Path(raw_dir)
        input_path = work_dir / "input.wav"
        output_path = work_dir / "output.ogg"
        input_path.write_bytes(wav_data)
        try:
            proc = run_owned_command(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "wav", "-i", str(input_path),
                    "-c:a", "libopus", "-b:a", "64k",
                    "-f", "ogg", str(output_path),
                ],
                capture_output=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioFormatError(
                f"ffmpeg timed out after {_FFMPEG_TIMEOUT_SECONDS:g}s"
            ) from exc
        if proc.returncode != 0:
            detail = (proc.stderr or "")[:200]
            raise AudioFormatError(f"ffmpeg failed: {detail}")
        max_output = len(wav_data) + _OPUS_OUTPUT_SLACK_BYTES
        try:
            with output_path.open("rb") as handle:
                output = handle.read(max_output + 1)
        except OSError as exc:
            raise AudioFormatError("ffmpeg produced no readable Opus output") from exc
        if len(output) > max_output:
            raise AudioFormatError("ffmpeg Opus output exceeded its safety bound")
        if not output:
            raise AudioFormatError("ffmpeg produced empty Opus output")
        return output
