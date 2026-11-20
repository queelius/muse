"""Image ingress shared by every image-input route (and chat's VLM path).

Three input shapes:
  - data:image/{png,jpeg,webp};base64,XYZ (``decode_image_input``)
  - http(s)://... (``decode_image_input``), fetched through
    ``muse.core.net_fetch`` with its SSRF guard and byte cap: hostnames
    that resolve to private/loopback/link-local IPs are refused unless
    MUSE_ALLOW_PRIVATE_FETCH=1. The response content type is not checked;
    the bytes must decode as an image.
  - multipart uploads (``read_and_decode_upload`` / ``read_image_file``)

The bounded read (or fetch) stays on the event loop as async I/O; the
pixel decode always runs in a worker thread through ``run_native_offload``,
and an image decoded after its request was cancelled is closed there.

Failures raise typed ``ImageInputError`` subclasses (all ValueError), and
``image_error_response`` maps them onto one wire table:

  ImageTooLargeError    byte cap            413 payload_too_large
  ImageDimensionsError  side / pixel caps   413 input_too_large
  ImageDecodeError      empty, malformed,   400 invalid_image
                        undecodable
  ImageFetchError       URL refused/failed  400 invalid_parameter

PIL is imported lazily (it is a pip_extra of the image runtimes).
"""
from __future__ import annotations

import base64
import inspect
import io
import logging
import re
from typing import Any

from muse.core import config
from muse.core.errors import error_response
from muse.core.net_fetch import ResponseTooLargeError, afetch_url_bytes
from muse.modalities._native_offload import run_native_offload


logger = logging.getLogger(__name__)


_DATA_URL_RE = re.compile(
    r"^data:([a-zA-Z0-9.+/-]+);base64,(.*)$",
    re.DOTALL,
)
_ALLOWED_IMAGE_MIME = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/webp",
})
_HTTP_TIMEOUT = 30.0

_BYTES_SETTING = "MUSE_IMAGE_INPUT_MAX_BYTES"
_PIXELS_SETTING = "MUSE_IMAGE_INPUT_MAX_PIXELS"
_TOTAL_PIXELS_SETTING = "MUSE_IMAGE_INPUT_MAX_TOTAL_PIXELS"


class ImageInputError(ValueError):
    """An image input was rejected; ``status`` / ``code`` are its wire form."""

    status = 400
    code = "invalid_parameter"


class ImageDecodeError(ImageInputError):
    """Empty, malformed or undecodable image input."""

    code = "invalid_image"


class ImageFetchError(ImageInputError):
    """An http(s) image URL was refused (SSRF guard) or could not be fetched."""


class ImageTooLargeError(ImageInputError):
    """Image input (raw or base64-encoded) exceeds the byte cap."""

    status = 413
    code = "payload_too_large"


class ImageDimensionsError(ImageInputError):
    """Decoded dimensions exceed a cap.

    ``limit`` is ``"side"`` (a per-route max side), ``"pixels"`` (one
    image) or ``"total_pixels"`` (a batch).
    """

    status = 413
    code = "input_too_large"

    def __init__(self, message: str, *, limit: str):
        super().__init__(message)
        self.limit = limit


def image_error_response(
    exc: ImageInputError,
    *,
    field: str | None = None,
    side_setting: str | None = None,
    bytes_setting: str = _BYTES_SETTING,
):
    """The OpenAI-shape error response for a rejected image input.

    Size errors name the setting that raises the cap: ``bytes_setting``
    for the byte cap, ``side_setting`` for a route's side cap.
    """
    message = f"{field}: {exc}" if field else str(exc)
    setting = None
    if isinstance(exc, ImageTooLargeError):
        setting = bytes_setting
    elif isinstance(exc, ImageDimensionsError):
        setting = {
            "side": side_setting,
            "pixels": _PIXELS_SETTING,
            "total_pixels": _TOTAL_PIXELS_SETTING,
        }.get(exc.limit)
    if setting:
        message += f" (set {setting} to raise)"
    return error_response(exc.status, exc.code, message)


def _default_max_bytes() -> int:
    """Image-input byte cap (env MUSE_IMAGE_INPUT_MAX_BYTES), read live so
    operators can raise it without a restart; <= 0 means the default."""
    return config.get_positive("limits.image_input_max_bytes")


def _default_max_pixels() -> int:
    """Decoded-raster cap shared by every image ingress shape."""
    return int(config.get_positive("limits.image_input_max_pixels"))


def _default_max_total_pixels() -> int:
    """Aggregate decoded-raster cap shared by batched image routes."""
    return int(config.get_positive("limits.image_input_max_total_pixels"))


def validate_total_image_pixels(
    images: list[Any], *, max_total_pixels: int | None = None,
) -> int:
    """Return aggregate pixels, rejecting batches above the configured cap."""
    cap = (
        max_total_pixels
        if max_total_pixels is not None
        else _default_max_total_pixels()
    )
    if cap <= 0:
        raise ValueError("max_total_pixels must be positive")
    total = sum(int(image.size[0]) * int(image.size[1]) for image in images)
    if total > cap:
        raise ImageDimensionsError(
            f"image batch too large: {total} decoded pixels exceeds "
            f"aggregate limit {cap}",
            limit="total_pixels",
        )
    return total


def close_decoded_images(images: list[Any]) -> None:
    """Best-effort deterministic cleanup for request-owned PIL images."""
    seen: set[int] = set()
    for image in images:
        if image is None:
            continue
        identity = id(image)
        if identity in seen:
            continue
        seen.add(identity)
        close = getattr(image, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.debug("failed to close decoded request image", exc_info=True)


async def _decode_off_loop(raw: bytes, **caps) -> Any:
    return await run_native_offload(
        lambda: _bytes_to_pil(raw, **caps),
        cleanup_abandoned=lambda img: close_decoded_images([img]),
    )


async def decode_image_input(
    value: str,
    *,
    max_bytes: int | None = None,
    max_pixels: int | None = None,
) -> Any:
    """Parse a data URL or HTTP(S) URL into a PIL.Image.

    The fetch is async I/O; base64 and pixel decoding run in a worker
    thread. ``max_bytes`` / ``max_pixels`` default to the live
    MUSE_IMAGE_INPUT_MAX_BYTES / MUSE_IMAGE_INPUT_MAX_PIXELS settings.
    """
    cap = max_bytes if max_bytes is not None else _default_max_bytes()
    pixel_cap = max_pixels if max_pixels is not None else _default_max_pixels()
    if value.startswith("data:"):
        return await run_native_offload(
            lambda: _decode_data_url(value, max_bytes=cap, max_pixels=pixel_cap),
            cleanup_abandoned=lambda img: close_decoded_images([img]),
        )
    if value.startswith(("http://", "https://")):
        raw = await _fetch_http_url(value, max_bytes=cap)
        return await _decode_off_loop(raw, max_pixels=pixel_cap)
    raise ImageDecodeError(
        f"image must be a data: URL or http(s):// URL; got {value[:30]!r}..."
    )


async def decode_image_file(
    file: Any,
    *,
    max_bytes: int | None = None,
    max_side: int | None = None,
    max_pixels: int | None = None,
) -> Any:
    """Read a multipart upload (bounded, on the loop) and decode it in a
    worker thread. Raises ``ImageInputError`` subclasses.

    The argument is typed Any so this helper doesn't bind us to FastAPI
    or Starlette at import time: anything with an async ``.read()`` works.
    """
    raw = await read_image_file(file, max_bytes=max_bytes)
    return await _decode_off_loop(raw, max_side=max_side, max_pixels=max_pixels)


async def read_and_decode_upload(
    upload: Any,
    field: str = "image",
    *,
    max_bytes: int | None = None,
    max_side: int | None = None,
    side_setting: str | None = None,
    bytes_setting: str = _BYTES_SETTING,
):
    """Read and decode one upload; returns ``(image, None)`` or
    ``(None, error_response)`` per the module's error table."""
    try:
        image = await decode_image_file(
            upload, max_bytes=max_bytes, max_side=max_side,
        )
    except ImageInputError as e:
        return None, image_error_response(
            e, field=field, side_setting=side_setting,
            bytes_setting=bytes_setting,
        )
    return image, None


async def read_and_decode_uploads(
    uploads: list[tuple[str, Any]],
    *,
    max_side: int | None = None,
    side_setting: str | None = None,
    check_total_pixels: bool = False,
):
    """Decode ``(field, upload)`` pairs in order; returns
    ``(images, None)`` or ``(None, error_response)``.

    On any failure or cancellation the images decoded so far are closed.
    With ``check_total_pixels`` the batch must fit
    MUSE_IMAGE_INPUT_MAX_TOTAL_PIXELS.
    """
    images: list[Any] = []
    try:
        for field, upload in uploads:
            image, error = await read_and_decode_upload(
                upload, field, max_side=max_side, side_setting=side_setting,
            )
            if error is not None:
                close_decoded_images(images)
                return None, error
            images.append(image)
        if check_total_pixels:
            validate_total_image_pixels(images)
    except ImageInputError as e:
        close_decoded_images(images)
        return None, image_error_response(e)
    except BaseException:
        close_decoded_images(images)
        raise
    return images, None


async def read_image_file(file: Any, *, max_bytes: int | None = None) -> bytes:
    """The bounded read half of ``decode_image_file``: read at most
    ``max_bytes + 1`` bytes, close the upload, and reject empty
    (ImageDecodeError) or oversized (ImageTooLargeError) input. No pixel
    decoding."""
    cap = max_bytes if max_bytes is not None else _default_max_bytes()
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("max_bytes must be a positive integer")
    try:
        raw = await file.read(cap + 1)
    finally:
        # Multipart UploadFile objects may be backed by a spooled temporary
        # file.  The decoder owns the upload after its single bounded read, so
        # close it on success, rejection, and cancellation.  Test doubles with
        # no close method remain supported.
        close = getattr(file, "close", None)
        if callable(close):
            try:
                outcome = close()
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception:  # noqa: BLE001
                logger.debug("failed to close image upload", exc_info=True)
    if not raw:
        raise ImageDecodeError("empty image file")
    if len(raw) > cap:
        raise ImageTooLargeError(
            f"image bytes exceeds max ({len(raw)} > {cap})"
        )
    return raw


def decode_image_bytes(
    raw: bytes,
    *,
    max_side: int | None = None,
    max_pixels: int | None = None,
) -> Any:
    """The decode half of ``decode_image_file`` (synchronous, so it can run
    in a worker thread): dimension caps first, then a full decode."""
    return _bytes_to_pil(raw, max_side=max_side, max_pixels=max_pixels)


def _decode_data_url(value: str, *, max_bytes: int, max_pixels: int):
    m = _DATA_URL_RE.match(value)
    if not m:
        raise ImageDecodeError("malformed data URL")
    mime = m.group(1).lower()
    if mime not in _ALLOWED_IMAGE_MIME:
        raise ImageDecodeError(
            f"unsupported MIME {mime!r}; allowed: {sorted(_ALLOWED_IMAGE_MIME)}"
        )
    b64 = m.group(2)
    # Reject before decoding: base64 inflates ~4/3, so any b64 string
    # larger than (max_bytes * 4 / 3) cannot fit under the cap. Adding 4
    # for padding slack avoids false-positives on exact-fit boundaries.
    max_b64_len = (max_bytes * 4) // 3 + 4
    if len(b64) > max_b64_len:
        raise ImageTooLargeError(
            f"image bytes exceeds max ({len(b64)} b64 chars > {max_b64_len} limit)"
        )
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ImageDecodeError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ImageTooLargeError(
            f"image bytes exceeds max ({len(raw)} > {max_bytes})"
        )
    return _bytes_to_pil(raw, max_pixels=max_pixels)


async def _fetch_http_url(value: str, *, max_bytes: int) -> bytes:
    # An over-cap body is net_fetch's typed ResponseTooLargeError (413, like
    # an oversize upload); SSRF refusals, bad redirects and the deadline are
    # plain ValueErrors and map to ImageFetchError (400).
    try:
        return await afetch_url_bytes(
            value, max_bytes=max_bytes, timeout=_HTTP_TIMEOUT,
        )
    except ResponseTooLargeError as e:
        raise ImageTooLargeError(str(e)) from e
    except ValueError as e:
        raise ImageFetchError(str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise ImageFetchError(f"fetch failed: {e}") from e


def _bytes_to_pil(
    raw: bytes,
    *,
    max_side: int | None = None,
    max_pixels: int | None = None,
):
    from PIL import Image, UnidentifiedImageError
    img = None
    try:
        img = Image.open(io.BytesIO(raw))
        width, height = img.size
        if max_side is not None and (
            width > max_side or height > max_side
        ):
            raise ImageDimensionsError(
                f"image too large: {width}x{height} exceeds max input side "
                f"{max_side}",
                limit="side",
            )
        pixel_cap = max_pixels if max_pixels is not None else _default_max_pixels()
        if pixel_cap <= 0:
            raise ValueError("max_pixels must be positive")
        pixels = width * height
        if pixels > pixel_cap:
            raise ImageDimensionsError(
                f"image too large: {width}x{height} ({pixels} pixels) "
                f"exceeds max input pixels {pixel_cap}",
                limit="pixels",
            )
        img.load()  # force decode now so errors surface here, not later
    except UnidentifiedImageError as e:
        if img is not None:
            img.close()
        # PIL's message embeds the BytesIO repr; it says nothing useful.
        raise ImageDecodeError(
            "image decode failed: unrecognised image format"
        ) from e
    except Image.DecompressionBombError as e:
        # A plain Exception (not OSError): a tiny PNG declaring huge
        # dimensions would otherwise escape as a 500. PIL still blocks the
        # memory DoS; it is a pixel-cap failure, not an undecodable image.
        if img is not None:
            img.close()
        raise ImageDimensionsError(
            f"image too large: {e}", limit="pixels",
        ) from e
    except OSError as e:
        if img is not None:
            img.close()
        raise ImageDecodeError(f"image decode failed: {e}") from e
    except BaseException:
        if img is not None:
            img.close()
        raise
    return img
