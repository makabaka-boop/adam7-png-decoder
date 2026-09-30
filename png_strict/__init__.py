"""A small, strict, pure-Python 8-bit RGB/RGBA PNG decoder.

The public entry point is :func:`decode_png`.
"""

from .decoder import (
    PNGDecodeError,
    PNGImage,
    PassEvidence,
    decode_png,
)

__all__ = ["PNGDecodeError", "PNGImage", "PassEvidence", "decode_png"]
