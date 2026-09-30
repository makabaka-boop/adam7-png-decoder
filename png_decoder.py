"""A small, strict PNG decoder for the subset used by this project.

Only truecolor RGB and RGBA images with 8 bits per channel are accepted.
Both non-interlaced and Adam7 interlaced images are supported.  The public
:func:`decode_png` function returns RGBA bytes plus metadata, including the
number of scanlines delivered in each Adam7 pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import zlib
from typing import BinaryIO

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IHDR = b"IHDR"
PLTE = b"PLTE"
TRNS = b"tRNS"
IDAT = b"IDAT"
IEND = b"IEND"

# The specification allows encoders to split the single zlib stream over many
# consecutive IDAT chunks.  This independent cap protects callers from feeding
# an unbounded amount of compressed bytes before the dimension-relative stream
# length can be proven wrong.
MAX_COMPRESSED_BYTES = 24 * 1024 * 1024


class PNGDecodeError(ValueError):
    """Raised when a PNG is malformed or outside the supported subset."""


@dataclass(frozen=True)
class DecodedPNG:
    width: int
    height: int
    pixels: bytes
    """RGBA pixel bytes, row-major, four bytes per pixel."""

    interlace_method: int
    pass_lines: tuple[int, ...]
    """Number of encoded scanlines consumed for each pass.

    A non-interlaced image has one pass.  An Adam7 image always has seven
    entries; several can legitimately be zero.
    """

    def to_rows(self) -> list[bytes]:
        return [
            self.pixels[y * self.width * 4 : (y + 1) * self.width * 4]
            for y in range(self.height)
        ]


# Adam7 pass geometry.  Each tuple is (x_start, y_start, x_step, y_step).
# Starts are zero based.
ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


def _pass_dimensions(width: int, height: int, pass_index: int) -> tuple[int, int]:
    x0, y0, dx, dy = ADAM7_PASSES[pass_index]
    pass_width = 0 if width <= x0 else (width - x0 + dx - 1) // dx
    pass_height = 0 if height <= y0 else (height - y0 + dy - 1) // dy
    return pass_width, pass_height


def expected_data_size(width: int, height: int, channels: int, interlace: int) -> int:
    """Return the exact number of pre-filter bytes required by the image."""
    if interlace == 0:
        return height * (1 + width * channels)

    total = 0
    for pass_index in range(7):
        pw, ph = _pass_dimensions(width, height, pass_index)
        if pw and ph:
            total += ph * (1 + pw * channels)
    return total


@dataclass
class _IHDR:
    width: int
    height: int
    bit_depth: int
    color_type: int
    compression_method: int
    filter_method: int
    interlace_method: int


class _ChunkReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.pos = len(PNG_SIGNATURE)

    def read_exact(self, count: int, what: str) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise PNGDecodeError(f"truncated PNG while reading {what}")
        self.pos += len(data)
        return data

    def read_chunk_header(self) -> tuple[bytes, int]:
        length_raw = self.stream.read(4)
        if not length_raw:
            raise PNGDecodeError("truncated PNG: missing IEND")
        if len(length_raw) != 4:
            raise PNGDecodeError("truncated PNG while reading chunk length")
        length = int.from_bytes(length_raw, "big")
        self.pos += 4
        chunk_type = self.read_exact(4, "chunk type")
        return chunk_type, length

    def _finish_payload(
        self,
        chunk_type: bytes,
        length: int,
        actual_crc: int,
        collect: bool,
        chunks: list[bytes],
    ) -> bytes:
        expected_crc = int.from_bytes(self.read_exact(4, "CRC"), "big")
        if actual_crc != expected_crc:
            raise PNGDecodeError(
                f"CRC mismatch in {chunk_type.decode('latin1', 'replace')} chunk"
            )
        self.pos += length
        return b"".join(chunks) if collect else b""

    def read_payload(
        self,
        chunk_type: bytes,
        length: int,
        *,
        collect: bool = True,
        max_bytes: int | None = None,
    ) -> bytes:
        if max_bytes is not None and length > max_bytes:
            raise PNGDecodeError("chunk data exceeds safety limit")

        crc32 = zlib.crc32
        actual_crc = crc32(chunk_type)
        remaining = length
        chunks: list[bytes] = []
        while remaining:
            block = self.stream.read(min(65536, remaining))
            if not block:
                raise PNGDecodeError(
                    f"truncated PNG while reading {chunk_type!r} data"
                )
            actual_crc = crc32(block, actual_crc)
            if collect:
                chunks.append(block)
            remaining -= len(block)
        return self._finish_payload(
            chunk_type, length, actual_crc & 0xFFFFFFFF, collect, chunks
        )


def paeth_predictor(left: int, above: int, upper_left: int) -> int:
    p = left + above - upper_left
    pa = abs(p - left)
    pb = abs(p - above)
    pc = abs(p - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return above
    return upper_left


def _unfilter_scanline(
    raw_line: bytes,
    previous: bytes | None,
    bpp: int,
    width: int,
    output: bytearray,
    out_offset: int = 0,
) -> None:
    if len(raw_line) != 1 + width * bpp:
        raise PNGDecodeError("invalid compressed scanline length")
    filter_type = raw_line[0]
    line = raw_line[1:]

    if filter_type == 0:
        output[out_offset : out_offset + len(line)] = line
        return

    if previous is None:
        previous = bytes(len(line))

    if filter_type == 1:  # Sub
        for x in range(bpp):
            output[out_offset + x] = line[x]
        for x in range(bpp, len(line)):
            output[out_offset + x] = (line[x] + output[out_offset + x - bpp]) & 0xFF
    elif filter_type == 2:  # Up
        for x, value in enumerate(line):
            output[out_offset + x] = (value + previous[x]) & 0xFF
    elif filter_type == 3:  # Average
        for x, value in enumerate(line):
            left = output[out_offset + x - bpp] if x >= bpp else 0
            output[out_offset + x] = (value + ((left + previous[x]) >> 1)) & 0xFF
    elif filter_type == 4:  # Paeth
        for x, value in enumerate(line):
            if x >= bpp:
                left = output[out_offset + x - bpp]
                upper_left = previous[x - bpp]
            else:
                left = 0
                upper_left = 0
            pred = paeth_predictor(left, previous[x], upper_left)
            output[out_offset + x] = (value + pred) & 0xFF
    else:
        raise PNGDecodeError(f"unsupported PNG filter type {filter_type}")


def _parse_ihdr(data: bytes) -> _IHDR:
    if len(data) != 13:
        raise PNGDecodeError("IHDR chunk must contain exactly 13 data bytes")
    values = (
        int.from_bytes(data[0:4], "big"),
        int.from_bytes(data[4:8], "big"),
        data[8],
        data[9],
        data[10],
        data[11],
        data[12],
    )
    header = _IHDR(*values)

    # Dimensions are checked before any IDAT data is collected or inflated.
    if header.width == 0 or header.height == 0:
        raise PNGDecodeError("PNG width and height must be non-zero")
    if header.width > 0x7FFFFFFF or header.height > 0x7FFFFFFF:
        raise PNGDecodeError("PNG dimensions exceed 2^31-1")
    if header.width * header.height > 4_000_000:
        raise PNGDecodeError("PNG exceeds limit of 4,000,000 pixels")
    if header.bit_depth != 8:
        raise PNGDecodeError("only 8-bit images are supported")
    if header.color_type not in (2, 6):
        if header.color_type == 3:
            raise PNGDecodeError("palette PNG images are not supported")
        raise PNGDecodeError(f"unsupported PNG color type {header.color_type}")
    if header.compression_method != 0:
        raise PNGDecodeError(f"unsupported compression method {header.compression_method}")
    if header.filter_method != 0:
        raise PNGDecodeError(f"unsupported filter method {header.filter_method}")
    if header.interlace_method not in (0, 1):
        raise PNGDecodeError(f"unsupported interlace method {header.interlace_method}")
    return header


def _read_chunks(stream: BinaryIO) -> tuple[_IHDR, bytes]:
    signature = stream.read(len(PNG_SIGNATURE))
    if signature != PNG_SIGNATURE:
        raise PNGDecodeError("invalid PNG signature")

    reader = _ChunkReader(stream)
    first_type, first_length = reader.read_chunk_header()
    if first_type != IHDR:
        raise PNGDecodeError("IHDR must be the first PNG chunk")
    header = _parse_ihdr(reader.read_payload(first_type, first_length, max_bytes=13))

    idat_parts: list[bytes] = []
    compressed_length = 0
    seen_idat = False
    idat_finished = False

    while True:
        chunk_type, length = reader.read_chunk_header()

        if chunk_type == IHDR:
            raise PNGDecodeError("IHDR must appear exactly once")
        if chunk_type == PLTE:
            # Color type 3 was rejected in IHDR; this makes the unsupported
            # case explicit for palette payloads as well.
            raise PNGDecodeError("palette PNG images are not supported")
        if chunk_type == TRNS:
            # The decoder has no representation for RGB-with-transparency
            # input; silently producing opaque output would be incorrect.
            raise PNGDecodeError("tRNS transparency chunks are not supported")

        if chunk_type == IEND:
            if seen_idat and not idat_finished:
                data = reader.read_payload(chunk_type, length, max_bytes=0)
                if data:
                    raise PNGDecodeError("IEND data must be empty")
                break
            if not seen_idat:
                raise PNGDecodeError("IEND must appear after IDAT")
            raise PNGDecodeError("IEND must appear exactly once")

        if chunk_type == IDAT:
            if idat_finished:
                raise PNGDecodeError("IDAT chunks must be consecutive")
            if compressed_length + length > MAX_COMPRESSED_BYTES:
                raise PNGDecodeError("compressed IDAT data exceeds safety limit")
            data = reader.read_payload(
                chunk_type,
                length,
                max_bytes=MAX_COMPRESSED_BYTES - compressed_length,
            )
            idat_parts.append(data)
            compressed_length += len(data)
            seen_idat = True
            continue

        if seen_idat:
            # Ancillary chunks between IDAT and IEND split the zlib stream;
            # a second later IDAT is then non-consecutive as required.
            idat_finished = True
        if chunk_type[0] & 0x20 == 0:
            name = chunk_type.decode("latin1", "replace")
            raise PNGDecodeError(f"unknown critical chunk {name!r}")
        reader.read_payload(
            chunk_type, length, collect=False, max_bytes=MAX_COMPRESSED_BYTES
        )

    trailing = stream.read(1)
    if trailing:
        raise PNGDecodeError("trailing data after IEND")

    return header, b"".join(idat_parts)


def _inflate_exact(compressed: bytes, exact_length: int) -> bytes:
    # wbits=15 requests a zlib wrapper (CMF/FLG and Adler-32).  A max output
    # length is supplied before inflation, so malformed highly-compressible
    # streams cannot expand without bound.
    decompressor = zlib.decompressobj(15)
    try:
        data = decompressor.decompress(compressed, max_length=exact_length + 1)
    except zlib.error as exc:
        raise PNGDecodeError(f"invalid zlib data: {exc}") from exc

    if len(data) > exact_length:
        raise PNGDecodeError("trailing image data in zlib stream")

    # The stream may have been deliberately truncated but still produced the
    # expected prefix.  Requesting one more output byte forces zlib to visit
    # the end marker and checksum.
    try:
        extra = decompressor.decompress(b"", max_length=1)
        if extra:
            raise PNGDecodeError("trailing image data in zlib stream")
    except zlib.error as exc:
        raise PNGDecodeError(f"invalid zlib data: {exc}") from exc

    if not decompressor.eof:
        raise PNGDecodeError("truncated zlib image stream")
    if len(data) != exact_length:
        raise PNGDecodeError(
            f"zlib data length mismatch: expected {exact_length} bytes, got {len(data)}"
        )
    if decompressor.unused_data:
        raise PNGDecodeError("trailing compressed image data after zlib stream")
    return data


def _decode_non_interlaced(
    filtered: bytes, width: int, height: int, channels: int
) -> tuple[bytes, tuple[int, ...]]:
    row_bytes = width * channels
    stride = row_bytes + 1
    decoded = bytearray(height * row_bytes)
    previous: bytes | None = None

    for y in range(height):
        start = y * stride
        raw = filtered[start : start + stride]
        _unfilter_scanline(raw, previous, channels, width, decoded, y * row_bytes)
        previous = bytes(decoded[y * row_bytes : (y + 1) * row_bytes])

    return _expand_rgba(decoded, channels, width * height), (height,)


def _decode_interlaced(
    filtered: bytes, width: int, height: int, channels: int
) -> tuple[bytes, tuple[int, ...]]:
    rgba = bytearray(width * height * 4)
    pass_lines: list[int] = []
    offset = 0

    for pass_index in range(7):
        pw, ph = _pass_dimensions(width, height, pass_index)
        # A reduced pass with zero width contains no scanlines, even when its
        # nominal height is positive.
        pass_lines.append(ph if pw else 0)
        if pw == 0 or ph == 0:
            continue

        pass_row_bytes = pw * channels
        stride = pass_row_bytes + 1
        pass_bytes = bytearray(ph * pass_row_bytes)
        previous: bytes | None = None

        for y in range(ph):
            start = offset + y * stride
            raw = filtered[start : start + stride]
            _unfilter_scanline(raw, previous, channels, pw, pass_bytes, y * pass_row_bytes)
            previous = bytes(
                pass_bytes[y * pass_row_bytes : (y + 1) * pass_row_bytes]
            )
        offset += ph * stride

        x0, y0, dx, dy = ADAM7_PASSES[pass_index]
        for py in range(ph):
            image_y = y0 + py * dy
            for px in range(pw):
                image_x = x0 + px * dx
                src = (py * pw + px) * channels
                dst = (image_y * width + image_x) * 4
                rgba[dst] = pass_bytes[src]
                rgba[dst + 1] = pass_bytes[src + 1]
                rgba[dst + 2] = pass_bytes[src + 2]
                rgba[dst + 3] = pass_bytes[src + 3] if channels == 4 else 255

    if offset != len(filtered):
        raise PNGDecodeError("trailing interlaced image data")
    return bytes(rgba), tuple(pass_lines)


def _expand_rgba(samples: bytes, channels: int, pixel_count: int) -> bytes:
    if channels == 4:
        return bytes(samples)
    rgba = bytearray(pixel_count * 4)
    source = 0
    destination = 0
    while source < len(samples):
        rgba[destination : destination + 3] = samples[source : source + 3]
        rgba[destination + 3] = 255
        source += 3
        destination += 4
    return bytes(rgba)


def decode_png(source: bytes | BinaryIO) -> DecodedPNG:
    """Decode a supported PNG and return RGBA pixels and pass-line evidence."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        import io

        stream: BinaryIO = io.BytesIO(bytes(source))
    else:
        stream = source

    header, compressed = _read_chunks(stream)
    channels = 4 if header.color_type == 6 else 3
    exact_length = expected_data_size(
        header.width, header.height, channels, header.interlace_method
    )
    filtered = _inflate_exact(compressed, exact_length)

    if header.interlace_method == 0:
        pixels, pass_lines = _decode_non_interlaced(
            filtered, header.width, header.height, channels
        )
    else:
        pixels, pass_lines = _decode_interlaced(
            filtered, header.width, header.height, channels
        )

    return DecodedPNG(
        width=header.width,
        height=header.height,
        pixels=pixels,
        interlace_method=header.interlace_method,
        pass_lines=pass_lines,
    )
