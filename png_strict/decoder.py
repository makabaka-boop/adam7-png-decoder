"""Strict decoder for the small subset of PNG required by this project."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# PNG's four-byte unsigned length field allows at most 2**31 - 1 bytes.
MAX_PNG_DIMENSION = (1 << 31) - 1
# Product limit applied before any compressed data is touched.
MAX_PIXELS = 4_000_000

# Filter types 0-4 are the only filters defined for PNG's interlace methods.
_SUPPORTED_FILTERS = frozenset(range(5))

# Adam7 starts (x, y; zero based), x increments and y increments.
_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


class PNGDecodeError(ValueError):
    """Raised when input is not a supported and valid PNG file."""


@dataclass(frozen=True)
class PassEvidence:
    """Decompressed and unfiltered image rows for one interlace pass.

    For a non-interlaced image there is exactly one pass.  An Adam7 pass whose
    grid does not intersect the image has ``row_count == 0`` and ``rows == ()``.
    """

    index: int
    width: int
    height: int
    rows: tuple[bytes, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class PNGImage:
    width: int
    height: int
    interlace_method: int
    pixels: bytes
    passes: tuple[PassEvidence, ...]


def decode_png(source: bytes | bytearray | "memoryview") -> PNGImage:
    """Decode an 8-bit RGB or RGBA PNG and return RGBA pixels.

    Only compression method 0, filter method 0 and interlace methods 0 (none)
    and 1 (Adam7) are accepted.
    """

    data = _as_input_bytes(source)
    if not data.startswith(PNG_SIGNATURE):
        raise PNGDecodeError("invalid PNG signature")

    pos = len(PNG_SIGNATURE)
    ihdr = None
    saw_plte = False
    compressed_parts: list[bytes] = []
    saw_iend = False
    state = "before_idat"

    while pos < len(data):
        if saw_iend:
            raise PNGDecodeError("data is present after IEND")
        if pos + 8 > len(data):
            raise PNGDecodeError("truncated chunk header")

        length = struct.unpack_from(">I", data, pos)[0]
        chunk_type_pos = pos + 4
        data_pos = chunk_type_pos + 4
        data_end = data_pos + length
        crc_end = data_end + 4
        if crc_end > len(data):
            raise PNGDecodeError(f"truncated {_chunk_name(data, chunk_type_pos)} chunk")

        chunk_type = bytes(data[chunk_type_pos:data_pos])
        payload = bytes(data[data_pos:data_end])
        actual_crc = struct.unpack_from(">I", data, data_end)[0]
        expected_crc = zlib.crc32(chunk_type)
        expected_crc = zlib.crc32(payload, expected_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PNGDecodeError(
                f"CRC mismatch in {_safe_chunk_name(chunk_type)} chunk"
            )

        pos = crc_end

        if chunk_type == b"IHDR":
            if ihdr is not None:
                raise PNGDecodeError("duplicate IHDR chunk")
            ihdr = _parse_ihdr(payload)
        elif chunk_type == b"PLTE":
            if ihdr is None:
                raise PNGDecodeError("PLTE appears before IHDR")
            if state != "before_idat":
                raise PNGDecodeError("PLTE appears after IDAT")
            if saw_plte:
                raise PNGDecodeError("duplicate PLTE chunk")
            if ihdr[2] == 4:
                raise PNGDecodeError("PLTE is forbidden in RGBA PNGs")
            _validate_plte(payload)
            saw_plte = True
        elif chunk_type == b"IDAT":
            if ihdr is None:
                raise PNGDecodeError("IDAT appears before IHDR")
            if state == "after_idat":
                raise PNGDecodeError("IDAT chunks are not contiguous")
            state = "in_idat"
            compressed_parts.append(payload)
        elif chunk_type == b"IEND":
            if ihdr is None:
                raise PNGDecodeError("IEND appears before IHDR")
            if state == "before_idat":
                raise PNGDecodeError("IEND appears before IDAT")
            if payload:
                raise PNGDecodeError("non-empty IEND chunk")
            state = "after_idat"
            saw_iend = True
            if pos != len(data):
                raise PNGDecodeError("data is present after IEND")
        else:
            _validate_unknown_or_ancillary(chunk_type, state, ihdr)
            if state == "in_idat":
                state = "after_idat"

    if not saw_iend:
        raise PNGDecodeError("PNG stream ends without IEND")

    assert ihdr is not None
    width, height, channels, interlace = ihdr
    pass_specs = _pass_specifications(width, height, interlace)
    expected_bytes = sum(
        pass_height * (1 + pass_width * channels)
        for pass_width, pass_height, _start_x, _start_y, _y_step in pass_specs
    )
    try:
        raw = _bounded_inflate(compressed_parts, expected_bytes)
    except zlib.error as exc:
        raise PNGDecodeError(f"invalid zlib image data: {exc}") from exc

    passes = []
    pixels = bytearray(width * height * 4)
    offset = 0

    for pass_index, (
        pass_width,
        pass_height,
        start_x,
        start_y,
        y_step,
    ) in enumerate(pass_specs):
        rows: list[bytes] = []
        bytes_per_row = pass_width * channels
        for _ in range(pass_height):
            if offset >= len(raw):
                raise PNGDecodeError("truncated scanline data")
            filter_type = raw[offset]
            offset += 1
            scanline_end = offset + bytes_per_row
            if scanline_end > len(raw):
                raise PNGDecodeError("truncated scanline data")
            scanline = bytearray(raw[offset:scanline_end])
            offset = scanline_end
            previous = rows[-1] if rows else b"\x00" * bytes_per_row
            _unfilter(filter_type, scanline, previous, channels)
            row = bytes(scanline)
            rows.append(row)
            _place_pass_row(
                row,
                pixels,
                width,
                channels,
                pass_width,
                start_x,
                start_y + (len(rows) - 1) * y_step,
                _ADAM7_PASSES[pass_index][2] if interlace == 1 else 1,
            )

        passes.append(
            PassEvidence(pass_index, pass_width, pass_height, tuple(rows))
        )

    if offset != len(raw):
        # The bounded inflater also rejects excess compressed bytes, but this
        # guards the scanline layout calculation.
        raise PNGDecodeError("trailing image data after final scanline")

    return PNGImage(
        width=width,
        height=height,
        interlace_method=interlace,
        pixels=bytes(pixels),
        passes=tuple(passes),
    )


def _as_input_bytes(source: bytes | bytearray | memoryview) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, bytearray):
        return bytes(source)
    if isinstance(source, memoryview):
        return source.tobytes()
    raise TypeError("PNG input must be bytes, bytearray or memoryview")


def _parse_ihdr(payload: bytes) -> tuple[int, int, int, int]:
    if len(payload) != 13:
        raise PNGDecodeError("invalid IHDR length")

    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", payload)
    )

    if width == 0 or height == 0:
        raise PNGDecodeError("zero-width or zero-height PNGs are not supported")
    if width > MAX_PNG_DIMENSION or height > MAX_PNG_DIMENSION:
        raise PNGDecodeError("PNG dimensions exceed the format limit")
    if width * height > MAX_PIXELS:
        raise PNGDecodeError("PNG exceeds the 4,000,000 pixel limit")
    if bit_depth != 8:
        raise PNGDecodeError("only 8-bit PNGs are supported")
    if color_type == 2:
        channels = 3
    elif color_type == 6:
        channels = 4
    elif color_type == 3:
        raise PNGDecodeError("palette PNGs are not supported")
    else:
        raise PNGDecodeError("only RGB and RGBA color types are supported")
    if compression != 0:
        raise PNGDecodeError("unsupported PNG compression method")
    if filtering != 0:
        raise PNGDecodeError("unsupported PNG filter method")
    if interlace not in (0, 1):
        raise PNGDecodeError("unsupported PNG interlace method")

    return width, height, channels, interlace


def _validate_unknown_or_ancillary(
    chunk_type: bytes, state: str, ihdr: object | None
) -> None:
    if ihdr is None:
        raise PNGDecodeError(
            f"{_safe_chunk_name(chunk_type)} appears before IHDR"
        )

    # All valid type letters have one upper/lower-case bit each.  Reject
    # malformed type bytes rather than silently treating them as ancillary.
    try:
        text = chunk_type.decode("ascii")
    except UnicodeDecodeError:
        raise PNGDecodeError("invalid PNG chunk type") from None
    if not all(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text):
        raise PNGDecodeError("invalid PNG chunk type")

    ancillary_bit = 0x20
    is_ancillary = bool(chunk_type[0] & ancillary_bit)
    if not is_ancillary:
        raise PNGDecodeError(f"unknown critical chunk {text} is not supported")


def _validate_plte(payload: bytes) -> None:
    if len(payload) == 0 or len(payload) % 3 != 0 or len(payload) // 3 > 256:
        raise PNGDecodeError("invalid PLTE chunk")


def _bounded_inflate(parts: list[bytes], expected_bytes: int) -> bytes:
    if not parts:
        raise PNGDecodeError("missing IDAT image data")

    decompressor = zlib.decompressobj()
    output = bytearray()
    part_index = 0
    part_offset = 0

    # Never ask zlib to produce more than one byte beyond the exact expected
    # scanline length.  Thus malformed or malicious data cannot expand without
    # limit before being identified as trailing image data.
    while True:
        remaining_output = expected_bytes + 1 - len(output)
        if remaining_output <= 0:
            raise PNGDecodeError("trailing compressed image data")

        if part_index < len(parts):
            compressed_input = memoryview(parts[part_index])[part_offset:]
        else:
            compressed_input = b""

        chunk = decompressor.decompress(
            compressed_input, max(1, min(64 * 1024, remaining_output))
        )
        output.extend(chunk)
        if len(output) > expected_bytes:
            raise PNGDecodeError("trailing compressed image data")

        unconsumed = decompressor.unconsumed_tail
        if unconsumed:
            part_offset = len(parts[part_index]) - len(unconsumed)
        else:
            part_index += 1
            part_offset = 0

        if decompressor.eof:
            if (
                len(output) != expected_bytes
                or unconsumed
                or decompressor.unused_data
                or part_index != len(parts)
            ):
                if len(output) != expected_bytes:
                    raise PNGDecodeError("truncated zlib image data")
                raise PNGDecodeError("trailing compressed image data")
            return bytes(output)

        if part_index == len(parts):
            if not chunk:
                raise PNGDecodeError("truncated zlib image data")
            try:
                chunk = decompressor.decompress(b"", max(1, min(64 * 1024, remaining_output)))
            except zlib.error as exc:
                raise PNGDecodeError("truncated zlib image data") from exc
            output.extend(chunk)
            if len(output) > expected_bytes:
                raise PNGDecodeError("trailing compressed image data")
            if not decompressor.eof and not chunk:
                raise PNGDecodeError("truncated zlib image data")


def _unfilter(
    filter_type: int,
    scanline: bytearray,
    previous: bytes,
    channels: int,
) -> None:
    if filter_type not in _SUPPORTED_FILTERS:
        raise PNGDecodeError(f"unsupported scanline filter type {filter_type}")
    if filter_type == 0:
        return

    bpp = channels
    row_length = len(scanline)

    if filter_type == 1:
        for x in range(row_length):
            left = scanline[x - bpp] if x >= bpp else 0
            scanline[x] = (scanline[x] + left) & 0xFF
        return

    if filter_type == 2:
        for x in range(row_length):
            scanline[x] = (scanline[x] + previous[x]) & 0xFF
        return

    if filter_type == 3:
        for x in range(row_length):
            left = scanline[x - bpp] if x >= bpp else 0
            up = previous[x]
            scanline[x] = (scanline[x] + ((left + up) // 2)) & 0xFF
        return

    if filter_type == 4:
        for x in range(row_length):
            left = scanline[x - bpp] if x >= bpp else 0
            up = previous[x]
            upper_left = previous[x - bpp] if x >= bpp else 0
            estimate = left + up - upper_left
            pa = abs(estimate - left)
            pb = abs(estimate - up)
            pc = abs(estimate - upper_left)
            if pa <= pb and pa <= pc:
                predictor = left
            elif pb <= pc:
                predictor = up
            else:
                predictor = upper_left
            scanline[x] = (scanline[x] + predictor) & 0xFF


def _pass_specifications(
    width: int, height: int, interlace: int
) -> list[tuple[int, int, int, int, int]]:
    if interlace == 0:
        return [(width, height, 0, 0, 1)]

    specs = []
    for start_x, start_y, x_step, y_step in _ADAM7_PASSES:
        pass_width = max(0, (width - start_x + x_step - 1) // x_step)
        pass_height = max(0, (height - start_y + y_step - 1) // y_step)
        if pass_width == 0:
            pass_height = 0
        specs.append((pass_width, pass_height, start_x, start_y, y_step))
    return specs


def _place_pass_row(
    row: bytes,
    output: bytearray,
    image_width: int,
    channels: int,
    pass_width: int,
    start_x: int,
    destination_y: int,
    x_step: int,
) -> None:
    for column in range(pass_width):
        source_x = column * channels
        destination_x = start_x + column * x_step
        target = (destination_y * image_width + destination_x) * 4
        if channels == 3:
            output[target : target + 3] = row[source_x : source_x + 3]
            output[target + 3] = 255
        else:
            output[target : target + 4] = row[source_x : source_x + 4]


def _chunk_name(data: bytes | bytearray, pos: int) -> str:
    end = min(pos + 4, len(data))
    return _safe_chunk_name(bytes(data[pos:end]))


def _safe_chunk_name(chunk_type: bytes) -> str:
    try:
        text = chunk_type.decode("ascii")
    except UnicodeDecodeError:
        return repr(chunk_type)
    if all(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text):
        return text
    return repr(chunk_type)
