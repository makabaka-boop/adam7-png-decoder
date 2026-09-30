from __future__ import annotations

from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib

from png_strict import PNGDecodeError, decode_png

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ADAM7 = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


def expected_pixels(width: int, height: int, channels: int) -> bytes:
    result = bytearray()
    for y in range(height):
        for x in range(width):
            r = (x * 37 + y * 19) & 0xFF
            g = (x * 11 + y * 53 + 7) & 0xFF
            b = (x * 71 + y * 5 + 31) & 0xFF
            result.extend((r, g, b))
            if channels == 4:
                result.append((x * 13 + y * 17 + 9) & 0xFF)
    return bytes(result)


def to_rgba(raw: bytes, channels: int) -> bytes:
    if channels == 4:
        return raw
    result = bytearray()
    for offset in range(0, len(raw), 3):
        result.extend(raw[offset : offset + 3])
        result.append(255)
    return bytes(result)


def paeth(left: int, up: int, upper_left: int) -> int:
    value = left + up - upper_left
    pa = abs(value - left)
    pb = abs(value - up)
    pc = abs(value - upper_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return upper_left


def encode_filter(filter_type: int, current: bytes, previous: bytes, bpp: int) -> bytes:
    result = bytearray(len(current))
    for i, value in enumerate(current):
        left = current[i - bpp] if i >= bpp else 0
        up = previous[i]
        upper_left = previous[i - bpp] if i >= bpp else 0
        if filter_type == 0:
            filtered = value
        elif filter_type == 1:
            filtered = (value - left) & 0xFF
        elif filter_type == 2:
            filtered = (value - up) & 0xFF
        elif filter_type == 3:
            filtered = (value - ((left + up) // 2)) & 0xFF
        else:
            filtered = (value - paeth(left, up, upper_left)) & 0xFF
        result[i] = filtered
    return bytes(result)


def chunk(chunk_type: bytes, payload: bytes, corrupt_crc: bool = False) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    if corrupt_crc:
        crc ^= 0x01
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def ihdr(width: int, height: int, color_type: int = 2, interlace: int = 0,
         bit_depth: int = 8) -> bytes:
    return chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace),
    )


def pass_dimensions(width: int, height: int):
    for start_x, start_y, step_x, step_y in ADAM7:
        pass_width = max(0, (width - start_x + step_x - 1) // step_x)
        pass_height = max(0, (height - start_y + step_y - 1) // step_y)
        if pass_width == 0:
            pass_height = 0
        yield pass_width, pass_height, start_x, start_y, step_x, step_y


def make_png(
    width: int,
    height: int,
    *,
    color_type: int = 2,
    interlace: int = 0,
    idat_parts: int = 1,
    filter_pattern=lambda row_number: row_number % 5,
) -> bytes:
    channels = {2: 3, 6: 4}[color_type]
    pixels = expected_pixels(width, height, channels)

    if interlace == 0:
        passes = [(width, height, 0, 0, 1, 1)]
    else:
        passes = list(pass_dimensions(width, height))

    raw = bytearray()
    global_row = 0
    for pass_width, pass_height, start_x, start_y, step_x, step_y in passes:
        previous = b"\x00" * (pass_width * channels)
        for row_in_pass in range(pass_height):
            current = bytearray()
            y = start_y + row_in_pass * step_y
            for column in range(pass_width):
                x = start_x + column * step_x
                current.extend(
                    pixels[(y * width + x) * channels : (y * width + x + 1) * channels]
                )
            current = bytes(current)
            filter_type = filter_pattern(global_row)
            raw.append(filter_type)
            raw.extend(encode_filter(filter_type, current, previous, channels))
            previous = current
            global_row += 1

    compressed = zlib.compress(bytes(raw), 9)
    boundaries = [
        (len(compressed) * index) // idat_parts for index in range(idat_parts + 1)
    ]
    idat = b"".join(
        chunk(b"IDAT", compressed[boundaries[i] : boundaries[i + 1]])
        for i in range(idat_parts)
    )
    return PNG_SIGNATURE + ihdr(width, height, color_type, interlace) + idat + chunk(
        b"IEND", b""
    )


def parse_chunks(data: bytes):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    pos = len(PNG_SIGNATURE)
    while pos < len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        end = pos + 8 + length
        yield pos, end + 4, data[pos + 4 : pos + 8], data[pos + 8 : end]
        pos = end + 4


def replace_first_chunk(data: bytes, chunk_type: bytes, payload: bytes) -> bytes:
    for start, _end, current_type, _payload in parse_chunks(data):
        if current_type == chunk_type:
            prefix = data[:start]
            suffix = data[start + 8 + len(_payload) + 4 :]
            return prefix + chunk(chunk_type, payload) + suffix
    raise AssertionError(f"missing {chunk_type!r}")


def insert_after_ihdr(data: bytes, new_chunk: bytes) -> bytes:
    chunks = list(parse_chunks(data))
    end = chunks[0][1]
    return data[:end] + new_chunk + data[end:]


def insert_between_idat(data: bytes, new_chunk: bytes) -> bytes:
    idat_ends = [end for _start, end, current_type, _p in parse_chunks(data) if current_type == b"IDAT"]
    if len(idat_ends) < 2:
        raise AssertionError("need at least two IDAT chunks")
    position = idat_ends[0]
    return data[:position] + new_chunk + data[position:]


def insert_after_idat(data: bytes, new_chunk: bytes) -> bytes:
    for _start, end, current_type, _payload in parse_chunks(data):
        if current_type == b"IDAT":
            return data[:end] + new_chunk + data[end:]
    raise AssertionError("missing IDAT")


def with_corrupt_first_crc(data: bytes, chunk_type: bytes) -> bytes:
    for start, end, current_type, _payload in parse_chunks(data):
        if current_type == chunk_type:
            position = end - 1
            return data[:position] + bytes([data[position] ^ 0xFF]) + data[end:]
    raise AssertionError("chunk not found")


def modify_idat_payload(data: bytes, callback) -> bytes:
    for start, end, current_type, payload in parse_chunks(data):
        if current_type == b"IDAT":
            payload = callback(bytearray(payload))
            prefix = data[:start]
            suffix = data[end:]
            return prefix + chunk(b"IDAT", bytes(payload)) + suffix
    raise AssertionError("missing IDAT")


def trusted_decode_rgba(png_data: bytes, width: int, height: int) -> bytes | None:
    convert = shutil.which("convert")
    if convert is None:
        return None
    with tempfile.TemporaryDirectory() as directory:
        png_path = Path(directory) / "input.png"
        rgba_path = Path(directory) / "output.rgba"
        png_path.write_bytes(png_data)
        subprocess.run(
            [convert, str(png_path), f"RGBA:{rgba_path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        result = rgba_path.read_bytes()
    if len(result) != width * height * 4:
        raise AssertionError("trusted decoder returned an unexpected size")
    return result


class PNGDecoderTests(unittest.TestCase):
    def assert_matches_trusted_decoder(self, png_data: bytes, width: int, height: int):
        image = decode_png(png_data)
        self.assertEqual((image.width, image.height), (width, height))
        self.assertEqual(len(image.pixels), width * height * 4)
        trusted = trusted_decode_rgba(png_data, width, height)
        if trusted is None:
            self.skipTest("ImageMagick convert is not available")
        self.assertEqual(image.pixels, trusted)

    def test_noninterlaced_rgb_all_filters(self):
        data = make_png(5, 5, color_type=2, interlace=0, idat_parts=3)
        image = decode_png(data)
        self.assertEqual(image.interlace_method, 0)
        self.assertEqual(len(image.passes), 1)
        self.assertEqual(image.passes[0].row_count, 5)
        self.assertEqual([len(row) for row in image.passes[0].rows], [15] * 5)
        self.assertEqual(image.pixels, to_rgba(expected_pixels(5, 5, 3), 3))
        self.assert_matches_trusted_decoder(data, 5, 5)

    def test_noninterlaced_rgba_size_and_filters(self):
        data = make_png(7, 6, color_type=6, interlace=0, idat_parts=4)
        image = decode_png(data)
        self.assertEqual(image.pixels, expected_pixels(7, 6, 4))
        self.assertEqual(image.passes[0].row_count, 6)
        self.assert_matches_trusted_decoder(data, 7, 6)

    def test_adam7_with_multiple_empty_passes(self):
        data = make_png(1, 1, color_type=6, interlace=1)
        image = decode_png(data)
        self.assertEqual(image.interlace_method, 1)
        self.assertEqual([pass_.row_count for pass_ in image.passes], [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual([pass_.width for pass_ in image.passes], [1, 0, 1, 0, 1, 0, 1])
        self.assertEqual([pass_.height for pass_ in image.passes], [1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(image.pixels, expected_pixels(1, 1, 4))
        self.assert_matches_trusted_decoder(data, 1, 1)

    def test_adam7_rgb_partial_passes(self):
        data = make_png(2, 1, color_type=2, interlace=1, idat_parts=2)
        image = decode_png(data)
        self.assertEqual([pass_.row_count for pass_ in image.passes], [1, 0, 0, 0, 0, 1, 0])
        self.assertEqual([pass_.width for pass_ in image.passes], [1, 0, 1, 0, 1, 1, 2])
        self.assertEqual([pass_.height for pass_ in image.passes], [1, 0, 0, 0, 0, 1, 0])
        self.assertEqual(image.pixels, to_rgba(expected_pixels(2, 1, 3), 3))
        self.assert_matches_trusted_decoder(data, 2, 1)

    def test_adam7_large_grid_all_filters_and_pass_rows(self):
        data = make_png(11, 9, color_type=6, interlace=1, idat_parts=5)
        image = decode_png(data)
        self.assertEqual(
            [pass_.row_count for pass_ in image.passes], [2, 2, 1, 3, 2, 5, 4]
        )
        self.assertEqual(
            [pass_.width for pass_ in image.passes], [2, 1, 3, 3, 6, 5, 11]
        )
        self.assertEqual([len(row) for row in image.passes[6].rows], [44] * 4)
        self.assertEqual(image.pixels, expected_pixels(11, 9, 4))
        self.assert_matches_trusted_decoder(data, 11, 9)

    def test_trusted_decoder_matrix(self):
        sizes = [(1, 1), (1, 7), (7, 1), (8, 8), (9, 10), (12, 5)]
        for width, height in sizes:
            for color_type in (2, 6):
                for interlace in (0, 1):
                    with self.subTest(
                        width=width,
                        height=height,
                        color_type=color_type,
                        interlace=interlace,
                    ):
                        channels = 3 if color_type == 2 else 4
                        data = make_png(
                            width,
                            height,
                            color_type=color_type,
                            interlace=interlace,
                            idat_parts=2,
                        )
                        image = decode_png(data)
                        expected = expected_pixels(width, height, channels)
                        self.assertEqual(
                            image.pixels,
                            to_rgba(expected, channels),
                        )
                        self.assert_matches_trusted_decoder(data, width, height)

    def test_decompression_bomb_rejected_without_unbounded_output(self):
        # The dimensions describe only four pixels, while the stream expands to
        # far more than the calculated scanline capacity.
        raw = bytes([0]) + b"\x00" * 12
        data = (
            PNG_SIGNATURE
            + ihdr(4, 1)
            + chunk(b"IDAT", zlib.compress(raw * 1_000_000, 9))
            + chunk(b"IEND", b"")
        )
        with self.assertRaisesRegex(PNGDecodeError, "trailing compressed image data"):
            decode_png(data)

    def test_crc_corruption_rejected(self):
        data = with_corrupt_first_crc(make_png(3, 3, idat_parts=2), b"IDAT")
        with self.assertRaisesRegex(PNGDecodeError, "CRC mismatch"):
            decode_png(data)

    def test_compressed_stream_corruption_rejected(self):
        data = make_png(8, 8, color_type=6, idat_parts=1)

        def corrupt(payload):
            payload[2] ^= 0xFF
            return payload

        data = modify_idat_payload(data, corrupt)
        with self.assertRaisesRegex(PNGDecodeError, "invalid zlib"):
            decode_png(data)

    def test_truncated_zlib_stream_rejected(self):
        data = make_png(8, 8)
        data = modify_idat_payload(data, lambda payload: payload[:-4])
        with self.assertRaisesRegex(PNGDecodeError, "truncated zlib"):
            decode_png(data)

    def test_truncated_png_file_rejected(self):
        data = make_png(4, 4)[:-13]
        with self.assertRaises(PNGDecodeError):
            decode_png(data)

    def test_trailing_compressed_image_data_rejected(self):
        data = make_png(4, 4)
        data = modify_idat_payload(data, lambda payload: payload + b"\x00")
        with self.assertRaisesRegex(PNGDecodeError, "trailing compressed image data"):
            decode_png(data)

    def test_bytes_after_iend_rejected(self):
        data = make_png(2, 2) + b"extra"
        with self.assertRaisesRegex(PNGDecodeError, "after IEND"):
            decode_png(data)

    def test_noncontiguous_idat_rejected(self):
        data = make_png(4, 4, idat_parts=2)
        data = insert_between_idat(data, chunk(b"tEXt", b"comment\0value"))
        with self.assertRaisesRegex(PNGDecodeError, "not contiguous|between IDAT"):
            decode_png(data)

    def test_unknown_critical_chunk_rejected(self):
        data = insert_after_ihdr(make_png(2, 2), chunk(b"XzTX", b"critical"))
        with self.assertRaisesRegex(PNGDecodeError, "unknown critical chunk"):
            decode_png(data)

    def test_palette_mode_rejected(self):
        raw = bytes([0, 0, 1, 2, 3, 1, 2, 3, 0])
        data = (
            PNG_SIGNATURE
            + ihdr(4, 2, color_type=3)
            + chunk(b"PLTE", bytes(range(12)))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
        with self.assertRaisesRegex(PNGDecodeError, "palette PNGs are not supported"):
            decode_png(data)

    def test_ancillary_chunk_after_idat_but_before_iend_is_allowed(self):
        data = insert_after_idat(make_png(2, 2, idat_parts=1), chunk(b"tEXt", b"x\0y"))
        # There is only one IDAT, so insertion happens immediately after it;
        # this is ordinary post-IDAT ancillary data, not an IDAT split.
        image = decode_png(data)
        self.assertEqual(image.pixels, to_rgba(expected_pixels(2, 2, 3), 3))

    def test_iend_without_idat_rejected(self):
        data = PNG_SIGNATURE + ihdr(2, 2) + chunk(b"IEND", b"")
        with self.assertRaisesRegex(PNGDecodeError, "before IDAT|missing IDAT"):
            decode_png(data)

    def test_invalid_filter_rejected(self):
        raw = bytes([5, 1, 2, 3])
        data = (
            PNG_SIGNATURE
            + ihdr(1, 1)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
        with self.assertRaisesRegex(PNGDecodeError, "unsupported scanline filter"):
            decode_png(data)

    def test_unsupported_bit_depth_and_color_type(self):
        for color_type in (0, 4):
            with self.subTest(color_type=color_type):
                data = PNG_SIGNATURE + ihdr(1, 1, color_type=color_type)
                with self.assertRaisesRegex(PNGDecodeError, "RGB and RGBA"):
                    decode_png(data)
        data = PNG_SIGNATURE + ihdr(1, 1, bit_depth=16)
        with self.assertRaisesRegex(PNGDecodeError, "only 8-bit"):
            decode_png(data)

    def test_dimension_limit_checked_before_image_data(self):
        data = PNG_SIGNATURE + ihdr(2001, 2000) + chunk(b"IDAT", b"not inflated")
        with self.assertRaisesRegex(PNGDecodeError, "4,000,000 pixel"):
            decode_png(data)

    def test_zero_dimensions_rejected(self):
        data = PNG_SIGNATURE + ihdr(0, 1)
        with self.assertRaisesRegex(PNGDecodeError, "zero-width"):
            decode_png(data)

    def test_bad_signature_and_unsupported_interlace(self):
        with self.assertRaisesRegex(PNGDecodeError, "invalid PNG signature"):
            decode_png(b"not a png")
        payload = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 2)
        data = PNG_SIGNATURE + chunk(b"IHDR", payload)
        with self.assertRaisesRegex(PNGDecodeError, "interlace"):
            decode_png(data)


if __name__ == "__main__":
    unittest.main()
