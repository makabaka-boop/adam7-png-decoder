from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib

from png_decoder import (
    ADAM7_PASSES,
    PNGDecodeError,
    PNG_SIGNATURE,
    decode_png,
    expected_data_size,
)


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def filter_scanline(raw: bytes, previous: bytes | None, filter_type: int, bpp: int) -> bytes:
    previous = previous or bytes(len(raw))
    result = bytearray(len(raw))
    for x, value in enumerate(raw):
        left = raw[x - bpp] if x >= bpp else 0
        above = previous[x]
        upper_left = previous[x - bpp] if x >= bpp else 0
        if filter_type == 0:
            filtered = value
        elif filter_type == 1:
            filtered = (value - left) & 0xFF
        elif filter_type == 2:
            filtered = (value - above) & 0xFF
        elif filter_type == 3:
            filtered = (value - ((left + above) >> 1)) & 0xFF
        elif filter_type == 4:
            filtered = (value - paeth(left, above, upper_left)) & 0xFF
        else:
            raise AssertionError(filter_type)
        result[x] = filtered
    return bytes([filter_type]) + bytes(result)


def chunk(kind: bytes, data: bytes, corrupt_crc: bool = False) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    if corrupt_crc:
        crc ^= 0x01010101
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def pass_dimensions(width: int, height: int, pass_index: int) -> tuple[int, int]:
    x0, y0, dx, dy = ADAM7_PASSES[pass_index]
    pw = 0 if width <= x0 else (width - x0 + dx - 1) // dx
    ph = 0 if height <= y0 else (height - y0 + dy - 1) // dy
    return pw, ph


def pass_line_counts(width: int, height: int) -> tuple[int, ...]:
    return tuple(
        ph if pw else 0
        for pw, ph in (pass_dimensions(width, height, i) for i in range(7))
    )


def make_png(
    width: int,
    height: int,
    channels: int,
    pixels: bytes,
    *,
    interlace: int = 0,
    filter_chooser=None,
    idat_parts: int = 1,
    extra_chunks: tuple[tuple[bytes, bytes], ...] = (),
    corrupt_crc_chunk: bytes | None = None,
    compressed_override: bytes | None = None,
    omit_iend: bool = False,
    trailing_after_iend: bytes = b"",
) -> bytes:
    if filter_chooser is None:
        filter_chooser = lambda y, pass_index, pass_height: (y + pass_index) % 5

    if interlace == 0:
        passes = [(0, width, height, 0, 0, 1, 1)]
    else:
        passes = []
        for pass_index, (x0, y0, dx, dy) in enumerate(ADAM7_PASSES):
            pw, ph = pass_dimensions(width, height, pass_index)
            passes.append((pass_index, pw, ph, x0, y0, dx, dy))

    filtered = bytearray()
    for _enumerated_index, (pass_index, pw, ph, x0, y0, dx, dy) in enumerate(passes):
        if pw == 0 or ph == 0:
            continue
        previous = None
        for y in range(ph):
            row = bytearray()
            image_y = y0 + y * dy
            for x in range(pw):
                image_x = x0 + x * dx
                start = (image_y * width + image_x) * channels
                row.extend(pixels[start : start + channels])
            encoded = filter_scanline(
                bytes(row), previous, filter_chooser(y, pass_index, ph), channels
            )
            filtered.extend(encoded)
            previous = bytes(row)

    if compressed_override is None:
        compressor = zlib.compressobj(level=9, wbits=15)
        compressed = compressor.compress(bytes(filtered)) + compressor.flush()
    else:
        compressed = compressed_override

    color_type = 2 if channels == 3 else 6
    ihdr_data = struct.pack(
        ">IIBBBBB", width, height, 8, color_type, 0, 0, interlace
    )

    maybe_corrupt = lambda kind, data: chunk(
        kind, data, corrupt_crc=(kind == corrupt_crc_chunk)
    )

    pieces = [PNG_SIGNATURE, maybe_corrupt(b"IHDR", ihdr_data)]
    for kind, data in extra_chunks:
        pieces.append(maybe_corrupt(kind, data))

    idat_chunks = []
    starts = [
        index * len(compressed) // idat_parts for index in range(idat_parts)
    ]
    boundaries = starts + [len(compressed)]
    for start, end in zip(boundaries, boundaries[1:]):
        idat_chunks.append(maybe_corrupt(b"IDAT", compressed[start:end]))

    pieces.extend(idat_chunks)
    if not omit_iend:
        pieces.append(maybe_corrupt(b"IEND", b""))
    pieces.append(trailing_after_iend)
    return b"".join(pieces)


def sample_pixels(width: int, height: int, channels: int) -> bytes:
    pixels = bytearray(width * height * channels)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * channels
            pixels[offset] = (x * 37 + y * 11) & 0xFF
            pixels[offset + 1] = (x * 7 + y * 23 + 64) & 0xFF
            pixels[offset + 2] = (x * 19 - y * 13 + 128) & 0xFF
            if channels == 4:
                pixels[offset + 3] = (x * 5 + y * 31 + 7) & 0xFF
    return bytes(pixels)


class PNGDecoderTests(unittest.TestCase):
    def assert_decodes_expected(
        self, width: int, height: int, channels: int, interlace: int
    ):
        pixels = sample_pixels(width, height, channels)
        encoded = make_png(
            width, height, channels, pixels, interlace=interlace, idat_parts=3
        )
        decoded = decode_png(encoded)
        expected = pixels if channels == 4 else rgba_from_rgb(pixels)

        self.assertEqual(decoded.width, width)
        self.assertEqual(decoded.height, height)
        self.assertEqual(decoded.interlace_method, interlace)
        self.assertEqual(decoded.pixels, expected)
        if interlace:
            self.assertEqual(
                decoded.pass_lines,
                tuple(pass_line_counts(width, height)),
            )
        else:
            self.assertEqual(decoded.pass_lines, (height,))
        return encoded, expected

    def test_non_interlaced_sizes_colors_and_all_filters(self):
        for channels in (3, 4):
            for width, height in ((1, 1), (3, 2), (8, 8), (13, 9)):
                with self.subTest(channels=channels, size=(width, height)):
                    self.assert_decodes_expected(width, height, channels, 0)

    def test_adam7_sizes_including_empty_passes(self):
        for channels in (3, 4):
            # 1x1 leaves six passes empty; 2x2 still leaves the first three empty.
            for width, height in ((1, 1), (2, 2), (13, 9), (16, 16)):
                with self.subTest(channels=channels, size=(width, height)):
                    self.assert_decodes_expected(width, height, channels, 1)

    def test_explicit_every_filter_non_interlaced_and_interlaced(self):
        for interlace in (0, 1):
            for filter_type in range(5):
                pixels = sample_pixels(11, 7, 4)
                encoded = make_png(
                    11,
                    7,
                    4,
                    pixels,
                    interlace=interlace,
                    filter_chooser=lambda y, p, h, f=filter_type: f,
                )
                with self.subTest(interlace=interlace, filter=filter_type):
                    self.assertEqual(decode_png(encoded).pixels, pixels)

    def test_accepts_binary_stream(self):
        pixels = sample_pixels(4, 3, 3)
        encoded = make_png(4, 3, 3, pixels)
        self.assertEqual(decode_png(io.BytesIO(encoded)).pixels, rgba_from_rgb(pixels))

    def test_rejects_bad_signature(self):
        encoded = bytearray(make_png(1, 1, 3, sample_pixels(1, 1, 3)))
        encoded[0] ^= 1
        with self.assertRaisesRegex(PNGDecodeError, "signature"):
            decode_png(bytes(encoded))

    def test_rejects_ihdr_not_first(self):
        pixels = sample_pixels(1, 1, 3)
        body = make_png(1, 1, 3, pixels)[len(PNG_SIGNATURE) :]
        encoded = PNG_SIGNATURE + chunk(b"acTL", struct.pack(">I", 1)) + body
        with self.assertRaisesRegex(PNGDecodeError, "IHDR"):
            decode_png(encoded)

    def test_rejects_unknown_critical_chunk(self):
        pixels = sample_pixels(1, 1, 3)
        encoded = make_png(
            1, 1, 3, pixels, extra_chunks=((b"BzTG", b"nope"),)
        )
        with self.assertRaisesRegex(PNGDecodeError, "unknown critical"):
            decode_png(encoded)

    def test_accepts_unknown_ancillary_chunk(self):
        pixels = sample_pixels(2, 2, 4)
        encoded = make_png(
            2, 2, 4, pixels, extra_chunks=((b"xyZA", b"ignored"),)
        )
        self.assertEqual(decode_png(encoded).pixels, pixels)

    def test_rejects_palette_mode(self):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0)
        compressed = zlib.compress(b"\x00\x00")
        encoded = (
            PNG_SIGNATURE
            + chunk(b"IHDR", ihdr)
            + chunk(b"PLTE", b"\x00\x00\x00\xff\xff\xff")
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )
        with self.assertRaisesRegex(PNGDecodeError, "palette"):
            decode_png(encoded)

    def test_rejects_non_eight_bit_or_other_color_types(self):
        for bit_depth, color_type in ((1, 0), (2, 2), (8, 0), (8, 4), (16, 6)):
            ihdr = struct.pack(
                ">IIBBBBB", 1, 1, bit_depth, color_type, 0, 0, 0
            )
            encoded = (
                PNG_SIGNATURE
                + chunk(b"IHDR", ihdr)
                + chunk(b"IDAT", zlib.compress(b""))
                + chunk(b"IEND", b"")
            )
            with self.subTest(bit_depth=bit_depth, color_type=color_type):
                with self.assertRaises(PNGDecodeError):
                    decode_png(encoded)

    def test_dimension_and_pixel_limits_before_inflation(self):
        huge = 3000 * 3000  # 9,000,000 pixels
        # The compressed payload is a tiny bomb.  It must be rejected by header
        # limits before decompression, regardless of exact stream geometry.
        bomb = zlib.compressobj(9, zlib.DEFLATED, 15)
        bomb_data = bomb.compress(b"\x00" * 1_000_000) + bomb.flush()
        ihdr = struct.pack(">IIBBBBB", 3000, 3000, 8, 6, 0, 0, 0)
        encoded = (
            PNG_SIGNATURE
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", bomb_data)
            + chunk(b"IEND", b"")
        )
        self.assertLess(len(bomb_data), 20_000)
        with self.assertRaisesRegex(PNGDecodeError, "4,000,000"):
            decode_png(encoded)
        self.assertEqual(huge, 9_000_000)

    def test_rejects_zero_dimensions(self):
        for width, height in ((0, 1), (1, 0)):
            ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
            encoded = PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
            with self.subTest(size=(width, height)):
                with self.assertRaisesRegex(PNGDecodeError, "non-zero"):
                    decode_png(encoded)

    def test_rejects_corrupt_crc(self):
        pixels = sample_pixels(2, 2, 4)
        encoded = make_png(2, 2, 4, pixels, corrupt_crc_chunk=b"IDAT")
        with self.assertRaisesRegex(PNGDecodeError, "CRC"):
            decode_png(encoded)

    def test_rejects_corrupt_ancillary_crc(self):
        pixels = sample_pixels(2, 2, 4)
        encoded = make_png(
            2,
            2,
            4,
            pixels,
            extra_chunks=((b"xyZA", b"ignored"),),
            corrupt_crc_chunk=b"xyZA",
        )
        with self.assertRaisesRegex(PNGDecodeError, "CRC"):
            decode_png(encoded)

    def test_rejects_huge_chunk_length_without_allocating_payload(self):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        encoded = (
            PNG_SIGNATURE
            + chunk(b"IHDR", ihdr)
            + struct.pack(">I", 0xFFFFFFFF)
            + b"xyZA"
            + b"\x00"
        )
        with self.assertRaises(PNGDecodeError):
            decode_png(encoded)

    def test_rejects_corrupt_compressed_stream(self):
        pixels = sample_pixels(4, 4, 4)
        good = bytearray(make_png(4, 4, 4, pixels))
        first_idat = good.index(b"IDAT")
        data_length = struct.unpack(">I", good[first_idat - 4 : first_idat])[0]
        data_start = first_idat + 4
        # Corrupt compressed payload data, then recalculate the chunk CRC so
        # that chunk validation passes but zlib/deflate validation fails.
        good[data_start + min(3, data_length - 1)] ^= 0x40
        crc = zlib.crc32(bytes(good[first_idat : data_start + data_length])) & 0xFFFFFFFF
        crc_start = data_start + data_length
        good[crc_start : crc_start + 4] = struct.pack(">I", crc)
        with self.assertRaisesRegex(PNGDecodeError, "zlib"):
            decode_png(bytes(good))

    def test_rejects_truncated_compressed_stream(self):
        pixels = sample_pixels(5, 3, 4)
        filtered_len = expected_data_size(5, 3, 4, 0)
        raw = b"\x00" * filtered_len
        compressor = zlib.compressobj(9, zlib.DEFLATED, 15)
        compressed = compressor.compress(raw) + compressor.flush()
        truncated = make_png(
            5, 3, 4, pixels, compressed_override=compressed[:-3]
        )
        with self.assertRaisesRegex(PNGDecodeError, "truncated zlib"):
            decode_png(truncated)

    def test_rejects_short_compressed_stream(self):
        pixels = sample_pixels(3, 2, 4)
        shorter = zlib.compress(b"\x00" * 2)
        encoded = make_png(3, 2, 4, pixels, compressed_override=shorter)
        with self.assertRaisesRegex(PNGDecodeError, "length mismatch|truncated zlib"):
            decode_png(encoded)

    def test_rejects_trailing_compressed_image_data(self):
        pixels = sample_pixels(1, 1, 4)
        encoded = make_png(
            1,
            1,
            4,
            pixels,
            compressed_override=zlib.compress(b"\x00" * 10),
        )
        with self.assertRaisesRegex(PNGDecodeError, "trailing image data|length mismatch"):
            decode_png(encoded)

    def test_rejects_truncated_chunk_stream(self):
        pixels = sample_pixels(4, 3, 4)
        encoded = make_png(4, 3, 4, pixels)
        # Cut inside a real chunk rather than merely omitting IEND.
        with self.assertRaisesRegex(PNGDecodeError, "truncated"):
            decode_png(encoded[: len(encoded) - 20])

    def test_rejects_non_consecutive_idat(self):
        pixels = sample_pixels(2, 2, 4)
        good = bytearray(make_png(2, 2, 4, pixels, idat_parts=2))
        first_idat = good.index(b"IDAT")
        length = struct.unpack(">I", good[first_idat - 4 : first_idat])[0]
        first_idat_end = first_idat + 4 + length + 4
        good[first_idat_end:first_idat_end] = chunk(b"xyZA", b"split")
        with self.assertRaisesRegex(PNGDecodeError, "IDAT chunks must be consecutive"):
            decode_png(bytes(good))

    def test_rejects_chunk_order_missing_iend(self):
        pixels = sample_pixels(1, 1, 3)
        encoded = make_png(1, 1, 3, pixels, omit_iend=True)
        with self.assertRaisesRegex(PNGDecodeError, "missing IEND|truncated"):
            decode_png(encoded)

    def test_rejects_second_iend(self):
        pixels = sample_pixels(1, 1, 4)
        encoded = make_png(1, 1, 4, pixels) + chunk(b"IEND", b"")
        with self.assertRaisesRegex(PNGDecodeError, "missing IEND|IEND"):
            decode_png(encoded)

    def test_rejects_iend_before_idat(self):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
        encoded = (
            PNG_SIGNATURE
            + chunk(b"IHDR", ihdr)
            + chunk(b"IEND", b"")
            + chunk(b"IDAT", zlib.compress(b"\x00" * 5))
        )
        with self.assertRaisesRegex(PNGDecodeError, "IEND|IDAT"):
            decode_png(encoded)

    def test_rejects_bytes_after_iend(self):
        pixels = sample_pixels(1, 1, 3)
        encoded = make_png(1, 1, 3, pixels, trailing_after_iend=b"x")
        with self.assertRaisesRegex(PNGDecodeError, "trailing"):
            decode_png(encoded)

    def test_rejects_invalid_filter_type(self):
        width = height = 2
        channels = 4
        filtered = bytearray()
        for y in range(height):
            filtered.append(5)
            filtered.extend(b"\x00" * (width * channels))
        encoded = make_png(
            width,
            height,
            channels,
            sample_pixels(width, height, channels),
            compressed_override=zlib.compress(bytes(filtered)),
        )
        with self.assertRaisesRegex(PNGDecodeError, "filter type 5"):
            decode_png(encoded)

    def test_adam7_expected_pass_line_evidence(self):
        pixels = sample_pixels(13, 9, 4)
        encoded = make_png(13, 9, 4, pixels, interlace=1)
        decoded = decode_png(encoded)
        self.assertEqual(decoded.pass_lines, (2, 2, 1, 3, 2, 5, 4))

    @unittest.skipUnless(shutil.which("convert"), "ImageMagick convert is unavailable")
    def test_compares_fixtures_with_trusted_imagemagick_decode(self):
        # Our fixture encoder exercises split IDAT and all five filters. This
        # independently asks a widely used trusted decoder to interpret it.
        for channels in (3, 4):
            for interlace in (0, 1):
                width, height = (13, 9)
                pixels = sample_pixels(width, height, channels)
                encoded = make_png(
                    width,
                    height,
                    channels,
                    pixels,
                    interlace=interlace,
                    idat_parts=4,
                )
                with tempfile.TemporaryDirectory() as directory:
                    png_path = os.path.join(directory, "fixture.png")
                    rgba_path = os.path.join(directory, "fixture.rgba")
                    with open(png_path, "wb") as output:
                        output.write(encoded)
                    subprocess.run(
                        [
                            "convert",
                            png_path,
                            "-depth",
                            "8",
                            f"rgba:{rgba_path}",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    with open(rgba_path, "rb") as readable:
                        trusted = readable.read()
                with self.subTest(channels=channels, interlace=interlace):
                    decoded = decode_png(encoded)
                    self.assertEqual(decoded.pixels, trusted)


def rgba_from_rgb(rgb: bytes) -> bytes:
    rgba = bytearray(len(rgb) // 3 * 4)
    for i in range(len(rgb) // 3):
        rgba[i * 4 : i * 4 + 3] = rgb[i * 3 : i * 3 + 3]
        rgba[i * 4 + 3] = 255
    return bytes(rgba)


if __name__ == "__main__":
    unittest.main()
