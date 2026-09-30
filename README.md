# Strict backend PNG decoder

`png_decoder.py` is a dependency-free Python implementation for the requested
PNG subset:

- 8-bit RGB (`color type 2`) and RGBA (`color type 6`)
- non-interlaced images (`interlace method 0`)
- Adam7 interlaced images (`interlace method 1`)
- all five PNG scanline filters: None, Sub, Up, Average, Paeth
- split but consecutive `IDAT` chunks

RGB output is expanded with an alpha value of 255 so successful decodes always
return packed RGBA bytes.

## Safety and rejection behavior

The decoder checks dimensions before collecting or inflating image data:

- width and height must be non-zero
- dimensions must fit in the PNG 31-bit range
- images may contain at most 4,000,000 pixels

It explicitly rejects:

- invalid PNG signature
- malformed `IHDR`
- non-8-bit depth, unsupported color types, palette images, compression,
  filter, or interlace methods
- unknown critical chunks
- invalid chunk order, including non-consecutive `IDAT`, early/duplicate
  `IEND`, and bytes after `IEND`
- CRC failures in all chunks, including skipped ancillary chunks
- truncated chunk or zlib streams
- compressed streams that are too short or that contain trailing image data
- invalid scanline filter bytes
- `tRNS` chunks (RGB-with-transparency is not silently converted to opaque
  output)

Inflation is bounded by the exact pre-filter byte count derived from image
dimensions and interlace geometry. Chunk payloads are also streamed in blocks
with a compressed-data safety cap, so a declared giant chunk or a small
compression bomb cannot allocate unbounded memory first.

## Result metadata

`decode_png` returns a `DecodedPNG`:

```python
from png_decoder import decode_png

result = decode_png(image_bytes)
result.width
result.height
result.pixels      # bytes, four RGBA bytes per pixel, top-to-bottom
result.pass_lines  # encoded scanline count per interlace pass
```

`pass_lines` has one entry for a non-interlaced image and always seven entries
for Adam7. Empty Adam7 passes report zero encoded lines. For example, a 13×9
image reports `(2, 2, 1, 3, 2, 5, 4)`.

## Tests

Run:

```sh
python3 -m unittest -v
```

The suite covers multiple sizes, RGB/RGBA, every filter, split IDAT, empty
Adam7 passes, CRC damage, zlib damage/truncation/trailing data, and chunk-order
violations. When ImageMagick `convert` is installed, generated fixtures are
also decoded by that trusted implementation and compared pixel-for-pixel with
this decoder.
