# Strict PNG decoder

`png_strict` is a dependency-free, pure-backend PNG decoder for the project's
supported subset:

- 8-bit RGB (`color type 2`) and RGBA (`color type 6`)
- non-interlaced and Adam7-interlaced images
- all five PNG scanline filters
- at most 4,000,000 pixels

The decoder returns RGBA bytes plus per-pass evidence. Empty Adam7 passes are
represented with zero rows.

```python
from png_strict import decode_png

image = decode_png(png_bytes)
image.width
image.height
image.pixels          # width * height * 4 bytes
image.passes[0].rows  # reconstructed rows in one interlace pass
```

Invalid signatures, CRCs, chunk ordering, non-contiguous IDAT data, truncated
zlib/scanline streams, trailing image data, palette mode and unknown critical
chunks are rejected. The pixel and decompression limits are enforced before
inflation; zlib output is bounded to the exact expected scanline byte count.

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

When ImageMagick `convert` is available, the fixture tests also compare every
decoded fixture pixel-for-pixel with its RGBA output.
