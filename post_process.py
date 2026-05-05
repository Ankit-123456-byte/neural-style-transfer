import numpy as np
from PIL import Image, ImageEnhance


def color_preserve(content_pil: Image.Image, styled_pil: Image.Image) -> Image.Image:
    """Keep content colors, apply only style texture/luminance.

    Converts both images to YCbCr, swaps only the Y (luminance) channel from
    the styled result into the content image, and converts back to RGB.  This
    preserves the original color palette while adopting the stylised lighting
    and texture patterns.
    """
    c = np.array(content_pil.convert("YCbCr")).astype(float)
    s = np.array(styled_pil.resize(content_pil.size).convert("YCbCr")).astype(float)
    out = c.copy()
    out[:, :, 0] = s[:, :, 0]   # take luminance from styled image
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "YCbCr").convert("RGB")


def histogram_match(source_pil: Image.Image, target_pil: Image.Image) -> Image.Image:
    """Match the output histogram to the style image.

    Applies a per-channel histogram equalization mapping so that the color
    distribution of `source_pil` is transformed to match `target_pil`.  This
    makes colors in the output more vibrant and stylistically consistent.
    """
    src = np.array(source_pil).astype(float)
    tgt = np.array(target_pil.resize(source_pil.size)).astype(float)
    result = np.zeros_like(src)

    for ch in range(3):
        src_hist, _ = np.histogram(src[:, :, ch].flatten(), 256, [0, 256])
        tgt_hist, _ = np.histogram(tgt[:, :, ch].flatten(), 256, [0, 256])
        src_cdf = src_hist.cumsum() / src_hist.sum()
        tgt_cdf = tgt_hist.cumsum() / tgt_hist.sum()

        lut = np.zeros(256)
        j = 0
        for i in range(256):
            while j < 255 and tgt_cdf[j] < src_cdf[i]:
                j += 1
            lut[i] = j

        result[:, :, ch] = lut[src[:, :, ch].clip(0, 255).astype(int)]

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def sharpen(pil_img: Image.Image, strength: float = 2.0) -> Image.Image:
    """Unsharp mask sharpening via PIL ImageEnhance."""
    return ImageEnhance.Sharpness(pil_img).enhance(strength)


def apply_pipeline(
    result_pil: Image.Image,
    content_pil: Image.Image,
    style_pil: Image.Image,
    do_color_preserve: bool = False,
    do_histogram_match: bool = False,
    sharpen_strength: float = 1.0,
) -> Image.Image:
    """Apply all enabled post-processing steps in the correct order.

    Order matters:
      1. Histogram matching — global color distribution correction first.
      2. Color preservation — overrides chrominance with content colors.
      3. Sharpening — applied last so it acts on the final pixel values.
    """
    out = result_pil
    if do_histogram_match:
        out = histogram_match(out, style_pil)
    if do_color_preserve:
        out = color_preserve(content_pil, out)
    if sharpen_strength > 1.0:
        out = sharpen(out, sharpen_strength)
    return out
