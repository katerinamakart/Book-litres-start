#!/usr/bin/env python3
"""Generate a paint-by-numbers template from an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage


def is_skin_rgb(rgb: np.ndarray) -> bool:
    r, g, b = rgb / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c
    if max_c == 0:
        return False
    h = 0.0
    if delta:
        if max_c == r:
            h = ((g - b) / delta) % 6
        elif max_c == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        h *= 60
    lightness = (max_c + min_c) / 2
    saturation = delta / (1 - abs(2 * lightness - 1) + 1e-6)
    return h <= 48 and 0.1 <= saturation <= 0.62 and 0.22 <= lightness <= 0.9 and r > g and r > b * 0.95


def suppress_texture(image: Image.Image) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    gray = arr.mean(axis=2)
    variance = ndimage.generic_filter(gray, np.var, size=5, mode="nearest")
    blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
    threshold = np.percentile(variance, 55)
    mask = (variance > threshold)[..., None]
    mixed = np.where(mask, arr * 0.18 + blurred * 0.82, arr * 0.85 + blurred * 0.15)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def enhance_image(image: Image.Image, saturation: float = 1.9, contrast: float = 1.16) -> Image.Image:
    """Make colors vivid while keeping textures calmer."""
    from PIL import ImageEnhance

    image = suppress_texture(image)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return ImageEnhance.Color(image).enhance(saturation)


def vibrant_palette(palette: np.ndarray, min_saturation: float = 0.38) -> np.ndarray:
    """Boost palette saturation while keeping whites and near-grays natural."""
    result = palette.astype(np.float32).copy()
    for i, rgb in enumerate(result):
        r, g, b = rgb / 255.0
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        delta = max_c - min_c
        lightness = (max_c + min_c) / 2.0
        if delta < 0.03 or lightness > 0.94:
            continue
        if is_skin_rgb(rgb):
            continue

        saturation = delta / (1.0 - abs(2.0 * lightness - 1.0) + 1e-6)
        boosted = min(1.0, saturation * 1.85)
        boosted = max(min_saturation, boosted)

        if lightness < 0.5:
            chroma = boosted * (2.0 * lightness)
        else:
            chroma = boosted * (2.0 - 2.0 * lightness)

        # Simple RGB spread from gray axis.
        gray = lightness
        scale = chroma / (delta + 1e-6)
        result[i, 0] = np.clip(gray + (r - gray) * scale, 0, 1) * 255
        result[i, 1] = np.clip(gray + (g - gray) * scale, 0, 1) * 255
        result[i, 2] = np.clip(gray + (b - gray) * scale, 0, 1) * 255

    return result.astype(np.uint8)


def simplify_labels(labels: np.ndarray, min_region: int = 500, smooth_size: int = 5) -> np.ndarray:
    """Remove tiny regions and smooth borders for cleaner paint-by-numbers lines."""
    result = labels.copy()

    for _ in range(2):
        for color_id in np.unique(result):
            mask = result == color_id
            component_ids, count = ndimage.label(mask)
            if count == 0:
                continue
            sizes = ndimage.sum(mask, component_ids, range(1, count + 1))
            for component, size in enumerate(sizes, start=1):
                if size >= min_region:
                    continue
                component_mask = component_ids == component
                dilated = ndimage.binary_dilation(component_mask, iterations=2)
                neighbors = result[dilated & ~component_mask]
                if neighbors.size == 0:
                    continue
                result[component_mask] = int(np.bincount(neighbors).argmax())

    def mode(values: np.ndarray) -> float:
        values = values.astype(np.int64)
        return float(np.bincount(values).argmax())

    if smooth_size > 1:
        result = ndimage.generic_filter(result, mode, size=smooth_size, mode="nearest").astype(np.int32)

    return result


def detail_mask(image: Image.Image) -> np.ndarray:
    """Highlight faces and detailed areas that should stay sharper."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    edges = ndimage.gaussian_filter(np.hypot(gx, gy), sigma=2.0)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)

    height, width = gray.shape
    y, x = np.ogrid[:height, :width]
    center_y = height * 0.42
    center_x = width * 0.5
    focus = np.exp(-(((x - center_x) / (width * 0.24)) ** 2 + ((y - center_y) / (height * 0.3)) ** 2))

    return np.clip(edges * 0.65 + focus * 0.35, 0.0, 1.0)


def labels_at_scale(
    enhanced: Image.Image,
    palette: np.ndarray,
    factor: int,
    smooth_size: int,
) -> np.ndarray:
    if factor > 1:
        small = enhanced.resize(
            (max(1, enhanced.width // factor), max(1, enhanced.height // factor)),
            Image.Resampling.BILINEAR,
        )
    else:
        small = enhanced

    pixels = np.array(small, dtype=np.float32).reshape(-1, 3)
    palette_f = palette.astype(np.float32)
    labels_small = np.argmin(((pixels[:, None, :] - palette_f[None, :, :]) ** 2).sum(axis=2), axis=1)
    labels_small = labels_small.astype(np.int32).reshape(small.height, small.width)

    min_region = max(6, (small.width * small.height) // (len(palette) * 55))
    labels_small = simplify_labels(labels_small, min_region=min_region, smooth_size=smooth_size)

    if factor > 1:
        labels_img = Image.fromarray(labels_small.astype(np.uint8), mode="L")
        labels = np.array(labels_img.resize(enhanced.size, Image.Resampling.NEAREST), dtype=np.int32)
        if smooth_size > 1:
            labels = simplify_labels(labels, min_region=min_region * factor, smooth_size=max(3, smooth_size - 2))
        return labels

    return labels_small


def quantize_colors(
    image: Image.Image,
    num_colors: int,
    simplify_factor: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    enhanced = enhance_image(image)

    fine_factor = max(1, simplify_factor - 1)
    coarse_factor = simplify_factor + 1

    quantized = enhanced.resize(
        (max(1, enhanced.width // fine_factor), max(1, enhanced.height // fine_factor)),
        Image.Resampling.BILINEAR,
    ).quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
    raw_palette = np.array(quantized.getpalette(), dtype=np.uint8).reshape(-1, 3)[:num_colors]
    palette = vibrant_palette(raw_palette)

    labels_fine = labels_at_scale(enhanced, palette, fine_factor, smooth_size=3)
    labels_coarse = labels_at_scale(enhanced, palette, coarse_factor, smooth_size=5)

    mask = detail_mask(enhanced)
    keep_detail = mask >= np.percentile(mask, 100 - (38 + simplify_factor * 4))
    labels = np.where(keep_detail, labels_fine, labels_coarse)

    return labels.astype(np.int32), palette


def create_outline(labels: np.ndarray) -> Image.Image:
    edges = np.zeros_like(labels, dtype=bool)
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rgb = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    rgb[edges] = (25, 25, 25)
    return Image.fromarray(rgb, mode="RGB")


def choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_numbers(
    canvas: Image.Image,
    labels: np.ndarray,
    min_label_area: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = choose_font(max(12, canvas.width // 50))

    for color_id in np.unique(labels):
        paint_number = str(int(color_id) + 1)
        mask = labels == color_id
        component_ids, count = ndimage.label(mask)
        if count == 0:
            continue

        sizes = ndimage.sum(mask, component_ids, range(1, count + 1))
        for component, size in enumerate(sizes, start=1):
            if size < min_label_area:
                continue
            ys, xs = np.where(component_ids == component)
            cx, cy = int(xs.mean()), int(ys.mean())
            bbox = draw.textbbox((0, 0), paint_number, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = cx - tw // 2, cy - th // 2
            draw.text((x + 1, y + 1), paint_number, fill=(210, 210, 210), font=font)
            draw.text((x, y), paint_number, fill=(15, 15, 15), font=font)


def create_palette_image(palette: np.ndarray, swatch_size: int = 52) -> Image.Image:
    columns = 4
    rows_count = (len(palette) + columns - 1) // columns
    width = 520
    row_height = swatch_size + 36
    height = row_height * rows_count + 56
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = choose_font(24)
    label_font = choose_font(18)
    draw.text((20, 12), "Палитра цветов", fill=(20, 20, 20), font=title_font)

    col_width = width // columns
    y_base = 52
    for index, rgb in enumerate(palette):
        col = index % columns
        row = index // columns
        x = 20 + col * col_width
        y = y_base + row * row_height
        color = tuple(int(v) for v in rgb)
        draw.rectangle((x, y, x + swatch_size, y + swatch_size), fill=color, outline=(35, 35, 35), width=2)
        draw.text((x, y + swatch_size + 6), f"№ {index + 1}", fill=(20, 20, 20), font=label_font)

    return img


def create_reference_image(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    return Image.fromarray(palette[labels].astype(np.uint8), mode="RGB")


def generate_paint_by_numbers(
    input_path: Path,
    output_dir: Path,
    num_colors: int = 20,
    min_label_area: int = 1400,
    max_dimension: int = 1200,
    simplify_factor: int = 3,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path)
    try:
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = image.convert("RGB")

    scale = min(1.0, max_dimension / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    labels, palette = quantize_colors(image, num_colors, simplify_factor=simplify_factor)

    template = create_outline(labels)
    draw_numbers(template, labels, min_label_area)

    reference = create_reference_image(labels, palette)
    palette_img = create_palette_image(palette)

    stem = input_path.stem
    outputs = {
        "template": output_dir / f"{stem}-template.png",
        "reference": output_dir / f"{stem}-reference.png",
        "palette": output_dir / f"{stem}-palette.png",
    }

    template.save(outputs["template"], dpi=(300, 300))
    reference.save(outputs["reference"], dpi=(300, 300))
    palette_img.save(outputs["palette"], dpi=(300, 300))

    combined = Image.new("RGB", (template.width, template.height + palette_img.height + 20), "white")
    combined.paste(template, (0, 0))
    combined.paste(palette_img, (0, template.height + 20))
    combined_path = output_dir / f"{stem}-printable.png"
    combined.save(combined_path, dpi=(300, 300))
    outputs["printable"] = combined_path

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paint-by-numbers files from an image.")
    parser.add_argument("input", type=Path, help="Source image path")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("paint-by-numbers"))
    parser.add_argument("--colors", type=int, default=20, help="Number of paint colors")
    parser.add_argument("--min-label", type=int, default=1400, help="Minimum area to show a number")
    parser.add_argument("--max-size", type=int, default=1200, help="Max width/height")
    parser.add_argument("--simplify", type=int, default=3, help="Shape simplification (2=detailed, 5=clean)")
    args = parser.parse_args()

    outputs = generate_paint_by_numbers(
        args.input,
        args.output_dir,
        num_colors=args.colors,
        min_label_area=args.min_label,
        max_dimension=args.max_size,
        simplify_factor=args.simplify,
    )

    print("Generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
