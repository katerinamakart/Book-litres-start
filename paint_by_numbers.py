#!/usr/bin/env python3
"""Generate a paint-by-numbers template from an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


STORYBOOK_PALETTE = np.array([
    [255, 252, 246],
    [255, 235, 210],
    [255, 205, 175],
    [255, 165, 185],
    [240, 175, 135],
    [255, 215, 95],
    [255, 235, 70],
    [255, 185, 45],
    [255, 135, 55],
    [255, 115, 135],
    [255, 95, 155],
    [185, 225, 255],
    [95, 165, 255],
    [50, 115, 235],
    [30, 55, 150],
    [25, 45, 110],
    [155, 235, 185],
    [75, 195, 95],
    [35, 135, 65],
    [200, 175, 255],
    [130, 75, 195],
    [155, 95, 60],
    [85, 50, 35],
    [115, 105, 125],
], dtype=np.float32)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    mask = rgb > 0.04045
    rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    x = (rgb[..., 0] * 0.4124564 + rgb[..., 1] * 0.3575761 + rgb[..., 2] * 0.1804375) / 0.95047
    y = rgb[..., 0] * 0.2126729 + rgb[..., 1] * 0.7151522 + rgb[..., 2] * 0.0721750
    z = (rgb[..., 0] * 0.0193339 + rgb[..., 1] * 0.1191920 + rgb[..., 2] * 0.9503041) / 1.08883

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + (16 / 116))

    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([(116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def map_to_storybook_palette(image: Image.Image, num_colors: int) -> tuple[np.ndarray, np.ndarray]:
    palette = STORYBOOK_PALETTE[: max(12, min(num_colors, len(STORYBOOK_PALETTE)))]
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixel_lab = _rgb_to_lab(pixels.reshape(-1, 3))
    palette_lab = _rgb_to_lab(palette)
    dist = np.sum((pixel_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dist, axis=1).reshape(pixels.shape[:2]).astype(np.int32)
    return labels, palette.astype(np.uint8)


def quantize_colors(image: Image.Image, num_colors: int) -> tuple[np.ndarray, np.ndarray]:
    return map_to_storybook_palette(image, num_colors)


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
    num_colors: int = 24,
    min_label_area: int = 600,
    max_dimension: int = 1300,
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

    labels, palette = quantize_colors(image, num_colors)

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
    parser.add_argument("--colors", type=int, default=24, help="Number of fixed storybook colors (12-24)")
    parser.add_argument("--min-label", type=int, default=600, help="Minimum area to show a number")
    parser.add_argument("--max-size", type=int, default=1300, help="Max width/height")
    args = parser.parse_args()

    outputs = generate_paint_by_numbers(
        args.input,
        args.output_dir,
        num_colors=args.colors,
        min_label_area=args.min_label,
        max_dimension=args.max_size,
    )

    print("Generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
