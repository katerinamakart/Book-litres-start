#!/usr/bin/env python3
"""Generate a paint-by-numbers template from an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def prepare_image(path: Path, max_dimension: int) -> np.ndarray:
    image = Image.open(path)
    try:
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
    return np.array(image)


def smooth_for_segmentation(rgb: np.ndarray, strength: int) -> np.ndarray:
    diameter = 5 + strength * 2
    sigma = 25 + strength * 15
    return cv2.bilateralFilter(rgb, diameter, sigma, sigma)


def segment_image(rgb: np.ndarray, num_colors: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Segment with OpenCV k-means on a downscaled, smoothed image."""
    smooth = smooth_for_segmentation(rgb, strength=max(1, 6 - block_size))
    h, w = smooth.shape[:2]
    small_w = max(48, w // block_size)
    small_h = max(48, h // block_size)
    small = cv2.resize(smooth, (small_w, small_h), interpolation=cv2.INTER_AREA)

    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.8)
    _, labels, centers = cv2.kmeans(
        pixels,
        num_colors,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(small_h, small_w).astype(np.int32)
    labels = cv2.resize(labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

    palette = np.clip(centers, 0, 255).astype(np.uint8)
    return labels, palette


def create_outline(labels: np.ndarray) -> Image.Image:
    edges = np.zeros_like(labels, dtype=bool)
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    rgb = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    rgb[edges] = (20, 20, 20)
    return Image.fromarray(rgb, mode="RGB")


def draw_numbers(canvas: Image.Image, labels: np.ndarray, min_area: int) -> None:
    draw = ImageDraw.Draw(canvas)
    font = choose_font(max(11, canvas.width // 55))
    h, w = labels.shape

    for color_id in range(int(labels.max()) + 1):
        mask = (labels == color_id).astype(np.uint8)
        count, components = cv2.connectedComponents(mask)
        for comp in range(1, count):
            area = int(mask[components == comp].sum())
            if area < min_area:
                continue
            ys, xs = np.where(components == comp)
            cx, cy = int(xs.mean()), int(ys.mean())
            text = str(color_id + 1)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw // 2 + 1, cy - th // 2 + 1), text, fill=(210, 210, 210), font=font)
            draw.text((cx - tw // 2, cy - th // 2), text, fill=(15, 15, 15), font=font)


def create_palette_image(palette: np.ndarray) -> Image.Image:
    columns, swatch = 4, 50
    rows = (len(palette) + columns - 1) // columns
    img = Image.new("RGB", (500, rows * (swatch + 34) + 50), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 10), "Палитра цветов", fill=(20, 20, 20), font=choose_font(22))
    label_font = choose_font(16)
    col_w = 500 // columns
    for i, rgb in enumerate(palette):
        col, row = i % columns, i // columns
        x = 16 + col * col_w
        y = 44 + row * (swatch + 34)
        color = tuple(int(v) for v in rgb)
        draw.rectangle((x, y, x + swatch, y + swatch), fill=color, outline=(30, 30, 30), width=2)
        draw.text((x, y + swatch + 6), f"№ {i + 1}", fill=(20, 20, 20), font=label_font)
    return img


def generate_paint_by_numbers(
    input_path: Path,
    output_dir: Path,
    num_colors: int = 20,
    min_label_area: int = 1200,
    max_dimension: int = 1100,
    block_size: int = 3,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = prepare_image(input_path, max_dimension)
    labels, palette = segment_image(rgb, num_colors, block_size)

    template = create_outline(labels)
    draw_numbers(template, labels, min_label_area)
    reference = Image.fromarray(palette[labels].astype(np.uint8), mode="RGB")
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

    printable = Image.new("RGB", (template.width, template.height + palette_img.height + 16), "white")
    printable.paste(template, (0, 0))
    printable.paste(palette_img, (0, template.height + 16))
    outputs["printable"] = output_dir / f"{stem}-printable.png"
    printable.save(outputs["printable"], dpi=(300, 300))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paint-by-numbers files from an image.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("paint-by-numbers"))
    parser.add_argument("--colors", type=int, default=20)
    parser.add_argument("--min-label", type=int, default=1200)
    parser.add_argument("--max-size", type=int, default=1100)
    parser.add_argument("--block-size", type=int, default=3, help="3=portrait, 4=balanced, 5=simple")
    args = parser.parse_args()

    outputs = generate_paint_by_numbers(
        args.input,
        args.output_dir,
        num_colors=args.colors,
        min_label_area=args.min_label,
        max_dimension=args.max_size,
        block_size=args.block_size,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
