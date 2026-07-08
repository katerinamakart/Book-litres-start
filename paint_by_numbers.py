#!/usr/bin/env python3
"""Generate a paint-by-numbers template from an image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import gaussian_filter, generic_filter, zoom

SEGMENT_SIZE = 880
FACE_DETAIL_SIZE = 1200
MIN_REGION_AREA = 140
MERGE_PASSES = 5
FACE_COLORS_MIN = 10
FACE_COLORS_MAX = 18


def _rgb_to_hue(rgb: np.ndarray) -> float:
    r, g, b = rgb / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    d = mx - mn
    if mx == r:
        h = ((g - b) / d + (6 if g < b else 0)) / 6
    elif mx == g:
        h = ((b - r) / d + 2) / 6
    else:
        h = ((r - g) / d + 4) / 6
    return h * 360


def compact_used_colors(labels: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(labels)
    used_sorted = sorted(used, key=lambda idx: (_rgb_to_hue(palette[idx]), idx))
    remap = {int(old): new for new, old in enumerate(used_sorted)}
    remap_fn = np.vectorize(remap.__getitem__)
    compact_labels = remap_fn(labels).astype(np.int32)
    compact_palette = palette[np.array(used_sorted, dtype=np.int32)]
    return compact_labels, compact_palette


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


MASTER_PALETTE = np.array([
    [25, 22, 28],
    [54, 69, 79],
    [90, 55, 35],
    [120, 75, 45],
    [160, 90, 45],
    [100, 20, 40],
    [140, 20, 45],
    [150, 15, 45],
    [200, 90, 30],
    [220, 100, 50],
    [255, 140, 40],
    [250, 128, 114],
    [255, 175, 160],
    [255, 210, 185],
    [255, 200, 50],
    [255, 210, 60],
    [255, 225, 155],
    [255, 230, 120],
    [255, 245, 180],
    [255, 245, 200],
    [20, 70, 40],
    [30, 90, 45],
    [45, 130, 65],
    [75, 175, 85],
    [110, 120, 95],
    [143, 165, 140],
    [175, 195, 170],
    [170, 220, 80],
    [190, 230, 70],
    [160, 230, 180],
    [180, 210, 165],
    [0, 140, 90],
    [0, 150, 160],
    [50, 190, 200],
    [20, 40, 95],
    [30, 60, 180],
    [45, 90, 200],
    [70, 130, 210],
    [135, 200, 245],
    [135, 206, 235],
    [176, 196, 214],
    [190, 50, 90],
    [210, 30, 90],
    [220, 60, 120],
    [255, 50, 120],
    [255, 105, 180],
    [255, 130, 160],
    [255, 170, 190],
    [255, 182, 170],
    [255, 200, 210],
    [230, 210, 240],
    [220, 195, 160],
    [253, 245, 230],
    [255, 252, 248],
], dtype=np.uint8)


def _labels_from_image(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixel_lab = _rgb_to_lab(pixels.reshape(-1, 3))
    palette_lab = _rgb_to_lab(MASTER_PALETTE.astype(np.float32))
    dist = np.sum((pixel_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
    return np.argmin(dist, axis=1).reshape(pixels.shape[:2]).astype(np.int32)


def _majority_filter(values: np.ndarray) -> float:
    vals, counts = np.unique(values.astype(np.int32), return_counts=True)
    return float(vals[np.argmax(counts)])


def _is_skin_tone(r: int, g: int, b: int) -> bool:
    if max(r, g, b) - min(r, g, b) < 12:
        return False
    return r > 60 and g > 30 and b > 15 and r > g and r > b and abs(r - g) > 10


def filter_face_skin_blobs(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    total = h * w
    labeled, count = ndimage.label(mask)
    out = np.zeros_like(mask, dtype=bool)
    for component in range(1, count + 1):
        comp = labeled == component
        size = int(comp.sum())
        if size < total * 0.0012 or size > total * 0.2:
            continue
        ys = np.where(comp)[0]
        if ys.mean() >= h * 0.72:
            continue
        out |= comp
    if out.any():
        out = ndimage.binary_dilation(out, iterations=4)
    return out


def build_face_mask(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for y in range(int(h * 0.78)):
        for x in range(w):
            r, g, b = (int(v) for v in arr[y, x])
            if _is_skin_tone(r, g, b):
                mask[y, x] = True
    return filter_face_skin_blobs(mask)


def mode_filter_labels(labels: np.ndarray, passes: int = 1, size: int = 5, face_mask: np.ndarray | None = None) -> np.ndarray:
    out = labels.copy()
    for _ in range(passes):
        filtered = generic_filter(out.astype(np.float64), _majority_filter, size=size).astype(np.int32)
        if face_mask is not None:
            filtered[face_mask] = out[face_mask]
        out = filtered
    return out


def merge_small_regions(labels: np.ndarray, min_area: int, face_mask: np.ndarray | None = None) -> np.ndarray:
    result = labels.copy()
    h, w = labels.shape
    visited = np.zeros(labels.shape, dtype=bool)
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            color_id = labels[y, x]
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            neighbor_counts: dict[int, int] = {}

            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if labels[ny, nx] == color_id:
                        if not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                    else:
                        neighbor = int(labels[ny, nx])
                        neighbor_counts[neighbor] = neighbor_counts.get(neighbor, 0) + 1

            if len(pixels) < min_area and neighbor_counts:
                if face_mask is not None and any(face_mask[cy, cx] for cy, cx in pixels):
                    continue
                best_neighbor = max(neighbor_counts, key=neighbor_counts.get)
                for cy, cx in pixels:
                    result[cy, cx] = best_neighbor

    return result


def simplify_labels(labels: np.ndarray, face_mask: np.ndarray | None = None) -> np.ndarray:
    simplified = mode_filter_labels(labels, passes=2, size=5, face_mask=face_mask)
    for _ in range(MERGE_PASSES):
        simplified = merge_small_regions(simplified, MIN_REGION_AREA, face_mask)
        simplified = mode_filter_labels(simplified, passes=1, size=3, face_mask=face_mask)
    return simplified


def upscale_local_labels(local: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    src_h, src_w = local.shape
    tgt_w, tgt_h = target_size
    out = np.full((tgt_h, tgt_w), -1, dtype=np.int32)
    scale_x = src_w / tgt_w
    scale_y = src_h / tgt_h
    for y in range(tgt_h):
        for x in range(tgt_w):
            sx = min(src_w - 1, int(x * scale_x))
            sy = min(src_h - 1, int(y * scale_y))
            value = int(local[sy, sx])
            if value >= 0:
                out[y, x] = value
    return out


def build_face_kmeans(image: Image.Image, face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    ys, xs = np.where(face_mask)
    if len(xs) < 80:
        return None

    pixels = arr[ys, xs]
    k = min(FACE_COLORS_MAX, max(FACE_COLORS_MIN, len(pixels) // 400))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pixels), size=k, replace=False)
    centroids = pixels[indices].copy()

    for _ in range(20):
        pixel_lab = _rgb_to_lab(pixels)
        centroid_lab = _rgb_to_lab(centroids)
        dist = np.sum((pixel_lab[:, None, :] - centroid_lab[None, :, :]) ** 2, axis=2)
        assigns = np.argmin(dist, axis=1)
        for c in range(k):
            mask = assigns == c
            if np.any(mask):
                centroids[c] = pixels[mask].mean(axis=0)

    centroids = np.round(centroids).astype(np.uint8)
    pixel_lab = _rgb_to_lab(pixels)
    centroid_lab = _rgb_to_lab(centroids.astype(np.float32))
    dist = np.sum((pixel_lab[:, None, :] - centroid_lab[None, :, :]) ** 2, axis=2)
    local = np.full(face_mask.shape, -1, dtype=np.int32)
    local[ys, xs] = np.argmin(dist, axis=1)
    return local, centroids


def soften_image(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    smoothed = gaussian_filter(arr, sigma=1.2, mode="nearest")
    return Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8))


def resize_to_max(image: Image.Image, max_dimension: int) -> Image.Image:
    scale = min(1.0, max_dimension / max(image.size))
    if scale >= 1.0:
        return image
    return image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )


def upscale_labels(labels: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if labels.shape[1] == target_w and labels.shape[0] == target_h:
        return labels
    factor_y = target_h / labels.shape[0]
    factor_x = target_w / labels.shape[1]
    return zoom(labels, (factor_y, factor_x), order=0).astype(np.int32)


def min_label_area_for_size(width: int, height: int) -> int:
    return max(900, int((width * height) / 11000))


def map_to_master_palette(image: Image.Image, output_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    full_image = image
    if output_size is None:
        output_size = (full_image.width, full_image.height)

    segment_image = resize_to_max(full_image, SEGMENT_SIZE)
    segment_soft = soften_image(segment_image)
    face_mask_seg = build_face_mask(segment_image)

    bg_labels = simplify_labels(_labels_from_image(segment_soft), face_mask_seg)
    bg_labels = upscale_labels(bg_labels, output_size)
    full_face_mask = upscale_labels(face_mask_seg.astype(np.int32), output_size).astype(bool)

    face_image = resize_to_max(full_image, min(max(full_image.size), FACE_DETAIL_SIZE))
    face_mask_detail = upscale_labels(
        face_mask_seg.astype(np.int32),
        (face_image.width, face_image.height),
    ).astype(bool)
    face_result = build_face_kmeans(face_image, face_mask_detail)

    palette = MASTER_PALETTE.copy()
    labels = bg_labels.copy()
    if face_result is not None:
        local, centroids = face_result
        local_full = upscale_local_labels(local, output_size)
        offset = len(palette)
        palette = np.vstack([palette, centroids])
        for y in range(labels.shape[0]):
            for x in range(labels.shape[1]):
                if full_face_mask[y, x] and local_full[y, x] >= 0:
                    labels[y, x] = offset + local_full[y, x]

    return compact_used_colors(labels, palette)


def quantize_colors(image: Image.Image, num_colors: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return map_to_master_palette(image)


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
    face_mask: np.ndarray | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = choose_font(max(12, canvas.width // 50))
    face_min_area = 35

    for color_id in np.unique(labels):
        paint_number = str(int(color_id) + 1)
        mask = labels == color_id
        component_ids, count = ndimage.label(mask)
        if count == 0:
            continue

        sizes = ndimage.sum(mask, component_ids, range(1, count + 1))
        for component, size in enumerate(sizes, start=1):
            ys, xs = np.where(component_ids == component)
            cx, cy = int(xs.mean()), int(ys.mean())
            in_face = face_mask is not None and face_mask[cy, cx]
            area_limit = face_min_area if in_face else min_label_area
            if size < area_limit:
                continue
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
    num_colors: int = 28,
    min_label_area: int = 0,
    max_dimension: int = 1400,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_path)
    try:
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = image.convert("RGB")
    full_image = resize_to_max(image, max_dimension)

    labels, palette = map_to_master_palette(full_image, (full_image.width, full_image.height))
    if min_label_area <= 0:
        min_label_area = min_label_area_for_size(full_image.width, full_image.height)

    segment_for_mask = resize_to_max(full_image, SEGMENT_SIZE)
    face_mask = upscale_labels(
        build_face_mask(segment_for_mask).astype(np.int32),
        (full_image.width, full_image.height),
    ).astype(bool)

    template = create_outline(labels)
    draw_numbers(template, labels, min_label_area, face_mask)

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
    parser.add_argument("--colors", type=int, default=28, help="Number of colors extracted from photo")
    parser.add_argument("--min-label", type=int, default=0, help="Minimum area to show a number (0 = auto)")
    parser.add_argument("--max-size", type=int, default=1400, help="Max width/height")
    args = parser.parse_args()

    outputs = generate_paint_by_numbers(
        args.input,
        args.output_dir,
        num_colors=args.colors,
        min_label_area=args.min_label or 0,
        max_dimension=args.max_size,
    )

    print("Generated:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
