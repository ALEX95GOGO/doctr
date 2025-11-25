#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def load_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """
    Try to load a reasonable TTF font; fall back to the default if not available.
    """
    # Common fonts to try
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue

    # Fallback: PIL default bitmap font (no size control)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Get width/height of text using Pillow's textbbox (works in modern versions).
    """
    if not text:
        return 0, 0

    # textbbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def is_normalized(x: float, y: float) -> bool:
    """
    Heuristic: coordinates are normalized if they are in [0, ~1.5].
    """
    return 0 <= x <= 1.5 and 0 <= y <= 1.5


# ---------------------------------------------------------
# Core overlay logic
# ---------------------------------------------------------


def overlay_boxes_on_image(
    image_path: Path,
    rows: List[Dict[str, str]],
    out_path: Path,
    draw_line_boxes: bool = True,
):
    """
    Draw word-level bounding boxes and yellow text on the image.

    - image_path: path to the input PNG.
    - rows: list of CSV rows (all for this image).
    - out_path: where to save the overlaid image.
    """
    if not image_path.exists():
        print(f"  [WARN] Image not found, skipping: {image_path}")
        return

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    font = load_font(size=max(14, int(H * 0.025)))  # scale font slightly with height

    # Optional: group rows by (page_idx, line_idx) to also draw line-level boxes
    lines: Dict[tuple, List[Dict[str, str]]] = {}
    for row in rows:
        try:
            page_idx = int(row.get("page_idx", 0))
        except ValueError:
            page_idx = 0
        try:
            line_idx = int(row.get("line_idx", 0))
        except ValueError:
            line_idx = 0

        key = (page_idx, line_idx)
        lines.setdefault(key, []).append(row)

    # First: draw line-level boxes (background) if requested
    if draw_line_boxes:
        for (page_idx, line_idx), line_rows in lines.items():
            # Use the first row's line coords as the line box
            r0 = line_rows[0]
            try:
                lx0 = float(r0.get("line_x0", 0.0))
                ly0 = float(r0.get("line_y0", 0.0))
                lx1 = float(r0.get("line_x1", 0.0))
                ly1 = float(r0.get("line_y1", 0.0))
            except ValueError:
                continue

            # Convert to pixel coordinates (likely normalized)
            if is_normalized(lx0, ly0) and is_normalized(lx1, ly1):
                x0 = int(lx0 * W)
                y0 = int(ly0 * H)
                x1 = int(lx1 * W)
                y1 = int(ly1 * H)
            else:
                x0 = int(lx0)
                y0 = int(ly0)
                x1 = int(lx1)
                y1 = int(ly1)

            # Semi-transparent background for the whole line area
            line_bg = (0, 0, 0, 80)  # translucent black
            draw.rectangle([x0, y0, x1, y1], fill=line_bg)

    # Second: draw word-level boxes and yellow text
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            wx0 = float(row.get("word_x0", 0.0))
            wy0 = float(row.get("word_y0", 0.0))
            wx1 = float(row.get("word_x1", 0.0))
            wy1 = float(row.get("word_y1", 0.0))
        except ValueError:
            continue

        # Convert to pixel coordinates
        if is_normalized(wx0, wy0) and is_normalized(wx1, wy1):
            x0 = int(wx0 * W)
            y0 = int(wy0 * H)
            x1 = int(wx1 * W)
            y1 = int(wy1 * H)
        else:
            x0 = int(wx0)
            y0 = int(wy0)
            x1 = int(wx1)
            y1 = int(wy1)

        # Draw bounding box (e.g., red)
        box_color = (255, 0, 0, 255)  # solid red
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)

        # If there's no text, nothing more to draw
        if not text:
            continue

        # Compute text size
        tw, th = text_size(draw, text, font)
        if tw == 0 or th == 0:
            continue

        # Position text: above the box if there's space, else inside
        tx = x0
        ty = y0 - th - 2
        if ty < 0:
            ty = y0 + 2

        # Background for text: semi-transparent black
        label_bg = (0, 0, 0, 160)
        draw.rectangle(
            [tx - 2, ty - 2, tx + tw + 2, ty + th + 2],
            fill=label_bg,
        )

        # Optional black outline for readability
        outline_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for ox, oy in outline_offsets:
            draw.text((tx + ox, ty + oy), text, font=font, fill=(0, 0, 0, 255))

        # Main yellow text
        draw.text((tx, ty), text, font=font, fill=(255, 255, 0, 255))  # yellow

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"  Saved overlay: {out_path}")


# ---------------------------------------------------------
# CSV loading and per-image grouping
# ---------------------------------------------------------


def load_rows_grouped_by_image(csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Read CSV and group rows by source image name (stem of source_file, .png).

    Returns: dict mapping 'frame_XXXXXX.png' -> [rows...]
    """
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "source_file" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'source_file' column.")

        grouped: Dict[str, List[Dict[str, str]]] = {}
        count = 0
        for row in reader:
            src = (row.get("source_file") or "").strip()
            if not src:
                continue
            stem = Path(src).stem
            img_name = stem + ".png"
            grouped.setdefault(img_name, []).append(row)
            count += 1

    print(f"Loaded {count} rows from {csv_path}")
    print(f"Found {len(grouped)} unique images in CSV.")
    return grouped


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay OCR bounding boxes and detected text onto PNG frames.\n"
            "Expects a CSV with columns like source_file, word_x0, word_y0, "
            "word_x1, word_y1, etc., and PNG images named after source_file "
            "stems (e.g., frame_000001.json -> frame_000001.png)."
        )
    )
    parser.add_argument("csv", type=str, help="Input CSV with OCR data.")
    parser.add_argument(
        "images_dir",
        type=str,
        help="Directory containing PNG frames (frame_*.png).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="overlays",
        help="Directory where overlaid PNGs will be saved.",
    )
    parser.add_argument(
        "--no-line-box",
        action="store_true",
        help="Disable drawing line-level background boxes (only word boxes).",
    )

    args = parser.parse_args()
    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")

    grouped = load_rows_grouped_by_image(csv_path)

    for img_name, rows in grouped.items():
        image_path = images_dir / img_name
        out_path = out_dir / img_name
        print(f"Processing {img_name} with {len(rows)} rows...")
        overlay_boxes_on_image(
            image_path,
            rows,
            out_path,
            draw_line_boxes=not args.no_line_box,
        )


if __name__ == "__main__":
    main()

