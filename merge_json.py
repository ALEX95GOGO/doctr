#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def iter_words_from_doc(doc: dict, source_name: str):
    """
    Yield rows for each word in a DocTR JSON document.
    One row per word in pages -> blocks -> lines -> words.
    'sentence' = all words in the block (i.e. merged across lines).
    """
    pages = doc.get("pages", [])
    for page_idx, page in enumerate(pages):
        page_w, page_h = (page.get("dimensions") or [None, None])

        blocks = page.get("blocks", []) or []
        for block_idx, block in enumerate(blocks):
            block_geom = block.get("geometry") or [[None, None], [None, None]]
            (bx0, by0), (bx1, by1) = block_geom

            lines = block.get("lines", []) or []

            # --- NEW: build a block-level sentence (merge all lines) ---
            block_words = []
            for line in lines:
                words_in_line = line.get("words", []) or []
                for w in words_in_line:
                    text = (w.get("value") or "").strip()
                    if text:
                        block_words.append(text)
            block_sentence = " ".join(block_words)
            # -----------------------------------------------------------

            for line_idx, line in enumerate(lines):
                line_geom = line.get("geometry") or [[None, None], [None, None]]
                (lx0, ly0), (lx1, ly1) = line_geom

                words = line.get("words", []) or []

                for word_idx, word in enumerate(words):
                    value = word.get("value", "")
                    confidence = word.get("confidence", None)
                    w_geom = word.get("geometry") or [[None, None], [None, None]]
                    (wx0, wy0), (wx1, wy1) = w_geom
                    word_obj_score = word.get("objectness_score", None)

                    crop_orient = word.get("crop_orientation") or {}
                    crop_angle = crop_orient.get("value", None)
                    crop_angle_conf = crop_orient.get("confidence", None)

                    yield {
                        "source_file": source_name,
                        "page_idx": page_idx,
                        "page_width": page_w,
                        "page_height": page_h,
                        "block_idx": block_idx,
                        "block_x0": bx0,
                        "block_y0": by0,
                        "block_x1": bx1,
                        "block_y1": by1,
                        "line_idx": line_idx,
                        "line_x0": lx0,
                        "line_y0": ly0,
                        "line_x1": lx1,
                        "line_y1": ly1,
                        "word_idx": word_idx,
                        "text": value,
                        "confidence": confidence,
                        "word_x0": wx0,
                        "word_y0": wy0,
                        "word_x1": wx1,
                        "word_y1": wy1,
                        "word_objectness": word_obj_score,
                        "crop_angle": crop_angle,
                        "crop_angle_confidence": crop_angle_conf,
                        # Now: full sentence for the entire block (all lines merged)
                        "sentence": block_sentence,
                    }


def collect_json_files(input_path: Path):
    """
    Return a list of JSON files.
    - If input_path is a directory, scan for *.json.
    - If it's a file, just return that one (if .json).
    """
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() == ".json")
    else:
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file is not .json: {input_path}")
        return [input_path]


def merge_json_to_csv(input_path: Path, output_csv: Path):
    json_files = collect_json_files(input_path)

    fieldnames = [
        "source_file",
        "page_idx",
        "page_width",
        "page_height",
        "block_idx",
        "block_x0",
        "block_y0",
        "block_x1",
        "block_y1",
        "line_idx",
        "line_x0",
        "line_y0",
        "line_x1",
        "line_y1",
        "word_idx",
        "text",
        "confidence",
        "word_x0",
        "word_y0",
        "word_x1",
        "word_y1",
        "word_objectness",
        "crop_angle",
        "crop_angle_confidence",
        "sentence",      # block-level sentence (all lines merged)
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for json_path in json_files:
            with json_path.open("r", encoding="utf-8") as f_in:
                try:
                    doc = json.load(f_in)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON {json_path}: {e}")
                    continue

            for row in iter_words_from_doc(doc, source_name=json_path.name):
                print(row)
                writer.writerow(row)

    print(f"Wrote CSV: {output_csv} (from {len(json_files)} JSON files)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge DocTR JSON word outputs into a single CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a directory containing JSON files, or a single JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="merged_words.csv",
        help="Output CSV file path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input)
    output_csv = Path(args.output)
    merge_json_to_csv(input_path, output_csv)

