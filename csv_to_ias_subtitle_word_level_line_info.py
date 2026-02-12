#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import difflib
from collections import defaultdict


def _normalize_for_similarity(s: str) -> str:
    """Lowercase, strip punctuation and extra spaces for comparison."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s).strip() # collapse whitespace
    return s


def are_similar_sentences(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Return True if sentences are 'similar enough' to be treated as the same.

    - Normalizes text (lowercase, no punctuation, collapsed spaces).
    - If one normalized string contains the other, consider them the same.
    - Otherwise, use a SequenceMatcher ratio.
    """
    a_norm = _normalize_for_similarity(a)
    b_norm = _normalize_for_similarity(b)

    if not a_norm or not b_norm:
        return False

    if a_norm in b_norm or b_norm in a_norm:
        return True

    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= threshold


# ----------------- helpers ----------------- #

def is_normalized(x: float, y: float) -> bool:
    """Heuristic: coordinates are normalized if they lie roughly in [0, 1.5]."""
    return 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5


def parse_frame_index(source_file: str) -> int:
    """Extract frame index from a source_file like 'frame_006150.json'. Returns 0 if cannot parse."""
    stem = Path(source_file).stem
    m = re.search(r"(\d+)$", stem)
    if not m:
        return 0
    return int(m.group(1))


def compute_pixel_bbox(row: dict) -> Tuple[int, int, int, int]:
    """
    Compute pixel bounding box (x0, y0, x1, y1) from a CSV row.

    Uses word_x0/1, word_y0/1 and page_width/page_height.
    Handles normalized coords (0-1 range) or absolute pixels.
    """
    wx0 = float(row.get("word_x0", 0.0) or 0.0)
    wy0 = float(row.get("word_y0", 0.0) or 0.0)
    wx1 = float(row.get("word_x1", 0.0) or 0.0)
    wy1 = float(row.get("word_y1", 0.0) or 0.0)

    pw = float(row.get("page_height", 0.0) or 0.0)
    ph = float(row.get("page_width", 0.0) or 0.0)

    if pw <= 0 or ph <= 0:
        x0 = int(round(wx0))
        y0 = int(round(wy0))
        x1 = int(round(wx1))
        y1 = int(round(wy1))
        return x0, y0, x1, y1

    if is_normalized(wx0, wy0) and is_normalized(wx1, wy1):
        x0 = int(round(wx0 * pw))
        y0 = int(round(wy0 * ph))
        x1 = int(round(wx1 * pw))
        y1 = int(round(wy1 * ph))
    else:
        x0 = int(round(wx0))
        y0 = int(round(wy0))
        x1 = int(round(wx1))
        y1 = int(round(wy1))

    # manual offsets (kept from your original code)
    x0 += 600
    x1 += 600
    y1 += 1250
    y0 += 1050
    return x0, y0, x1, y1


# ----------------- MASTER (merge into IAS labels) ----------------- #

MASTER_COLS = ["Category", "LineNumber", "Subtitles", "SubNum"]


def sniff_delimiter(path: Path) -> str:
    """Auto-detect between TSV and CSV."""
    sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
    joined = "\n".join(sample)
    if joined.count("\t") >= joined.count(","):
        return "\t"
    return ","


def load_master_rows(master_path: Path) -> List[Dict[str, str]]:
    """Load master rows in order. Keeps duplicates. Accepts TSV or CSV."""
    delim = sniff_delimiter(master_path)
    with master_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError("Master file has no header.")
        missing = [c for c in MASTER_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Master file missing columns: {missing}. Found: {reader.fieldnames}")
        rows = list(reader)

    for r in rows:
        for c in MASTER_COLS:
            r[c] = (r.get(c) or "").strip()
    return rows


def match_group_to_master_future_only(
    sentence_group_id: str,
    master_rows: List[Dict[str, str]],
    master_ptr: int,
    lookahead: int,
    threshold: float = 0.85,
) -> Tuple[Dict[str, str], int]:
    """
    STRICT forward-only match of a sentence group to master.

    - sentence_group_id includes "__groupN" suffix
    - match base subtitle = before "__group"
    - search only in [master_ptr, master_ptr+lookahead)
    - pick first exact normalized match; else best similarity >= threshold
    - on success: advance pointer to matched index + 1
    - on failure: return blanks, pointer unchanged
    """
    base = sentence_group_id.split("__group", 1)[0].strip()
    base_norm = _normalize_for_similarity(base)

    if not base_norm:
        return {c: "" for c in MASTER_COLS}, master_ptr

    start = max(master_ptr, 0)
    end = min(len(master_rows), start + lookahead)
    if start >= end:
        return {c: "" for c in MASTER_COLS}, master_ptr

    # exact normalized match first
    for i in range(start, end):
        cand = master_rows[i].get("Subtitles", "")
        if _normalize_for_similarity(cand) == base_norm:
            meta = {c: master_rows[i].get(c, "") for c in MASTER_COLS}
            return meta, i + 1

    # similarity match
    best_i: Optional[int] = None
    best_score = -1.0
    for i in range(start, end):
        cand = master_rows[i].get("Subtitles", "")
        score = difflib.SequenceMatcher(None, base_norm, _normalize_for_similarity(cand)).ratio()
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score >= threshold:
        meta = {c: master_rows[best_i].get(c, "") for c in MASTER_COLS}
        return meta, best_i + 1

    return {c: "" for c in MASTER_COLS}, master_ptr


# ----------------- group OCR rows by sentence (consecutive) ----------------- #

def load_csv_group_by_sentence(csv_path: Path) -> Dict[str, List[dict]]:
    """Group consecutive rows into sentence groups like '<base>__groupN'."""
    groups: Dict[str, List[dict]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "source_file",
            "sentence",
            "text",
            "word_x0",
            "word_y0",
            "word_x1",
            "word_y1",
            "page_width",
            "page_height",
            "line_idx",
        }
        if not reader.fieldnames or not required_cols.issubset(reader.fieldnames):
            raise ValueError("CSV must contain columns: " + ", ".join(sorted(required_cols)))

        row_count = 0
        prev_sentence_raw: Optional[str] = None
        prev_base_key: Optional[str] = None
        group_index: Dict[str, int] = {}

        for row in reader:
            row_count += 1
            sentence_raw = (row.get("sentence") or "").strip()
            text = (row.get("text") or "").strip()
            if not sentence_raw or not text:
                continue

            if prev_sentence_raw is not None and are_similar_sentences(sentence_raw, prev_sentence_raw):
                base_key = prev_base_key or sentence_raw
            else:
                base_key = sentence_raw
                group_index[base_key] = group_index.get(base_key, 0) + 1

            group_name = f"{base_key}__group{group_index[base_key]}"
            groups.setdefault(group_name, []).append(row)

            prev_sentence_raw = sentence_raw
            prev_base_key = base_key

    print(f"Loaded {row_count} CSV rows.")
    print(f"Grouped into {len(groups)} sentence groups.")
    return groups


# ---------- sentence bbox helper (earliest frame) ---------- #

def compute_sentence_bbox_from_rows(rows: List[dict]) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute sentence bbox from the earliest frame only:
      left=min(x0), top=min(y0), right=max(x1), bottom=max(y1)
    Returns None if no valid words.
    """
    frame_map: Dict[int, List[dict]] = {}
    for r in rows:
        src = (r.get("source_file") or "").strip()
        frame_idx = parse_frame_index(src)
        frame_map.setdefault(frame_idx, []).append(r)

    if not frame_map:
        return None

    min_frame = min(frame_map.keys())
    first_frame_rows = frame_map[min_frame]

    xs0: List[int] = []
    ys0: List[int] = []
    xs1: List[int] = []
    ys1: List[int] = []

    for r in first_frame_rows:
        t = (r.get("text") or "").strip()
        if not t:
            continue
        x0, y0, x1, y1 = compute_pixel_bbox(r)
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)

    if not xs0:
        return None

    return min(xs0), min(ys0), max(xs1), max(ys1)


# ---------- WORD-LEVEL RECORDS ---------- #
# (start_val, end_val, rect_id, x0, y0, x1, y1, text, rid, line_idx, sentence_id, line_no, line_count, is_multiline)
WordRecord = Tuple[int, int, int, int, int, int, int, str, int, int, str, int, int, int]


def sentence_group_to_word_records(
    sentence: str,
    rows: List[dict],
    fps: float,
    negative_times: bool,
    next_id_start: int,
) -> Tuple[List[WordRecord], int]:
    frame_map: Dict[int, List[dict]] = {}
    for r in rows:
        src = (r.get("source_file") or "").strip()
        frame_idx = parse_frame_index(src)
        frame_map.setdefault(frame_idx, []).append(r)

    if not frame_map:
        return [], next_id_start

    frame_indices = sorted(frame_map.keys())
    min_frame = frame_indices[0]
    max_frame = frame_indices[-1]

    frame_duration_ms = 1000.0 / fps
    start_ms = round(min_frame * frame_duration_ms)
    end_ms = round((max_frame + 1) * frame_duration_ms)
    start_val = -start_ms if negative_times else start_ms
    end_val = -end_ms if negative_times else end_ms

    first_frame_rows = frame_map[min_frame]

    # line structure from earliest frame
    line_centers: Dict[int, List[float]] = defaultdict(list)
    for r in first_frame_rows:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        line_idx = int(r.get("line_idx", 0) or 0)
        x0, y0, x1, y1 = compute_pixel_bbox(r)
        line_centers[line_idx].append((y0 + y1) / 2.0)

    line_order = sorted(line_centers.keys(), key=lambda li: sum(line_centers[li]) / len(line_centers[li]))
    line_rank = {li: i + 1 for i, li in enumerate(line_order)}
    line_count = len(line_order)
    is_multiline = 1 if line_count > 1 else 0

    rect_id = next_id_start
    records: List[WordRecord] = []

    first_frame_rows_sorted = sorted(first_frame_rows, key=lambda r: float(r.get("word_x0", 0.0) or 0.0))

    for r in first_frame_rows_sorted:
        text = (r.get("text") or "").strip()
        if not text:
            continue

        x0, y0, x1, y1 = compute_pixel_bbox(r)
        line_idx = int(r.get("line_idx", 0) or 0)
        line_no = line_rank.get(line_idx, 1)

        records.append(
            (start_val, end_val, rect_id, x0, y0, x1, y1, text, rect_id, line_idx, sentence, line_no, line_count, is_multiline)
        )
        rect_id += 1

    return records, rect_id


# ---------- SENTENCE-LEVEL RECORDS ---------- #

def sentence_group_to_sentence_record(
    sentence: str,
    rows: List[dict],
    fps: float,
    negative_times: bool,
    next_id_start: int,
) -> Tuple[List[Tuple[int, int, int, int, int, int, int, str]], int]:
    frame_map: Dict[int, List[dict]] = {}
    for r in rows:
        src = (r.get("source_file") or "").strip()
        frame_idx = parse_frame_index(src)
        frame_map.setdefault(frame_idx, []).append(r)

    if not frame_map:
        return [], next_id_start

    frame_indices = sorted(frame_map.keys())
    min_frame = frame_indices[0]
    max_frame = frame_indices[-1]

    frame_duration_ms = 1000.0 / fps
    start_ms = round(min_frame * frame_duration_ms)
    end_ms = round((max_frame + 1) * frame_duration_ms)
    start_val = -start_ms if negative_times else start_ms
    end_val = -end_ms if negative_times else end_ms

    bbox = compute_sentence_bbox_from_rows(rows)
    if bbox is None:
        return [], next_id_start

    x0, y0, x1, y1 = bbox
    rect_id = next_id_start
    record = (start_val, end_val, rect_id, x0, y0, x1, y1, sentence)
    return [record], rect_id + 1


# ---------- time padding (±300ms, no overlap) ---------- #

def extend_spans_with_padding(records: List[tuple], negative_times: bool, padding: int = 300) -> List[tuple]:
    if not records:
        return records

    unique_spans = set()
    for rec in records:
        start_val, end_val = rec[0], rec[1]
        s_ms = -start_val if negative_times else start_val
        e_ms = -end_val if negative_times else end_val
        unique_spans.add((s_ms, e_ms))

    spans_sorted = sorted(unique_spans, key=lambda t: t[0])
    span_mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}
    prev_end: Optional[int] = None

    for (s_ms, e_ms) in spans_sorted:
        new_s = s_ms - padding
        new_e = e_ms + padding
        if prev_end is not None and new_s < prev_end:
            new_s = prev_end
        if new_s > new_e:
            new_s, new_e = s_ms, e_ms
        span_mapping[(s_ms, e_ms)] = (new_s, new_e)
        prev_end = new_e

    adjusted = []
    for rec in records:
        start_val, end_val = rec[0], rec[1]
        rest = rec[2:]

        s_ms = -start_val if negative_times else start_val
        e_ms = -end_val if negative_times else end_val
        new_s, new_e = span_mapping[(s_ms, e_ms)]

        new_start_val = -new_s if negative_times else new_s
        new_end_val = -new_e if negative_times else new_e
        adjusted.append((new_start_val, new_end_val, *rest))

    return adjusted


# ---------- avoid y overlap inside same sentence (word-level only) ---------- #

def adjust_y_per_sentence(records: List[WordRecord]) -> List[WordRecord]:
    if not records:
        return records

    span_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        span_to_indices[(rec[0], rec[1])].append(idx)

    new_records = list(records)

    for _, idxs in span_to_indices.items():
        if len(idxs) <= 1:
            continue

        line_groups: Dict[int, List[int]] = defaultdict(list)
        for idx in idxs:
            line_idx = int(new_records[idx][9])
            line_groups[line_idx].append(idx)

        if len(line_groups) <= 1:
            continue

        line_infos = []
        for line_idx, indices in line_groups.items():
            y0s = [new_records[i][4] for i in indices]
            y1s = [new_records[i][6] for i in indices]
            centers = [(new_records[i][4] + new_records[i][6]) / 2.0 for i in indices]
            line_infos.append((line_idx, sum(centers) / len(centers), min(y0s), max(y1s), indices))

        line_infos.sort(key=lambda x: x[1])
        mids: List[int] = []
        for i in range(len(line_infos) - 1):
            mids.append(int(round((line_infos[i][1] + line_infos[i + 1][1]) / 2.0)))

        k = len(line_infos)
        for i, (line_idx, _, min_y0, max_y1, indices) in enumerate(line_infos):
            if i == 0:
                seg_y0, seg_y1 = min_y0, mids[0]
            elif i == k - 1:
                seg_y0, seg_y1 = mids[-1], max_y1
            else:
                seg_y0, seg_y1 = mids[i - 1], mids[i]

            if seg_y1 <= seg_y0:
                seg_y0, seg_y1 = min_y0, max_y1

            for idx in indices:
                (start_val, end_val, rect_id, x0, _, x1, _, text, rid, line_idx_val,
                 sentence_id, line_no, line_count, is_multiline) = new_records[idx]

                new_records[idx] = (
                    start_val, end_val, rect_id, x0, seg_y0, x1, seg_y1,
                    text, rid, line_idx_val, sentence_id, line_no, line_count, is_multiline
                )

    return new_records


# ---------- write IAS ---------- #

def write_ias_word_level(
    groups: Dict[str, List[dict]],
    out_path: Path,
    fps: float,
    negative_times: bool,
    master_rows: List[Dict[str, str]],
    lookahead: int,
    match_threshold: float,
):
    # match master per group in order (future only)
    group_meta: Dict[str, Dict[str, str]] = {}
    master_ptr = 0
    for sentence_id in groups.keys():
        meta, master_ptr = match_group_to_master_future_only(
            sentence_id, master_rows, master_ptr=master_ptr, lookahead=lookahead, threshold=match_threshold
        )
        group_meta[sentence_id] = meta

    # build word records, but if master LineNumber == "Single", force y0/y1 = sentence bbox y0/y1
    all_records: List[WordRecord] = []
    next_id = 1

    for sentence_id, rows in groups.items():
        recs, next_id = sentence_group_to_word_records(
            sentence_id, rows, fps=fps, negative_times=negative_times, next_id_start=next_id
        )

        meta = group_meta.get(sentence_id, {c: "" for c in MASTER_COLS})
        if (meta.get("LineNumber") or "").strip().lower() == "single":
            bbox = compute_sentence_bbox_from_rows(rows)
            if bbox is not None:
                _sx0, sy0, _sx1, sy1 = bbox
                fixed = []
                for rec in recs:
                    (start_val, end_val, rect_id, x0, _y0, x1, _y1,
                     text, rid, line_idx, sid, line_no, line_count, is_multiline) = rec
                    fixed.append(
                        (start_val, end_val, rect_id, x0, sy0, x1, sy1,
                         text, rid, line_idx, sid, line_no, line_count, is_multiline)
                    )
                recs = fixed

        all_records.extend(recs)

    all_records = extend_spans_with_padding(all_records, negative_times)
    # Multi-line still gets per-line segmentation; Single-line won't be changed (only one line group)
    all_records = adjust_y_per_sentence(all_records)
    all_records.sort(key=lambda r: (r[0], r[2]), reverse=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            (start_val, end_val, rect_id, x0, y0, x1, y1,
             text, rid, line_idx, sentence_id, line_no, line_count, is_multiline) = rec

            meta = group_meta.get(sentence_id, {c: "" for c in MASTER_COLS})
            cat = meta["Category"]
            ln = meta["LineNumber"]
            sub = meta["Subtitles"]
            subnum = meta["SubNum"]

            # No spaces in label
            label = (
                f"[Category={cat}|LineNumber={ln}|SubNum={subnum}|"
                f"Subtitles={sub}|S={sentence_id}|L={line_no}/{line_count}|ML={is_multiline}|{text}#{rect_id}]"
            )

            f.write(f"{start_val} {end_val} RECTANGLE {rect_id} {x0} {y0} {x1} {y1} {label}\n")

    print(f"Wrote WORD-LEVEL IAS file with {len(all_records)} rectangles to: {out_path}")


def write_ias_sentence_level(
    groups: Dict[str, List[dict]],
    out_path: Path,
    fps: float,
    negative_times: bool,
    master_rows: List[Dict[str, str]],
    lookahead: int,
    match_threshold: float,
):
    group_meta: Dict[str, Dict[str, str]] = {}
    master_ptr = 0
    for sentence_id in groups.keys():
        meta, master_ptr = match_group_to_master_future_only(
            sentence_id, master_rows, master_ptr=master_ptr, lookahead=lookahead, threshold=match_threshold
        )
        group_meta[sentence_id] = meta

    all_records: List[Tuple[int, int, int, int, int, int, int, str]] = []
    next_id = 1
    for sentence_id, rows in groups.items():
        recs, next_id = sentence_group_to_sentence_record(
            sentence_id, rows, fps=fps, negative_times=negative_times, next_id_start=next_id
        )
        all_records.extend(recs)

    all_records = extend_spans_with_padding(all_records, negative_times)
    all_records.sort(key=lambda r: (r[0], r[2]), reverse=True)

    with out_path.open("w", encoding="utf-8") as f:
        for start_val, end_val, rect_id, x0, y0, x1, y1, sentence_id in all_records:
            meta = group_meta.get(sentence_id, {c: "" for c in MASTER_COLS})
            cat = meta["Category"]
            ln = meta["LineNumber"]
            sub = meta["Subtitles"]
            subnum = meta["SubNum"]

            label = f"[Category={cat}|LineNumber={ln}|SubNum={subnum}|Subtitles={sub}|{sentence_id}#{rect_id}]"
            f.write(f"{start_val} {end_val} RECTANGLE {rect_id} {x0} {y0} {x1} {y1} {label}\n")

    print(f"Wrote SENTENCE-LEVEL IAS file with {len(all_records)} rectangles to: {out_path}")


# ----------------- CLI ----------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Convert OCR CSV -> IAS (word + sentence level).\n"
            "Also merges master subtitles metadata into IAS labels (Category, LineNumber, Subtitles, SubNum).\n"
            "Matching is future-only (monotonic) within a lookahead window.\n"
            "NEW: if master LineNumber==Single, force word-level y0/y1 to sentence bbox y0/y1."
        ),
    )
    parser.add_argument("input_csv", type=str)
    parser.add_argument("master_file", type=str, help="Master TSV/CSV with columns: Category, LineNumber, Subtitles, SubNum")
    parser.add_argument("-o", "--output", type=str, default="output.ias")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--positive-times", action="store_true")
    parser.add_argument("--master-lookahead", type=int, default=50)
    parser.add_argument("--master-threshold", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    master_path = Path(args.master_file)
    out_base = Path(args.output)

    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")
    if not master_path.exists():
        raise SystemExit(f"Master file not found: {master_path}")

    base_stem = out_base.stem
    parent = out_base.parent if str(out_base.parent) not in ("", ".") else Path(".")
    word_out_path = parent / f"{base_stem}_word.ias"
    sentence_out_path = parent / f"{base_stem}_sentence.ias"

    groups = load_csv_group_by_sentence(csv_path)
    master_rows = load_master_rows(master_path)
    negative_times = not args.positive_times

    write_ias_word_level(
        groups, word_out_path,
        fps=args.fps, negative_times=negative_times,
        master_rows=master_rows,
        lookahead=args.master_lookahead,
        match_threshold=args.master_threshold,
    )
    write_ias_sentence_level(
        groups, sentence_out_path,
        fps=args.fps, negative_times=negative_times,
        master_rows=master_rows,
        lookahead=args.master_lookahead,
        match_threshold=args.master_threshold,
    )


if __name__ == "__main__":
    main()
