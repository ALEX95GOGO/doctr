#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

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

    # if one is basically a substring of the other (like your example)
    if a_norm in b_norm or b_norm in a_norm:
        return True

    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= threshold


# ----------------- helpers ----------------- #

def is_normalized(x: float, y: float) -> bool:
    """Heuristic: coordinates are normalized if they lie roughly in [0, 1.5]."""
    return 0.0 <= x <= 1.5 and 0.0 <= y <= 1.5


def parse_frame_index(source_file: str) -> int:
    """
    Extract frame index from a source_file like 'frame_006150.json'.

    Returns 0 if it cannot parse.
    """
    stem = Path(source_file).stem  # e.g. 'frame_006150'
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
        # Fallback: treat word coords as already in pixels
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


def load_csv_group_by_sentence(
    csv_path: Path,
) -> Dict[str, List[dict]]:
    """
    Group rows only when sentence text (or a similar variant) is the same
    **consecutively**.

    Example:
        sentence A
        sentence A
        sentence B
        sentence A   ← this becomes a NEW group for A

    Now also treats similar variants as the same sentence, e.g.:
        "thwart your finding us. I've Apparated..."
        "I've Apparated to more vaguely defined destinations than this. (chuckles)"
    """

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
            raise ValueError(
                "CSV must contain columns: "
                + ", ".join(sorted(required_cols))
            )

        row_count = 0

        prev_sentence_raw: str | None = None  # actual sentence of previous row
        prev_base_key: str | None = None      # sentence text used as base for the current group
        group_index: Dict[str, int] = {}      # track how many groups per base sentence

        for row in reader:
            row_count += 1
            sentence_raw = (row.get("sentence") or "").strip()
            text = (row.get("text") or "").strip()
            if not sentence_raw or not text:
                continue

            # Decide if this row continues the previous group or starts a new one.
            if (
                prev_sentence_raw is not None
                and are_similar_sentences(sentence_raw, prev_sentence_raw)
            ):
                # same group as previous (similar sentence)
                base_key = prev_base_key or sentence_raw
            else:
                # start a new group for this sentence (or similar cluster)
                base_key = sentence_raw
                group_index[base_key] = group_index.get(base_key, 0) + 1

            group_name = f"{base_key}__group{group_index[base_key]}"
            groups.setdefault(group_name, []).append(row)

            prev_sentence_raw = sentence_raw
            prev_base_key = base_key

    print(f"Loaded {row_count} CSV rows.")
    print(f"Grouped into {len(groups)} sentence groups.")
    return groups


# ---------- WORD-LEVEL RECORDS (原来的逻辑，名字修正) ---------- #

def sentence_group_to_word_records(
    sentence: str,
    rows: List[dict],
    fps: float,
    negative_times: bool,
    next_id_start: int,
) -> Tuple[List[Tuple[int, int, int, int, int, int, str, int, int]], int]:
    """
    For one sentence group (word-level AOI):
      - find earliest and latest frame index across all rows
      - use only rows from earliest frame for rectangles
      - assign the same (start,end) time for all words in that sentence
      - assign ids starting from next_id_start

    Returns (records, next_id_after), where each record is:
      (start_val, end_val, rect_id, x0, y0, x1, y1, text, rect_id, line_idx)
    """
    # Map frame_idx -> [rows in that frame]
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
    if negative_times:
        start_val = -start_ms
        end_val = -end_ms
    else:
        start_val = start_ms
        end_val = end_ms

    # Use rows from the first frame only (the "first sentence" visually)
    first_frame_rows = frame_map[min_frame]

    records: List[Tuple[int, int, int, int, int, int, str, int, int]] = []
    rect_id = next_id_start

    # We can be deterministic by sorting by word_x0
    first_frame_rows_sorted = sorted(
        first_frame_rows,
        key=lambda r: float(r.get("word_x0", 0.0) or 0.0),
    )

    for r in first_frame_rows_sorted:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = compute_pixel_bbox(r)

        # get line_idx from CSV row
        line_idx = int(r.get("line_idx", 0) or 0)

        records.append(
            (start_val, end_val, rect_id, x0, y0, x1, y1, text, rect_id, line_idx)
        )
        rect_id += 1

    return records, rect_id


# ---------- SENTENCE-LEVEL RECORDS (新的句子 AOI) ---------- #

def sentence_group_to_sentence_record(
    sentence: str,
    rows: List[dict],
    fps: float,
    negative_times: bool,
    next_id_start: int,
) -> Tuple[List[Tuple[int, int, int, int, int, int, str]], int]:
    """
    For one sentence group (sentence-level AOI):
      - find earliest and latest frame index across all rows
      - 使用句子内所有词来计算一个整体 bbox：
          left   = 所有词中最小 x0
          right  = 所有词中最大 x1
          top    = 所有词中最小 y0
          bottom = 所有词中最大 y1
      - 只使用 earliest frame 的词来计算 bbox（保持和 word-level 同样假设）
      - 返回一个 record:
          (start_val, end_val, rect_id, x0, y0, x1, y1, sentence_text)
    """
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
    if negative_times:
        start_val = -start_ms
        end_val = -end_ms
    else:
        start_val = start_ms
        end_val = end_ms

    # 仍然只使用最早 frame 的字来决定 AOI
    first_frame_rows = frame_map[min_frame]

    xs0: List[int] = []
    ys0: List[int] = []
    xs1: List[int] = []
    ys1: List[int] = []

    for r in first_frame_rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = compute_pixel_bbox(r)
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)

    if not xs0:
        return [], next_id_start

    x0 = min(xs0)
    y0 = min(ys0)
    x1 = max(xs1)
    y1 = max(ys1)

    rect_id = next_id_start
    # sentence 这里你原来用的是 group 名字，包含 __groupN，保留这个行为
    record = (start_val, end_val, rect_id, x0, y0, x1, y1, sentence)

    return [record], rect_id + 1


# ---------- 公共：时间扩展（±300ms，不重叠） ---------- #

def extend_spans_with_padding(
    records: List[tuple],
    negative_times: bool,
    padding: int = 300,
) -> List[tuple]:
    """
    给所有记录的时间做统一的 ±padding ms 扩展，并避免相邻句子之间时间重叠。

    records: 任意 tuple 序列，但要求前两个元素是 (start_val, end_val)
    返回：更新 start/end 后的新 records
    """
    if not records:
        return records

    # 1) collect unique (start_ms, end_ms) in logical (positive) ms
    unique_spans = set()

    for rec in records:
        start_val, end_val = rec[0], rec[1]
        if negative_times:
            s_ms = -start_val
            e_ms = -end_val
        else:
            s_ms = start_val
            e_ms = end_val
        unique_spans.add((s_ms, e_ms))

    # 2) sort spans in chronological order
    spans_sorted = sorted(unique_spans, key=lambda t: t[0])

    # 3) extend each span by padding on both sides, clamp to avoid overlap
    span_mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}
    prev_end: int | None = None

    for (s_ms, e_ms) in spans_sorted:
        new_s = s_ms - padding
        new_e = e_ms + padding

        if prev_end is not None and new_s < prev_end:
            # clamp start so it does not overlap previous sentence
            new_s = prev_end

        # safety: ensure non-negative duration
        if new_s > new_e:
            # fall back to original if padding fully collapsed
            new_s, new_e = s_ms, e_ms

        span_mapping[(s_ms, e_ms)] = (new_s, new_e)
        prev_end = new_e

    # 4) apply adjusted spans back to records
    adjusted_records: List[tuple] = []
    for rec in records:
        start_val, end_val = rec[0], rec[1]
        rest = rec[2:]

        if negative_times:
            s_ms = -start_val
            e_ms = -end_val
        else:
            s_ms = start_val
            e_ms = end_val

        new_s_ms, new_e_ms = span_mapping[(s_ms, e_ms)]

        if negative_times:
            new_start_val = -new_s_ms
            new_end_val = -new_e_ms
        else:
            new_start_val = new_s_ms
            new_end_val = new_e_ms

        adjusted_records.append((new_start_val, new_end_val, *rest))

    return adjusted_records


# ---------- 只在 word-level 里用的：避免同一句内多行 y 轴重叠 ---------- #

def adjust_y_per_sentence(
    records: List[Tuple[int, int, int, int, int, int, str, int, int]]
) -> List[Tuple[int, int, int, int, int, int, str, int, int]]:
    """
    Use CSV's line_idx column to avoid overlapping by cutting between line centers.

    Behaviour:
      - Group rectangles by (start,end) => one sentence span.
      - Within a span, group by line_idx (from CSV, stored at index 9).
      - For each line, keep its own min y0 and max y1 where possible.
      - Compute midpoints between line centers and use them as boundaries.

      For 2 lines:
        line0: y0 = original min y0 of line0, y1 = midpoint
        line1: y0 = midpoint, y1 = original max y1 of line1

      For 3+ lines:
        first:  [min_y0_0, mid01]
        middle: [mid(i-1,i), mid(i,i+1)]
        last:   [mid(last-1,last), max_y1_last]
    """
    if not records:
        return records

    # Group by sentence span (start,end)
    span_to_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        start_val, end_val = rec[:2]
        span_to_indices[(start_val, end_val)].append(idx)

    new_records = list(records)

    for span, idxs in span_to_indices.items():
        if len(idxs) <= 1:
            continue

        # Group by line_idx (stored at index 9)
        line_groups: Dict[int, List[int]] = defaultdict(list)
        for idx in idxs:
            line_idx = int(new_records[idx][9])
            line_groups[line_idx].append(idx)

        if len(line_groups) <= 1:
            # Only one visual line: nothing to do
            continue

        # Compute per-line stats: min_y0, max_y1, center
        # tuple = (line_idx, center, min_y0, max_y1, indices)
        line_infos = []
        for line_idx, line_indices in line_groups.items():
            y0s = []
            y1s = []
            centers = []
            for i in line_indices:
                y0 = new_records[i][4]
                y1 = new_records[i][6]
                y0s.append(y0)
                y1s.append(y1)
                centers.append((y0 + y1) / 2.0)
            min_y0 = min(y0s)
            max_y1 = max(y1s)
            center = sum(centers) / len(centers)
            line_infos.append((line_idx, center, min_y0, max_y1, line_indices))

        # Sort lines top -> bottom by center
        line_infos.sort(key=lambda x: x[1])
        k = len(line_infos)

        # Compute midpoints between adjacent line centers
        mids: List[int] = []
        for i in range(k - 1):
            c1 = line_infos[i][1]
            c2 = line_infos[i + 1][1]
            mids.append(int(round((c1 + c2) / 2.0)))

        # Assign segments per line
        for i, (line_idx, center, min_y0, max_y1, line_indices) in enumerate(line_infos):
            if k == 1:
                seg_y0, seg_y1 = min_y0, max_y1
            elif i == 0:
                # First line: keep its top, cut bottom at first midpoint
                seg_y0 = min_y0
                seg_y1 = mids[0]
            elif i == k - 1:
                # Last line: start at last midpoint, keep its bottom
                seg_y0 = mids[-1]
                seg_y1 = max_y1
            else:
                # Middle lines: between two midpoints
                seg_y0 = mids[i - 1]
                seg_y1 = mids[i]

            # Safety: if segment collapses or is reversed, fall back to original line bounds
            if seg_y1 <= seg_y0:
                seg_y0 = min_y0
                seg_y1 = max_y1

            # Apply new y0/y1 to all words in this line
            for idx in line_indices:
                start_val, end_val, rect_id, x0, _, x1, _, text, rid, line_idx_val = new_records[idx]
                new_records[idx] = (
                    start_val, end_val, rect_id,
                    x0, seg_y0, x1, seg_y1,
                    text, rid, line_idx_val
                )

    return new_records


# ---------- 写 IAS：WORD-LEVEL & SENTENCE-LEVEL 各一份 ---------- #

def write_ias_word_level(
    groups: Dict[str, List[dict]],
    out_path: Path,
    fps: float,
    negative_times: bool = False,
):
    """
    WORD-LEVEL AOI：
      - 每个词一个 RECTANGLE
      - 时间基于句子出现的最早/最晚 frame，并统一加上 ±300ms padding（不重叠）
    """
    all_records: List[Tuple[int, int, int, int, int, int, str, int, int]] = []
    next_id = 1

    for sentence, rows in groups.items():
        recs, next_id = sentence_group_to_word_records(
            sentence,
            rows,
            fps=fps,
            negative_times=negative_times,
            next_id_start=next_id,
        )
        all_records.extend(recs)

    # 扩展时间 ±300ms，避免重叠
    all_records = extend_spans_with_padding(all_records, negative_times)

    # 修正 y 轴，避免同一句内部多行重叠
    all_records = adjust_y_per_sentence(all_records)

    # Sort by start time then id (you can reverse if needed)
    all_records.sort(key=lambda r: (r[0], r[2]), reverse=True)

    with out_path.open("w", encoding="utf-8") as f:
        for start_val, end_val, rect_id, x0, y0, x1, y1, text, rid, line_idx in all_records:
            label = f"[{text}#{rect_id}]"
            line = (
                f"{start_val} {end_val} RECTANGLE {rect_id} "
                f"{x0} {y0} {x1} {y1} {label}"
            )
            f.write(line + "\n")

    print(f"Wrote WORD-LEVEL IAS file with {len(all_records)} rectangles to: {out_path}")


def write_ias_sentence_level(
    groups: Dict[str, List[dict]],
    out_path: Path,
    fps: float,
    negative_times: bool = False,
):
    """
    SENTENCE-LEVEL AOI：
      - 每个句子一个 RECTANGLE
      - AOI 边界：
          left   = 该句中所有词最小 x0
          right  = 该句中所有词最大 x1
          top    = 该句中所有词最小 y0
          bottom = 该句中所有词最大 y1
        （只使用最早 frame 的词，和 word-level 的假设一致）
      - 时间基于句子出现的最早/最晚 frame，并统一加上 ±300ms padding（不重叠）
    """
    # record: (start_val, end_val, rect_id, x0, y0, x1, y1, sentence_text)
    all_records: List[Tuple[int, int, int, int, int, int, str]] = []
    next_id = 1

    for sentence, rows in groups.items():
        recs, next_id = sentence_group_to_sentence_record(
            sentence,
            rows,
            fps=fps,
            negative_times=negative_times,
            next_id_start=next_id,
        )
        all_records.extend(recs)

    # 扩展时间 ±300ms，避免重叠（对句子级 AOI 同样适用）
    all_records = extend_spans_with_padding(all_records, negative_times)

    # Sort by start time then id (你之前是 reverse=True，这里保持一致)
    all_records.sort(key=lambda r: (r[0], r[2]), reverse=True)

    with out_path.open("w", encoding="utf-8") as f:
        for start_val, end_val, rect_id, x0, y0, x1, y1, text in all_records:
            label = f"[{text}#{rect_id}]"
            line = (
                f"{start_val} {end_val} RECTANGLE {rect_id} "
                f"{x0} {y0} {x1} {y1} {label}"
            )
            f.write(line + "\n")

    print(f"Wrote SENTENCE-LEVEL IAS file with {len(all_records)} rectangles to: {out_path}")


# ----------------- CLI ----------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert an OCR CSV into IAS format, both word-level and sentence-level:\n"
            "  • WORD-LEVEL: 一句内每个词一个 AOI（矩形），时间为该句在视频中出现的时间段。\n"
            "  • SENTENCE-LEVEL: 一句一个 AOI（矩形），边界为该句内所有词的最小/最大 x,y。\n"
            "  • 相同（或相似）句子在多个 frame 出现时，使用最早 frame 的空间布局，\n"
            "    但时间从最早 frame 到最晚 frame，并统一加 ±300ms padding（防重叠）。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help=(
            "Input CSV with columns: source_file, sentence, text, "
            "word_x0, word_y0, word_x1, word_y1, page_width, page_height, line_idx."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.ias",
        help=(
            "Base output IAS file path. "
            "实际会生成两个文件：<base>_word.ias 和 <base>_sentence.ias"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second used to convert frame index to time (ms).",
    )
    parser.add_argument(
        "--positive-times",
        action="store_true",
        help="Use positive ms for times instead of negative values (default: negative).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    out_base = Path(args.output)

    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    # 统一基名：<stem>_word.ias / <stem>_sentence.ias
    base_stem = out_base.stem
    parent = out_base.parent if out_base.parent != Path("") else Path(".")
    word_out_path = parent / f"{base_stem}_word.ias"
    sentence_out_path = parent / f"{base_stem}_sentence.ias"

    groups = load_csv_group_by_sentence(csv_path)

    negative_times = not args.positive_times

    # WORD-LEVEL AOI IAS
    write_ias_word_level(
        groups,
        word_out_path,
        fps=args.fps,
        negative_times=negative_times,
    )

    # SENTENCE-LEVEL AOI IAS
    write_ias_sentence_level(
        groups,
        sentence_out_path,
        fps=args.fps,
        negative_times=negative_times,
    )


if __name__ == "__main__":
    main()
