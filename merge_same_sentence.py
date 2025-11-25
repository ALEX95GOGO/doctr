#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from difflib import SequenceMatcher


def sentence_similarity(a: str, b: str) -> float:
    """Return a similarity score between 0 and 1 for two strings."""
    return SequenceMatcher(None, a, b).ratio()


def is_valid_sentence(sentence: str) -> bool:
    """
    A sentence is considered valid if:
    - non-empty after stripping
    - contains at least 3 alphabetic characters (letters)
    """
    if not sentence:
        return False
    s = sentence.strip()
    if not s:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    return len(letters) >= 3


def merge_adjacent_similar_sentence_keep_ends(
    input_csv: Path,
    output_csv: Path,
    threshold: float = 0.8,
):
    """
    Merge adjacent runs of similar sentences.

    - Rows with empty/too-short sentences are discarded.
    - First, we form "segments": contiguous rows that share the same
      'sentence' value (after stripping).
    - Then we group these segments if their representative sentences
      are similar (SequenceMatcher ratio >= threshold).
    - For each group of segments:
        * keep the FIRST segment (all its rows)
        * keep the LAST segment (all its rows, if different)
        * drop any segments in between
    - All original columns are preserved for kept rows.
    """
    # Try to auto-detect delimiter (comma vs tab etc.)
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # fallback to default (comma)

        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = reader.fieldnames
        if not fieldnames or "sentence" not in fieldnames:
            raise ValueError("Input CSV must contain a 'sentence' column.")

        all_rows = list(reader)

    if not all_rows:
        print("No rows found in input CSV.")
        return

    # Filter out rows with invalid sentences (empty or <3 letters)
    rows = []
    for row in all_rows:
        s = (row.get("sentence") or "").strip()
        if is_valid_sentence(s):
            rows.append(row)

    if not rows:
        print("No valid sentences (>=3 letters) found; nothing to write.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build safe writer settings (avoid broken dialect flags from Sniffer)
    delimiter = getattr(dialect, "delimiter", ",")
    quotechar = getattr(dialect, "quotechar", '"')

    # ---------------------------------------------------------
    # 1) Build segments: contiguous rows with the SAME sentence
    # ---------------------------------------------------------
    segments = []  # each: (start_idx, end_idx, sentence)
    current_start = None
    current_sentence = None

    for idx, row in enumerate(rows):
        s = (row.get("sentence") or "").strip()

        if current_start is None:
            # start first segment
            current_start = idx
            current_sentence = s
            continue

        if s == current_sentence:
            # still within the same sentence segment
            continue
        else:
            # close previous segment
            segments.append((current_start, idx - 1, current_sentence))
            # start new segment
            current_start = idx
            current_sentence = s

    # close last segment
    if current_start is not None:
        segments.append((current_start, len(rows) - 1, current_sentence))

    if not segments:
        print("No segments found; nothing to write.")
        return

    # ---------------------------------------------------------
    # 2) Group segments by similar sentences (adjacent only)
    # ---------------------------------------------------------
    groups: list[list[int]] = []  # list of lists of segment indices
    current_group: list[int] = []
    rep_sentence: str | None = None

    for seg_idx, (start_idx, end_idx, sent) in enumerate(segments):
        s = sent.strip()

        if not current_group:
            current_group = [seg_idx]
            rep_sentence = s
            continue

        assert rep_sentence is not None
        sim = sentence_similarity(rep_sentence, s)

        if sim >= threshold:
            # similar enough to belong to current group
            current_group.append(seg_idx)
        else:
            # start a new group
            groups.append(current_group)
            current_group = [seg_idx]
            rep_sentence = s

    if current_group:
        groups.append(current_group)

    # ---------------------------------------------------------
    # 3) Decide which segments to keep (first & last per group)
    #    and mark ALL rows in those segments as kept.
    # ---------------------------------------------------------
    keep_flags = [False] * len(rows)

    for g in groups:
        if not g:
            continue
        if len(g) == 1:
            # single segment in group -> keep all its rows
            seg_idx = g[0]
            start_idx, end_idx, _ = segments[seg_idx]
            for i in range(start_idx, end_idx + 1):
                keep_flags[i] = True
        else:
            # multiple segments -> keep all rows in first and last segment
            first_seg_idx = g[0]
            last_seg_idx = g[-1]

            # first segment
            f_start, f_end, _ = segments[first_seg_idx]
            for i in range(f_start, f_end + 1):
                keep_flags[i] = True

            # last segment
            if last_seg_idx != first_seg_idx:
                l_start, l_end, _ = segments[last_seg_idx]
                for i in range(l_start, l_end + 1):
                    keep_flags[i] = True

    # ---------------------------------------------------------
    # 4) Write output CSV with all columns, keeping only rows
    #    whose keep_flags[idx] is True.
    # ---------------------------------------------------------
    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=fieldnames,
            delimiter=delimiter,
            quotechar=quotechar,
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writeheader()

        kept = 0
        dropped = 0

        for idx, row in enumerate(rows):
            if not keep_flags[idx]:
                dropped += 1
                continue

            # Filter row to only known fieldnames (ignore any extra keys)
            clean_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(clean_row)
            kept += 1

    print(
        f"Done. Kept {kept} rows, dropped {dropped} rows "
        f"(similarity threshold = {threshold})."
    )
    print(f"Wrote merged CSV to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge adjacent rows when the 'sentence' column is identical or similar, "
            "keeping the first and last *segments* (each segment keeps ALL its rows), "
            "and discarding sentences that are empty or have fewer than 3 letters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input CSV file path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="merged_similar_sentence_keep_ends.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help=(
            "Similarity threshold (0-1) for merging adjacent sentences. "
            "Higher = stricter (default: 0.8)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_adjacent_similar_sentence_keep_ends(
        Path(args.input),
        Path(args.output),
        threshold=args.threshold,
    )
