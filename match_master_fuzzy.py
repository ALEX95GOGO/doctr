#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher

from rapidfuzz import process, fuzz


def sentence_similarity(a: str, b: str) -> float:
    """Return a similarity score between 0 and 1 for two strings."""
    return SequenceMatcher(None, a, b).ratio()


# ----------------------------
# Ground-truth (master) loading
# ----------------------------

def load_master_sentences(master_csv: Path, col: str) -> List[str]:
    """Load master/ground-truth strings from a CSV column. De-dupes while preserving order."""
    with master_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or col not in reader.fieldnames:
            raise ValueError(f"Master CSV must contain column '{col}'. Found: {reader.fieldnames}")
        sents: List[str] = []
        for row in reader:
            s = (row.get(col) or "").strip()
            if s:
                sents.append(s)

    seen = set()
    unique: List[str] = []
    for s in sents:
        if s not in seen:
            unique.append(s)
            seen.add(s)

    return unique


# ----------------------------
# Matching OCR variants -> master
# ----------------------------

def build_query_from_variants(variants: List[str]) -> str:
    """
    Combine variants into a query string for matching.

    Heuristic:
    - prefer longer variants
    - join top few to capture more signal
    """
    cleaned = [v.strip() for v in variants if v and v.strip()]
    cleaned.sort(key=len, reverse=True)

    if not cleaned:
        return ""

    # Join up to 3 longest variants (works well in practice)
    return " ".join(cleaned[:3])


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def match_to_ground_truth(
    variants: List[str],
    master_sentences: List[str],
    min_score: float,
    scorer_name: str = "WRatio",
    output_if_no_match: str = "-",
    match_mode: str = "full",  # "full" or "partial_in_paragraph"
) -> Tuple[str, float]:
    """
    Return (best_text, best_score). If best_score < min_score, returns (output_if_no_match, best_score).

    Modes:
    - full: uses rapidfuzz.process.extractOne over the whole master string
    - partial_in_paragraph: treats master strings as "paragraphs" and finds best substring match inside them
      using fuzz.partial_ratio_alignment(), returning the matching substring.
    """
    query = build_query_from_variants(variants)
    query = _normalize_ws(query)
    if not query:
        return output_if_no_match, 0.0

    if match_mode == "partial_in_paragraph":
        # Find best matching substring inside any master paragraph.
        best_score = -1.0
        best_substring = ""
        for para in master_sentences:
            target = _normalize_ws(para)
            if not target:
                continue

            # Alignment gives where the best partial match sits inside target.
            # Note: returns an object-like result with score + dest_start/dest_end
            aln = fuzz.partial_ratio_alignment(query, target)
            score = float(aln.score)

            if score > best_score:
                # Extract the best matching region from the paragraph
                sub = target[aln.dest_start:aln.dest_end]
                best_score = score
                best_substring = _normalize_ws(sub)

            # Tie-breaker: if same score, prefer substring length closer to query length
            elif score == best_score and best_score >= 0:
                sub = _normalize_ws(target[aln.dest_start:aln.dest_end])
                if best_substring:
                    if abs(len(sub) - len(query)) < abs(len(best_substring) - len(query)):
                        best_substring = sub

        if best_score < 0:
            return output_if_no_match, 0.0
        if best_score >= min_score and best_substring:
            return best_substring, best_score
        return output_if_no_match, best_score

    # Default: full-string matching against master entries
    scorer_map = {
        "WRatio": fuzz.WRatio,
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
    }
    scorer = scorer_map.get(scorer_name, fuzz.WRatio)

    best = process.extractOne(query, master_sentences, scorer=scorer)
    if best is None:
        return output_if_no_match, 0.0

    best_sentence, score, _idx = best
    score = float(score)
    if score >= min_score:
        return str(best_sentence), score
    return output_if_no_match, score


# ----------------------------
# Segmenting + grouping
# ----------------------------

def build_segments(rows: List[Dict[str, str]], sentence_col: str) -> List[Tuple[int, int, str]]:
    """
    Build segments: contiguous rows that share the same sentence string.

    Returns list of (start_idx, end_idx, sentence).
    """
    segments: List[Tuple[int, int, str]] = []
    current_start: Optional[int] = None
    current_sentence: Optional[str] = None

    for idx, row in enumerate(rows):
        s = (row.get(sentence_col) or "").strip()

        if current_start is None:
            current_start = idx
            current_sentence = s
            continue

        if s == current_sentence:
            continue
        else:
            segments.append((current_start, idx - 1, current_sentence or ""))
            current_start = idx
            current_sentence = s

    if current_start is not None:
        segments.append((current_start, len(rows) - 1, current_sentence or ""))

    return segments


def group_segments_by_similarity(
    segments: List[Tuple[int, int, str]],
    threshold: float,
) -> List[List[int]]:
    """
    Group adjacent segments whose sentences are similar (>= threshold).

    segments: list of (start_idx, end_idx, sentence)
    returns: list of groups, each group is a list of segment indices
    """
    groups: List[List[int]] = []
    current_group: List[int] = []
    rep_sentence: Optional[str] = None

    for seg_idx, (_, _, sent) in enumerate(segments):
        s = (sent or "").strip()

        if not current_group:
            current_group = [seg_idx]
            rep_sentence = s
            continue

        assert rep_sentence is not None
        sim = sentence_similarity(rep_sentence, s)

        if sim >= threshold:
            current_group.append(seg_idx)
        else:
            groups.append(current_group)
            current_group = [seg_idx]
            rep_sentence = s

    if current_group:
        groups.append(current_group)

    return groups


# ----------------------------
# Word updating (optional)
# ----------------------------

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_words(text: str) -> List[str]:
    """
    Tokenize into words/punctuation (keeps punctuation as separate tokens).
    This is intentionally simple and robust for OCR.
    """
    text = _normalize_ws(text)
    if not text:
        return []
    return _WORD_RE.findall(text)


def update_word_column_for_segment(
    rows: List[Dict[str, str]],
    start_idx: int,
    end_idx: int,
    word_col: str,
    new_sentence: str,
) -> None:
    """
    Update rows[start_idx:end_idx] word_col based on row offset within the segment.

    Assumption: 1 row == 1 word token (or near enough), in order.
    """
    tokens = tokenize_words(new_sentence)
    seg_len = end_idx - start_idx + 1

    # If tokens shorter, fill remainder with empty string; if longer, truncate.
    for k in range(seg_len):
        rows[start_idx + k][word_col] = tokens[k] if k < len(tokens) else ""


# -------------------------------------------------------------
#                      MAIN PIPELINE
# -------------------------------------------------------------

def process_csv(
    input_csv: Path,
    output_csv: Path,
    master_csv: Path,
    sentence_col: str = "sentence",
    master_col: str = "\ufeffSubtitles",
    word_col: Optional[str] = None,
    group_threshold: float = 0.85,
    match_score: float = 90.0,
    scorer_name: str = "WRatio",
    match_mode: str = "full",  # "full" or "partial_in_paragraph"
    output_if_no_match: str = "-",
    drop_if_no_match: bool = True,
):
    """
    Post-process OCR sentences by mapping to a master (ground-truth) list.

    Enhancements:
    - match_mode="partial_in_paragraph": sentence query can match as substring within master paragraphs.
    - If word_col is provided, also updates that column for each segment to align with the new matched sentence.
    """
    # 1) Load rows
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or sentence_col not in fieldnames:
            raise ValueError(f"Input CSV must contain a '{sentence_col}' column. Found: {fieldnames}")
        rows = list(reader)

    if not rows:
        print("No rows found in input CSV.")
        return

    if word_col is not None and (fieldnames is None or word_col not in fieldnames):
        raise ValueError(f"--word-col was set to '{word_col}' but that column was not found. Found: {fieldnames}")

    print(f"Loaded {len(rows)} rows from: {input_csv}")

    # 2) Load master strings
    master_sentences = load_master_sentences(master_csv, col=master_col)
    if not master_sentences:
        raise ValueError("Master CSV loaded but contained no sentences/paragraphs.")
    print(f"Loaded {len(master_sentences)} master entries from: {master_csv} (col='{master_col}')")

    # 3) Build segments and groups
    segments = build_segments(rows, sentence_col=sentence_col)
    print(f"Built {len(segments)} segments (contiguous same-sentence blocks).")

    groups = group_segments_by_similarity(segments, threshold=group_threshold)
    print(f"Grouped into {len(groups)} groups using group_threshold={group_threshold}.")

    # 4) For each group, match to master and write back / mark for dropping
    cache: Dict[Tuple[str, ...], Tuple[str, float]] = {}
    matched_group_count = 0
    drop_row_indices: set[int] = set()

    for group_id, seg_indices in enumerate(groups):
        variants: List[str] = []
        for seg_idx in seg_indices:
            _, _, sent = segments[seg_idx]
            s = (sent or "").strip()
            if s and s not in variants:
                variants.append(s)

        if not variants:
            continue

        key = tuple(variants)
        if key in cache:
            chosen, score = cache[key]
        else:
            chosen, score = match_to_ground_truth(
                variants=variants,
                master_sentences=master_sentences,
                min_score=match_score,
                scorer_name=scorer_name,
                output_if_no_match=output_if_no_match,
                match_mode=match_mode,
            )
            cache[key] = (chosen, score)
            matched_group_count += 1
            print(f"[group {group_id}] mode={match_mode} score={score:.1f} variants={variants} -> {chosen!r}")

        if drop_if_no_match and chosen == output_if_no_match:
            # mark all rows in these segments for deletion
            for seg_idx in seg_indices:
                start_idx, end_idx, _ = segments[seg_idx]
                drop_row_indices.update(range(start_idx, end_idx + 1))
            continue

        # Write matched sentence back to ALL rows in all segments of this group
        for seg_idx in seg_indices:
            start_idx, end_idx, _old = segments[seg_idx]
            for i in range(start_idx, end_idx + 1):
                rows[i][sentence_col] = chosen

            # Also update word column (if requested)
            if word_col is not None:
                update_word_column_for_segment(
                    rows=rows,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    word_col=word_col,
                    new_sentence=chosen,
                )

    print(f"Processed {matched_group_count} groups (matched-to-master).")

    # 5) Drop rows (if enabled)
    if drop_if_no_match and drop_row_indices:
        before = len(rows)
        rows = [r for i, r in enumerate(rows) if i not in drop_row_indices]
        after = len(rows)
        print(f"Dropped {before - after} row(s) due to no-match output ({output_if_no_match!r}).")

    # 6) Write output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved output CSV to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process OCR sentences by mapping them to a master (ground-truth) list.\n"
            "Groups contiguous same-sentence segments, then groups adjacent segments with similar\n"
            "sentences and replaces each group with the best-matching master entry.\n"
            "Can also do partial matching inside master 'paragraph' entries and optionally update a word column."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input CSV (must include a 'sentence' column, or set --sentence-col).",
    )
    parser.add_argument(
        "master",
        type=str,
        help="Master (ground-truth) CSV containing all valid sentences/paragraphs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="cleaned_matched_output.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--sentence-col",
        type=str,
        default="sentence",
        help="Column in input CSV that contains OCR sentence text.",
    )
    parser.add_argument(
        "--master-col",
        type=str,
        default="\ufeffSubtitles",
        help="Column in master CSV that contains ground-truth entries (sentences or paragraphs).",
    )

    # NEW: optional word column to update
    parser.add_argument(
        "--word-col",
        type=str,
        default=None,
        help="Optional column in input CSV containing per-row word tokens; will be updated to match the chosen sentence.",
    )

    parser.add_argument(
        "--group-threshold",
        type=float,
        default=0.85,
        help="0..1 threshold for grouping adjacent segments using difflib similarity.",
    )
    parser.add_argument(
        "--match-score",
        type=float,
        default=60.0,
        help="0..100 minimum RapidFuzz match score required to accept a match; else no-match behavior applies.",
    )

    parser.add_argument(
        "--scorer",
        type=str,
        default="WRatio",
        choices=["WRatio", "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"],
        help="RapidFuzz scoring function to use for matching (used in match_mode=full).",
    )

    # NEW: match mode
    parser.add_argument(
        "--match-mode",
        type=str,
        default="full",
        choices=["full", "partial_in_paragraph"],
        help="Matching behavior: full compares to entire master entry; partial_in_paragraph finds best substring match inside master paragraphs.",
    )

    parser.add_argument(
        "--no-match-output",
        type=str,
        default="-",
        help="What to write when no master match is found above --match-score (and not dropping).",
    )
    parser.add_argument(
        "--drop-if-no-match",
        action="store_true",
        help="If set, drop rows for groups that do not match the master above --match-score.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    process_csv(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        master_csv=Path(args.master),
        sentence_col=args.sentence_col,
        master_col=args.master_col,
        word_col=args.word_col,
        group_threshold=args.group_threshold,
        match_score=args.match_score,
        scorer_name=args.scorer,
        match_mode=args.match_mode,
        output_if_no_match=args.no_match_output,
        drop_if_no_match=bool(args.drop_if_no_match),
    )
