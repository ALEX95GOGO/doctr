#!/usr/bin/env python3
import argparse
import csv
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
    """Load master/ground-truth sentences from a CSV column. De-dupes while preserving order."""
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


def match_to_ground_truth_ordered(
    variants: List[str],
    master_sentences: List[str],
    start_idx: int,
    last_matched_idx: Optional[int],
    lookahead: int,
    min_score: float,
    scorer_name: str = "WRatio",
    output_if_no_match: str = "-",
) -> Tuple[str, float, Optional[int]]:
    """
    Ordered matching:
    - Only search forward from start_idx within a lookahead window.
    - Also allow matching last_matched_idx again (handles redundant OCR repeats).
    Returns (best_sentence, best_score, best_master_index).
    RapidFuzz scores are 0..100.
    """
    query = build_query_from_variants(variants)
    if not query:
        return output_if_no_match, 0.0, None

    scorer_map = {
        "WRatio": fuzz.WRatio,
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
    }
    scorer = scorer_map.get(scorer_name, fuzz.WRatio)

    # Build candidate indices: last match (optional) + forward window
    candidates: List[Tuple[int, str]] = []

    if last_matched_idx is not None and 0 <= last_matched_idx < len(master_sentences):
        candidates.append((last_matched_idx, master_sentences[last_matched_idx]))

    end_idx = min(len(master_sentences), start_idx + lookahead)

    for i in range(start_idx, end_idx):
        candidates.append((i, master_sentences[i]))

    if not candidates:
        return output_if_no_match, 0.0, None

    candidate_texts = [t for (_i, t) in candidates]
    #print(query, candidate_texts)
    best = process.extractOne(query, candidate_texts, scorer=scorer, processor=None)
    if best is None:
        return output_if_no_match, 0.0, None

    best_sentence, score, local_idx = best
    best_master_idx = candidates[local_idx][0]

    if score >= min_score:
        return best_sentence, float(score), best_master_idx
    return output_if_no_match, float(score), None


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
        #print(rep_sentence, s)
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


# -------------------------------------------------------------
#                      MAIN PIPELINE
# -------------------------------------------------------------

def process_csv(
    input_csv: Path,
    output_csv: Path,
    master_csv: Path,
    sentence_col: str = "sentence",
    master_col: str = "\ufeffSubtitles",
    group_threshold: float = 0.85,
    match_score: float = 90.0,
    scorer_name: str = "WRatio",
    output_if_no_match: str = "-",
    drop_if_no_match: bool = True,
    lookahead: int = 50,
):
    """
    Post-process OCR sentences by mapping to a master (ground-truth) list.

    NEW ASSUMPTION:
    - The correct sentence appears IN ORDER in the master file (monotonic progression).
      The input may contain noise and redundant repeats.

    Steps (same architecture as original):
    - Load input rows (must include sentence_col).
    - Build contiguous segments of identical sentence_col.
    - Group adjacent segments by difflib similarity >= group_threshold.
    - For each group: gather unique sentence variants, then match to closest master sentence
      but ONLY searching forward from the last matched position (plus optional repeat of last match).
    - If drop_if_no_match and no match: drop those rows from output.
      Otherwise: write output_if_no_match into sentence_col for those rows.
    - Writes output CSV with same remaining rows/columns.
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

    print(f"Loaded {len(rows)} rows from: {input_csv}")

    # 2) Load master sentences
    master_sentences = load_master_sentences(master_csv, col=master_col)
    if not master_sentences:
        raise ValueError("Master CSV loaded but contained no sentences.")
    print(f"Loaded {len(master_sentences)} master sentences from: {master_csv} (col='{master_col}')")

    # 3) Build segments and groups
    segments = build_segments(rows, sentence_col=sentence_col)
    print(f"Built {len(segments)} segments (contiguous same-sentence blocks).")

    groups = group_segments_by_similarity(segments, threshold=group_threshold)
    print(f"Grouped into {len(groups)} groups using group_threshold={group_threshold}.")
    #import pdb; pdb.set_trace()
    # 4) For each group, match to master and write back / mark for dropping
    cache: Dict[Tuple[str, ...], Tuple[str, float, Optional[int]]] = {}
    matched_group_count = 0
    drop_row_indices: set[int] = set()

    master_ptr = 0
    last_matched_idx: Optional[int] = None
    
    def letter_ratio(s):
        letters = sum(c.isalpha() for c in s)
        return letters / max(len(s), 1)
    
    def looks_like_word(s, threshold=0.6):
        return letter_ratio(s) >= threshold


    for group_id, seg_indices in enumerate(groups):
        variants: List[str] = []
        drop_group = False

        for seg_idx in seg_indices:
            _, _, sent = segments[seg_idx]
            s = (sent or "").strip()
        
            if not s:
                continue
            if not looks_like_word(s):
                drop_group = True
                break
                continue
            if len(s) <= 3:
                drop_group = True
                break
            if s in variants:
                continue
        
            variants.append(s)
        
        if drop_group:
            for seg_idx in seg_indices:
                start_idx, end_idx, _ = segments[seg_idx]
                drop_row_indices.update(range(start_idx, end_idx + 1))
            continue
        if not variants:
            continue
        #print(variants)
        #import pdb; pdb.set_trace()

        key = tuple(variants)
        if key in cache:
            chosen, score, chosen_master_idx = cache[key]
        else:
            chosen, score, chosen_master_idx = match_to_ground_truth_ordered(
                variants=variants,
                master_sentences=master_sentences,
                start_idx=master_ptr,
                last_matched_idx=last_matched_idx,
                lookahead=lookahead,
                min_score=match_score,
                scorer_name=scorer_name,
                output_if_no_match=output_if_no_match,
            )
            if score == 85.5:
                continue
            cache[key] = (chosen, score, chosen_master_idx)
            matched_group_count += 1

            where = f"ptr={master_ptr}"
            if chosen_master_idx is not None:
                where += f" -> master[{chosen_master_idx}]"
            if score>75:
                print(f"[group {group_id}] {where} score={score:.1f} variants={variants} -> {chosen!r}")
                
        # Advance pointer if we matched a NEW master index (not just repeating last match)
        if chosen != output_if_no_match and chosen_master_idx is not None:
            if last_matched_idx is None or chosen_master_idx != last_matched_idx:
                master_ptr = chosen_master_idx + 1
                #master_ptr += 1
                last_matched_idx = chosen_master_idx
            # else: repeated same sentence due to redundancy; do not advance

        if drop_if_no_match and chosen == output_if_no_match:
            # mark all rows in these segments for deletion
            for seg_idx in seg_indices:
                start_idx, end_idx, _ = segments[seg_idx]
                drop_row_indices.update(range(start_idx, end_idx + 1))
            continue

        # Write matched sentence back to ALL rows in all segments of this group
        for seg_idx in seg_indices:
            start_idx, end_idx, _ = segments[seg_idx]
            for i in range(start_idx, end_idx + 1):
                rows[i][sentence_col] = chosen

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
            "Post-process OCR sentences by mapping them to a master (ground-truth) sentence list.\n"
            "Groups contiguous same-sentence segments, then groups adjacent segments with similar\n"
            "sentences and replaces each group with the best-matching master sentence.\n"
            "NEW: assumes master order matches progression (monotonic), so matching searches forward\n"
            "from the last matched index with a configurable lookahead window.\n"
            "Optionally drops rows when no match is found."
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
        help="Master (ground-truth) CSV containing all valid sentences.",
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
        help="Column in master CSV that contains ground-truth sentences.",
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
        default=90.0,
        help="0..100 minimum RapidFuzz match score required to accept a master sentence; else no-match behavior applies.",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="WRatio",
        choices=["WRatio", "ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"],
        help="RapidFuzz scoring function to use for matching variants to master sentences.",
    )
    parser.add_argument(
        "--no-match-output",
        type=str,
        default="-",
        help="What to write when no master sentence matches above --match-score (and not dropping).",
    )
    parser.add_argument(
        "--drop-if-no-match",
        action="store_true",
        help="If set, drop rows for groups that do not match the master above --match-score.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=25,
        help="How many master sentences ahead to search from the current pointer (ordered matching).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    drop = bool(args.drop_if_no_match)

    process_csv(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        master_csv=Path(args.master),
        sentence_col=args.sentence_col,
        master_col=args.master_col,
        group_threshold=args.group_threshold,
        match_score=args.match_score,
        scorer_name=args.scorer,
        output_if_no_match=args.no_match_output,
        drop_if_no_match=drop,
        lookahead=args.lookahead,
    )

