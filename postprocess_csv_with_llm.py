#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from difflib import SequenceMatcher


def sentence_similarity(a: str, b: str) -> float:
    """Return a similarity score between 0 and 1 for two strings."""
    return SequenceMatcher(None, a, b).ratio()


def load_generator(model_name: str):
    """
    Load a text-generation pipeline for a causal LLM.
    - Uses multiple GPUs if available (model sharded across GPUs).
    - Falls back to single GPU or CPU otherwise.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} CUDA device(s).")

    if n_gpus > 1:
        # Use all GPUs with a max_memory map
        max_memory = {i: "40GiB" for i in range(n_gpus)}  # tweak if needed
        max_memory["cpu"] = "64GiB"  # optional CPU offload budget

        print("Using multi-GPU sharding with device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,  # half precision
        )
    elif n_gpus == 1:
        print("Using single GPU (cuda:0)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cuda:0"},
            torch_dtype=torch.float16,
        )
    else:
        print("No GPU detected, running on CPU (this will be slow).")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cpu"},
        )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
    )
    return gen


def call_llm_transformers(lines: List[str], generator, model_name: str) -> str:
    """
    Given one or more noisy OCR variants of a sentence, call an open-source LLM
    (via transformers) to return one clean, corrected sentence.
    """
    variants_text = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
    prompt = (
        "You are an OCR post-processing assistant.\n"
        "You will receive several noisy OCR variants of the SAME subtitle line.\n"
        "Your tasks:\n"
        "1. Correct spelling and obvious OCR mistakes (like 'oyalty' -> 'loyalty').\n"
        "2. Merge partial fragments if needed.\n"
        "3. Output exactly ONE clean English sentence.\n"
        "4. If sentence is not complete, output: -\n"
        "5. Do NOT add new information not present in the variants.\n"
        "6. Do NOT explain your reasoning, only output the final cleaned sentence.\n"
        "7. Mind the speeling of names, like Professor Fig \n\n"
        "Variants:\n"
        f"{variants_text}\n\n"
        "Cleaned sentence:"
    )

    result = generator(prompt)[0]["generated_text"]

    # Strip prompt echo if present
    if result.startswith(prompt):
        cleaned = result[len(prompt):].strip()
    else:
        cleaned = result.strip()

    # Take only the first line
    cleaned = cleaned.splitlines()[0].strip()
    return cleaned


def build_segments(rows: List[Dict[str, str]]) -> List[Tuple[int, int, str]]:
    """
    Build segments: contiguous rows that share the same 'sentence' string.

    Returns list of (start_idx, end_idx, sentence).
    """
    segments: List[Tuple[int, int, str]] = []
    current_start = None
    current_sentence = None

    for idx, row in enumerate(rows):
        s = (row.get("sentence") or "").strip()

        if current_start is None:
            current_start = idx
            current_sentence = s
            continue

        if s == current_sentence:
            # still same segment
            continue
        else:
            # close previous segment
            segments.append((current_start, idx - 1, current_sentence))
            current_start = idx
            current_sentence = s

    if current_start is not None:
        segments.append((current_start, len(rows) - 1, current_sentence))

    return segments


def group_segments_by_similarity(
    segments: List[Tuple[int, int, str]],
    threshold: float,
) -> List[List[int]]:
    """
    Group adjacent segments whose sentences are similar.

    segments: list of (start_idx, end_idx, sentence)
    returns: list of groups, each group is a list of segment indices
    """
    groups: List[List[int]] = []
    current_group: List[int] = []
    rep_sentence: str | None = None

    for seg_idx, (_, _, sent) in enumerate(segments):
        s = sent.strip()

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


# -------------------------------------------------------------
#                      MAIN PIPELINE
# -------------------------------------------------------------

def process_csv(
    input_csv: Path,
    output_csv: Path,
    threshold: float = 0.85,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
):
    """
    Clean each group of similar 'sentence' segments using an LLM.

    - Does NOT merge or drop any rows.
    - Builds contiguous segments of identical sentences (first/last from your
      previous merging step).
    - Groups adjacent segments whose sentences are similar (>= threshold).
    - For each group, passes all sentence variants to the LLM and gets ONE
      cleaned sentence.
    - Writes that cleaned sentence back into the 'sentence' column for ALL rows
      in those segments.
    """
    # 1. Load all rows
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "sentence" not in fieldnames:
            raise ValueError("CSV must contain a 'sentence' column.")
        rows = list(reader)

    if not rows:
        print("No rows found in CSV.")
        return

    print(f"Loaded {len(rows)} rows from {input_csv}")

    # 2. Build segments and groups
    segments = build_segments(rows)
    print(f"Built {len(segments)} segments (contiguous same-sentence blocks).")

    groups = group_segments_by_similarity(segments, threshold=threshold)
    print(f"Grouped into {len(groups)} groups using threshold={threshold}.")

    # 3. Load LLM once
    print(f"Loading model: {model_name}")
    generator = load_generator(model_name)
    print("Model loaded.")

    # Optional: cache by tuple of variants to avoid recomputing
    cache: Dict[Tuple[str, ...], str] = {}
    cleaned_group_count = 0

    # 4. For each group, call LLM on all sentence variants
    for group_id, seg_indices in enumerate(groups):
        # Collect sentence variants in this group (one per segment)
        variants: List[str] = []
        for seg_idx in seg_indices:
            _, _, sent = segments[seg_idx]
            s = (sent or "").strip()
            if s and s not in variants:  # avoid duplicates in the same group
                variants.append(s)

        if not variants:
            continue

        key = tuple(variants)
        if key in cache:
            clean = cache[key]
        else:
            clean = call_llm_transformers(variants, generator, model_name)
            cache[key] = clean
            cleaned_group_count += 1
            print(f"[group {group_id}] {variants} -> {clean!r}")

        # Write cleaned sentence back to ALL rows in all segments of this group
        for seg_idx in seg_indices:
            start_idx, end_idx, _ = segments[seg_idx]
            for i in range(start_idx, end_idx + 1):
                rows[i]["sentence"] = clean

    print(f"Cleaned {cleaned_group_count} segment-groups using the LLM.")

    # 5. Write output CSV: same rows, same columns, cleaned 'sentence'
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved cleaned CSV (no row merging) to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process OCR sentences using an open-source transformers LLM.\n"
            "This script groups contiguous same-sentence segments, then groups\n"
            "adjacent segments with similar sentences and cleans each group "
            "using all variants together. Rows are NOT merged or dropped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input CSV (must include a 'sentence' column).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="cleaned_llm_output.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold (0-1) for grouping adjacent sentence segments.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face model name for the LLM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_csv(
        Path(args.input),
        Path(args.output),
        threshold=args.threshold,
        model_name=args.model_name,
    )
