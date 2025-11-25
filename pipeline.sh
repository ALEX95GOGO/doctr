#!/bin/bash
set -e

for i in {3..9}; do
    FILE_NAME=$(printf "B%03d.mp4" "$i")
    BASE_NAME="${FILE_NAME%.*}"

    echo "=== Processing $FILE_NAME ==="

    rm -rf saved_frames
    python parse_video.py --file_name "$FILE_NAME"

    cd doctr
    rm -rf output
    python scripts/detect_text.py ../saved_frames/ --recognition master -f json

    MERGED_CSV="cleaned_${BASE_NAME}.csv"
    SENTENCE_CSV="cleaned_${BASE_NAME}_sentence.csv"
    SUBS_CSV="cleaned_subtitles_${BASE_NAME}.csv"

    python merge_json.py doctr/output -o "$MERGED_CSV"
    python merge_same_sentence.py "$MERGED_CSV" -o "$SENTENCE_CSV"
    python postprocess_csv_with_llm.py "$SENTENCE_CSV" -o "$SUBS_CSV" --model-name Qwen/Qwen3-30B-A3B-Instruct-2507
    python csv_to_ias_ori.py "$SUBS_CSV" -o "${BASE_NAME}.ias" --fps 30
    cd ..
    echo "=== Finished $FILE_NAME ==="
    echo
done

