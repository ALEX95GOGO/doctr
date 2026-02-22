#!/bin/bash
# ---- Persistent logging ----
WORKDIR="../"
LOGDIR="${WORKDIR}/logs"
#mkdir -p "$LOGDIR"

date
echo
#export HF_HOME="${WORKDIR}/hfcache/"

# ---- Run ----
for i in {1..42}; do
  cd "$WORKDIR"
  cd doctr

  FILE_NAME=$(printf "B%03d.mp4" "$i")
  BASE_NAME="${FILE_NAME%.*}"

  echo "=== Processing $FILE_NAME ==="
  date

  FRAMES_DIR="${FRAMES_BASE}_${BASE_NAME}"
  DOCTR_OUT_DIR="${DOCTR_OUT_BASE}_${BASE_NAME}"

  rm -rf "$FRAMES_DIR" "$DOCTR_OUT_DIR"
  mkdir -p "$FRAMES_DIR" "$DOCTR_OUT_DIR"

  MERGED_CSV="${WORKDIR}/cleaned_${BASE_NAME}.csv"
  SENTENCE_CSV="${WORKDIR}/cleaned_${BASE_NAME}_sentence.csv"
  SUBS_CSV="${WORKDIR}/cleaned_subtitles_${BASE_NAME}.csv"
  IAS_OUT="${WORKDIR}/${BASE_NAME}.ias"

  echo "[4/4] Matching + IAS export"
  python match_master.py "$MERGED_CSV" "Subtitles_all_cleaned_category.csv" \
    -o "$SUBS_CSV" --group-threshold 0.95 --match-score 80 --scorer WRatio --drop-if-no-match
  python csv_to_ias_subtitle_word_level_line_info.py "$SUBS_CSV" "Subtitles_all_cleaned_category.csv"  -o "$IAS_OUT" --fps 30

  echo "Cleaning temporary files"
  rm -rf "$FRAMES_DIR" "$DOCTR_OUT_DIR"

  echo "=== Finished $FILE_NAME ==="
  echo
done

echo "=== Job finished successfully ==="
date


