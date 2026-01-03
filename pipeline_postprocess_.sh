#!/bin/bash
#PBS -q dgxa100
#PBS -P dq25
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -l jobfs=200GB
#PBS -l walltime=32:05:00
#PBS -l storage=gdata/dq25+scratch/dq25
#PBS -N my_pipeline
#PBS -o pipeline.out
#PBS -e pipeline.err
#PBS -l wd


set -euo pipefail

# ---- Persistent logging ----
WORKDIR="/g/data/dq25/ocr_zhuoli"
LOGDIR="${WORKDIR}/logs"
mkdir -p "$LOGDIR"

LOGFILE="${LOGDIR}/my_pipeline_${PBS_JOBID}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== Job started ==="
echo "Job ID: ${PBS_JOBID}"
echo "Host: $(hostname)"
date
echo

# ---- Paths ----
JOBSCRATCH="${PBS_JOBFS}/my_pipeline"
FRAMES_BASE="${JOBSCRATCH}/saved_frames"
DOCTR_OUT_BASE="${JOBSCRATCH}/doctr_output"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$JOBSCRATCH"

echo "PBS_JOBFS=${PBS_JOBFS}"
echo

# Diagnostics
echo "== Filesystem (inode) usage =="
df -ih "$WORKDIR" || true
df -ih "$PBS_JOBFS" || true
echo

# ---- Environment ----
module load cuda/12.3.2

source /scratch/dq25/hl2637/apps/miniconda3/bin/activate
conda activate base

export HF_HOME="${WORKDIR}/hfcache/"

# ---- Run ----
for i in {2..5}; do
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
  python match_master.py "$MERGED_CSV" "Subtitles_all_cleaned.csv" \
    -o "$SUBS_CSV" --group-threshold 0.95 --match-score 75 --scorer WRatio --drop-if-no-match
  python csv_to_ias_subtitle_word_level.py "$SUBS_CSV" -o "$IAS_OUT" --fps 30

  echo "Cleaning temporary files"
  rm -rf "$FRAMES_DIR" "$DOCTR_OUT_DIR"

  echo "=== Finished $FILE_NAME ==="
  echo
done

echo "=== Job finished successfully ==="
date


