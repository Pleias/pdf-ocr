#!/bin/bash
# N workers draining the same pool. Each claims PDFs atomically, so the array
# size is a throughput knob only — any N is correct, and workers that find
# nothing left exit cleanly.
#   sbatch --array=0-19 slurm/ocr_array.slurm
#SBATCH --job-name=dots_ocr_pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --output=ocr_pool_%A_%a.out
# sbatch runs a spooled copy of this file, so $0 is not in the repo. submit.sh
# passes OCR_HOME; a plain `sbatch slurm/...` from the repo root also works.
OCR_HOME="${OCR_HOME:-$SLURM_SUBMIT_DIR}"
[ -f "$OCR_HOME/slurm/_common.sh" ] || { echo "cannot find the repo: submit with ./submit.sh, or sbatch from the repo root"; exit 2; }
. "$OCR_HOME/slurm/_common.sh"

# Every immediate subdirectory of OCR_PDF_ROOT, else OCR_PDF_ROOT itself.
DIRS="${OCR_PDF_DIRS:-}"
if [ -z "$DIRS" ]; then
    DIRS=$(find "$OCR_PDF_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | paste -sd, -)
    [ -n "$DIRS" ] || DIRS="$OCR_PDF_ROOT"
fi
echo "=== $(hostname)  task ${SLURM_ARRAY_TASK_ID:-0}  $(date) ==="
echo "folders: $(echo "$DIRS" | tr ',' '\n' | wc -l)"
run_worker "$DIRS"
rc=$?
echo "=== exit=$rc $(date) ==="
exit $rc
