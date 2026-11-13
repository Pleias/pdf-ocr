#!/bin/bash
# One worker over OCR_PDF_ROOT. Good for a handful of PDFs or a smoke test.
#   sbatch slurm/ocr_single.slurm
#SBATCH --job-name=dots_ocr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --output=ocr_single_%j.out
# sbatch runs a spooled copy of this file, so $0 is not in the repo. submit.sh
# passes OCR_HOME; a plain `sbatch slurm/...` from the repo root also works.
OCR_HOME="${OCR_HOME:-$SLURM_SUBMIT_DIR}"
[ -f "$OCR_HOME/slurm/_common.sh" ] || { echo "cannot find the repo: submit with ./submit.sh, or sbatch from the repo root"; exit 2; }
. "$OCR_HOME/slurm/_common.sh"
echo "=== $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run_worker "$OCR_PDF_ROOT"
rc=$?
echo "=== exit=$rc $(date) ==="
exit $rc
