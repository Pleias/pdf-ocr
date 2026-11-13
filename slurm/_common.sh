# Sourced by every launcher. Keeps the site settings in one place.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="${OCR_CONFIG:-$HERE/config.env}"
[ -f "$CONF" ] || { echo "no config at $CONF — copy config.example.env"; exit 2; }
# shellcheck disable=SC1090
. "$CONF"

for v in OCR_VENV OCR_MODEL OCR_CACHE_ROOT OCR_PDF_ROOT OCR_RESULTS_ROOT OCR_LOG_DIR; do
    [ -n "${!v:-}" ] || { echo "$v is unset in $CONF"; exit 2; }
done
mkdir -p "$OCR_RESULTS_ROOT" "$OCR_LOG_DIR"

# `module` is not a command in a non-interactive sbatch shell until the init is
# sourced, and the init itself trips over `set -u`.
if [ -n "${OCR_MODULES:-}" ]; then
    set +u
    [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh
    module purge 2>/dev/null || true
    for m in $OCR_MODULES; do module load "$m" 2>/dev/null || echo "warn: module $m not loaded"; done
    set -u
fi

[ -x "$OCR_VENV/bin/python" ] || { echo "no venv at $OCR_VENV — see docs/INSTALL.md"; exit 2; }
# shellcheck disable=SC1091
. "$OCR_VENV/bin/activate"

# Some sites purge scratch on atime and do NOT refresh atime on read, so merely
# using the venv does not protect it. Refresh it in the background; harmless
# elsewhere. Symlinks need -h or touch follows the link and stamps the target.
( find "$OCR_VENV" ! -type l -exec touch -a -m {} + 2>/dev/null
  find "$OCR_VENV"   -type l -exec touch -h -a -m {} + 2>/dev/null ) &

export DOTS_MODEL_PATH="$OCR_MODEL"
export DOTS_CACHE_ROOT="$OCR_CACHE_ROOT"
export HF_HOME="$OCR_CACHE_ROOT/hf_home"
export VLLM_CACHE_ROOT="$OCR_CACHE_ROOT/cache_vllm"
export TRITON_CACHE_DIR="$OCR_CACHE_ROOT/cache_triton"
export TORCHINDUCTOR_CACHE_DIR="$OCR_CACHE_ROOT/cache_inductor"
export XDG_CACHE_HOME="$OCR_CACHE_ROOT/cache_xdg"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
export TMPDIR="${JOBSCRATCH:-${SLURM_TMPDIR:-$OCR_CACHE_ROOT/tmp}}"
mkdir -p "$TMPDIR"

[ -d "$OCR_MODEL" ] || { echo "no model at $OCR_MODEL"; exit 2; }

run_worker() {   # $1 = comma-separated PDF dirs
    local stats="--no-stats"; [ "${OCR_STATS:-0}" = "1" ] && stats=""
    python "$HERE/src/pdf_ocr/worker.py" \
        --pdf-dirs "$1" \
        --results-root "$OCR_RESULTS_ROOT" \
        --log-dir "$OCR_LOG_DIR" \
        $stats \
        --dpi "${OCR_DPI:-150}" \
        --batch-size "${OCR_BATCH_SIZE:-100}" \
        --gpu-memory "${OCR_GPU_MEMORY:-0.95}"
}
