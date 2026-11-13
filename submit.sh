#!/usr/bin/env bash
# Submit a launcher with the resources from config.env.
#
# sbatch parses #SBATCH directives before the script body runs, so a sourced
# config cannot set them. This wrapper passes them on the command line instead,
# which keeps config.env as the single source of truth.
#
#   ./submit.sh single                 one worker
#   ./submit.sh array 20               twenty workers over the same pool
#   ./submit.sh single --test-only     ask Slurm when it would start
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONF="${OCR_CONFIG:-$HERE/config.env}"
[ -f "$CONF" ] || { echo "no config at $CONF — cp config.example.env config.env"; exit 2; }
# shellcheck disable=SC1090
. "$CONF"

mode="${1:-single}"; shift || true
case "$mode" in
  single) script="$HERE/slurm/ocr_single.slurm"; array="" ;;
  array)  script="$HERE/slurm/ocr_array.slurm"
          n="${1:-10}"; [[ "$n" =~ ^[0-9]+$ ]] && shift || n=10
          array="--array=0-$((n-1))" ;;
  *) echo "usage: $0 {single|array [N]} [extra sbatch args]"; exit 2 ;;
esac

opts=( -A "$OCR_ACCOUNT" --partition="$OCR_PARTITION"
       --time="$OCR_TIME" --cpus-per-task="$OCR_CPUS"
       --gres=gpu:"${OCR_GPUS:-1}" )
[ -n "${OCR_QOS:-}" ]        && opts+=( --qos="$OCR_QOS" )
[ -n "${OCR_CONSTRAINT:-}" ] && opts+=( -C "$OCR_CONSTRAINT" )
[ -n "$array" ]              && opts+=( "$array" )

echo "submitting $mode with: ${opts[*]}"
echo "  config: $CONF"
echo "  pdfs:   $OCR_PDF_ROOT"
echo "  out:    $OCR_RESULTS_ROOT"
# sbatch runs a spooled copy of the launcher, which then cannot locate the repo
# from $0. Tell it explicitly.
OCR_HOME="$HERE" OCR_CONFIG="$CONF" exec sbatch --export=ALL,OCR_HOME="$HERE",OCR_CONFIG="$CONF" \
    "${opts[@]}" "$@" "$script"
