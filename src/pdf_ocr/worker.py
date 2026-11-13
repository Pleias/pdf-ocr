#!/usr/bin/env python3
"""
Persistent DotsOCR worker: loads the model once, then drains a pool of PDF
directories until they are empty or the Slurm walltime runs out.

Designed for many workers against shared storage. Each worker claims a PDF with
an atomic O_CREAT|O_EXCL marker, so N workers can share one directory tree with
no coordinator and no double work.

Every path is read from the environment, so the Slurm script is the single
source of truth. Required:

    DOTS_MODEL_PATH   directory holding the DotsOCR weights + modeling code
    DOTS_CACHE_ROOT   writable scratch for HF / vLLM / dataset caches

Optional:
    HF_HOME, VLLM_CACHE_ROOT, HF_DATASETS_CACHE   override individual caches
    HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE          default 1 (compute nodes are
                                                  usually offline; set 0 if not)

Resilience notes earned in production:
- SLURM_JOB_END_TIME is not exported on every site; we fall back to parsing
  `scontrol show job` and then to no deadline at all.
- The remaining-PDF list is built once per sweep with a batch listdir and
  filtered against finished parquets, instead of stat-ing each candidate.
- Workers start at a random offset and resample periodically, so they spread
  out instead of all contending on the same prefix.
"""

import os
import sys
import json
import time
import errno
import socket
import signal
import logging
import random
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse

# src/ on the path, so `python src/pdf_ocr/worker.py` works without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- paths: no defaults, because a wrong default silently writes to the wrong
# filesystem and that is far worse than an immediate error ---------------------
MODEL_PATH = os.environ.get("DOTS_MODEL_PATH")
_CACHE_ROOT = os.environ.get("DOTS_CACHE_ROOT")
if not MODEL_PATH or not _CACHE_ROOT:
    sys.exit("DOTS_MODEL_PATH and DOTS_CACHE_ROOT must be set "
             "(see config.example.env)")

if not os.path.isfile(os.path.join(MODEL_PATH, "modeling_dots_ocr_vllm.py")):
    sys.exit(f"{MODEL_PATH} is not a dots.ocr model directory: it must be a local folder "
             "holding the weights and the modeling_*.py files. Fetch it with\n"
             "  huggingface-cli download rednote-hilab/dots.ocr --local-dir ./DotsOCR")

# The model's modeling code uses package-relative imports, so it must be imported
# as a package, and vLLM's registration expects that package to be `DotsOCR`.
# Rather than requiring the user's folder to carry that exact name, link it under
# the cache. Rename-into-place keeps concurrent workers from racing.
_pkg_root = os.path.join(_CACHE_ROOT, "model_pkg")
os.makedirs(_pkg_root, exist_ok=True)
_link = os.path.join(_pkg_root, "DotsOCR")
_target = os.path.realpath(MODEL_PATH)
if os.path.realpath(_link) != _target:
    _tmp = f"{_link}.{os.getpid()}"
    os.symlink(_target, _tmp)
    os.replace(_tmp, _link)
sys.path.insert(0, _pkg_root)
os.environ['PYTHONPATH'] = f"{_pkg_root}:{MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}"
os.environ.setdefault('HF_HOME', f"{_CACHE_ROOT}/hf_home")
os.environ.setdefault('HF_DATASETS_CACHE', f"{_CACHE_ROOT}/cache_dataset")
os.environ.setdefault('VLLM_CACHE_ROOT', f"{_CACHE_ROOT}/cache_vllm")
# Most HPC compute nodes have no internet. Without these, huggingface_hub
# retries network calls and stalls the cold start by minutes.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from DotsOCR import modeling_dots_ocr_vllm  # noqa: F401
import pandas as pd

STALE_SECS = 3600
QUEUE_EMPTY_GRACE = 60
WALLTIME_SAFETY_MARGIN = 600
RANDOM_SAMPLE_TRIES = 80           # quick random probes before linear fallback
REBUILD_AFTER_PROCESSED = 200      # rebuild the entries list every N processed PDFs
_stop_requested = False
_END_TIME_CACHE = None             # resolved once, see walltime_seconds_remaining


def handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def load_ocr_modules():
    global PDFVisionProcessorVLLM, pdf_to_images, clean_temp_paths
    from pdf_ocr.dots import PDFVisionProcessorVLLM, pdf_to_images, clean_temp_paths


def setup_worker_logger(log_dir: Path, worker_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / f"worker_{worker_id}.log", mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def parse_ocr_output_to_blocks(ocr_output: str) -> List[Dict]:
    try:
        parsed = json.loads(ocr_output)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            return [
                {'block_index': idx, 'category': obj.get('category', ''),
                 'bbox': obj.get('bbox', []), 'text': obj.get('text', '')}
                for idx, obj in enumerate(parsed)
            ]
        return []
    except json.JSONDecodeError:
        return []


def process_one_pdf(pdf_path: Path, processor, dpi: int, batch_size: int,
                    logger: logging.Logger) -> Tuple[List[Dict], int]:
    results = []
    total_pages = 0
    img_paths = []
    try:
        # NOTE: pdf_to_images writes page images through tempfile, so they land
        # in $TMPDIR. Point TMPDIR at a LOCAL disk, never at a network share —
        # on a cluster this is the single largest throughput lever.
        img_paths = pdf_to_images(str(pdf_path), dpi=dpi)
        if not img_paths:
            logger.warning(f"  No images extracted from {pdf_path.name}")
            return results, 0
        total_pages = len(img_paths)
        logger.info(f"  {pdf_path.name}: {total_pages} pages")
        for batch_start in range(0, total_pages, batch_size):
            batch_paths = img_paths[batch_start:batch_start + batch_size]
            outputs = processor.process_images_batch(batch_paths)
            for i, ocr_output in enumerate(outputs):
                page_num = batch_start + i + 1
                for block in parse_ocr_output_to_blocks(ocr_output):
                    results.append({
                        'filename': pdf_path.name,
                        'page': page_num,
                        'block_index': block['block_index'],
                        'category': block['category'],
                        'bbox': block['bbox'],
                        'text': block['text'],
                    })
    finally:
        if img_paths:
            try:
                clean_temp_paths(img_paths)
            except Exception:
                pass
    return results, total_pages


def is_claim_stale(claim_path: Path) -> bool:
    try:
        age = time.time() - claim_path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > STALE_SECS


def try_claim_pdf(pdf_path: Path, claims_dir: Path, worker_id: str,
                  logger: logging.Logger) -> Optional[Path]:
    claim_path = claims_dir / f"{pdf_path.stem}.claim"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Only do the stale check if the claim already exists — saves stat() on the common path
            if is_claim_stale(claim_path):
                try:
                    claim_path.unlink()
                    logger.debug(f"  Reaped stale claim: {claim_path.name}")
                    # Retry creation
                    try:
                        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    except OSError:
                        return None
                except FileNotFoundError:
                    return None
            else:
                return None
        else:
            raise
    try:
        payload = json.dumps({
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "start_time": datetime.now().isoformat(),
        }).encode()
        os.write(fd, payload)
    finally:
        os.close(fd)
    return claim_path


def _resolve_end_time() -> Optional[int]:
    """
    When is this worker supposed to stop?

    Under Slurm, stopping cleanly before the walltime matters: otherwise the job
    is killed mid-PDF and leaves a claim behind for STALE_SECS. Not every site
    exports SLURM_JOB_END_TIME, so resolve it three ways:
      1. SLURM_JOB_END_TIME, when the site exports it
      2. scontrol show job, parsed to an epoch
      3. None — no deadline. This is also the normal case when running
         locally with no scheduler: the worker simply runs until the pool
         is empty.
    """
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.isdigit():
        return int(end)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    try:
        out = subprocess.run(["scontrol", "show", "job", job_id, "-o"],
                             capture_output=True, text=True, timeout=30).stdout
        for field in out.split():
            if field.startswith("EndTime="):
                stamp = field.split("=", 1)[1]
                if stamp in ("Unknown", "None", ""):
                    return None
                return int(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").timestamp())
    except Exception:
        return None
    return None


def walltime_seconds_remaining() -> Optional[int]:
    global _END_TIME_CACHE
    if _END_TIME_CACHE is None:
        _END_TIME_CACHE = _resolve_end_time()
        if _END_TIME_CACHE is None:
            _END_TIME_CACHE = -1          # resolved, unavailable; do not retry
    if _END_TIME_CACHE == -1:
        return None
    return _END_TIME_CACHE - int(time.time())


def write_stats_sidecar(stats_dir: Path, pdf_stem: str, payload: Dict):
    stats_dir.mkdir(parents=True, exist_ok=True)
    tgt = stats_dir / f"{pdf_stem}.json"
    tmp = stats_dir / f"{pdf_stem}.json.tmp"
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, tgt)


def build_remaining_entries(pdf_dirs: List[Path], results_root: Path,
                            logger: logging.Logger):
    """
    Build a list of (pdf_path, output_dir, claims_dir, stats_dir) for PDFs
    that don't yet have a parquet output.

    Uses bulk os.listdir() once per folder + set membership instead of
    one stat() per PDF — orders of magnitude fewer filesystem round trips,
    which matters a great deal on a network filesystem.
    """
    entries = []
    for pdf_dir in pdf_dirs:
        folder_name = pdf_dir.name
        if folder_name == "pdfs" and pdf_dir.parent.name:
            folder_name = pdf_dir.parent.name
        output_dir = results_root / folder_name / "ocr_results"
        claims_dir = results_root / folder_name / "ocr_claims"
        stats_dir = results_root / folder_name / "ocr_stats"
        output_dir.mkdir(parents=True, exist_ok=True)
        claims_dir.mkdir(parents=True, exist_ok=True)

        # ONE listdir() instead of N stat()s
        try:
            done_stems = {f[:-len(".parquet")]
                          for f in os.listdir(output_dir)
                          if f.endswith(".parquet")}
        except FileNotFoundError:
            done_stems = set()

        try:
            all_pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        except FileNotFoundError:
            logger.warning(f"  {folder_name}: directory missing, skipping")
            continue

        remaining_in_folder = 0
        for pdf_name in all_pdfs:
            stem = pdf_name[:-4]  # ".pdf" or ".PDF"
            if stem in done_stems:
                continue
            entries.append((pdf_dir / pdf_name, output_dir, claims_dir, stats_dir))
            remaining_in_folder += 1
        logger.info(f"  {folder_name}: {remaining_in_folder} remaining "
                    f"({len(all_pdfs)} total, {len(done_stems)} done)")

    logger.info(f"  Total remaining across all folders: {len(entries):,}")
    return entries


def find_next_pdf(entries, claimed_skip_set):
    """
    Try RANDOM_SAMPLE_TRIES random positions first (fast on a mostly-unclaimed pool).
    Fall back to a linear scan from a random offset if random sampling fails.

    `claimed_skip_set` is an in-memory cache of indices we've recently seen claimed,
    to avoid re-checking them repeatedly within this worker's session.
    """
    n = len(entries)
    if n == 0:
        return None, -1

    # 1. Random sample (cheap; works while queue is mostly available)
    for _ in range(RANDOM_SAMPLE_TRIES):
        idx = random.randrange(n)
        if idx in claimed_skip_set:
            continue
        pdf, output_dir, claims_dir, stats_dir = entries[idx]
        if (output_dir / f"{pdf.stem}.parquet").exists():
            claimed_skip_set.add(idx)
            continue
        claim = claims_dir / f"{pdf.stem}.claim"
        if claim.exists() and not is_claim_stale(claim):
            claimed_skip_set.add(idx)
            continue
        return (pdf, output_dir, claims_dir, stats_dir), idx

    # 2. Linear fallback from random start (when queue is sparse / mostly claimed)
    start = random.randrange(n)
    for i in range(n):
        idx = (start + i) % n
        if idx in claimed_skip_set:
            continue
        pdf, output_dir, claims_dir, stats_dir = entries[idx]
        if (output_dir / f"{pdf.stem}.parquet").exists():
            claimed_skip_set.add(idx)
            continue
        claim = claims_dir / f"{pdf.stem}.claim"
        if claim.exists() and not is_claim_stale(claim):
            claimed_skip_set.add(idx)
            continue
        return (pdf, output_dir, claims_dir, stats_dir), idx
    return None, -1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persistent multi-folder DotsOCR worker")
    parser.add_argument("--pdf-dirs", type=str, required=True,
                        help="Comma-separated list of PDF directories")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--gpu-memory", type=float, default=0.95)
    args = parser.parse_args(argv)

    pdf_dirs = [Path(p) for p in args.pdf_dirs.split(",") if p.strip()]
    results_root = Path(args.results_root)
    log_dir = Path(args.log_dir)

    array_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    worker_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{array_idx}"
    logger = setup_worker_logger(log_dir, worker_id)

    # Seed random so workers diverge
    random.seed(int(time.time() * 1000) + array_idx * 7919)

    logger.info("=" * 60)
    logger.info(f"WORKER {worker_id} starting on {socket.gethostname()}")
    logger.info(f"Folders ({len(pdf_dirs)}):")
    for d in pdf_dirs:
        logger.info(f"  {d}")
    logger.info(f"Results root: {results_root}")
    logger.info(f"Model:        {MODEL_PATH}")
    logger.info(f"TMPDIR:       {os.environ.get('TMPDIR', '(unset)')}")
    rem = walltime_seconds_remaining()
    logger.info(f"Walltime left: {rem if rem is not None else 'unknown'}s")
    logger.info(f"Stats: {'disabled' if args.no_stats else 'enabled'}")
    logger.info("=" * 60)

    logger.info("Building initial remaining-PDF index...")
    t0 = time.time()
    entries = build_remaining_entries(pdf_dirs, results_root, logger)
    logger.info(f"Initial index built in {time.time() - t0:.1f}s")
    if not entries:
        # Re-running a finished job, or a surplus array worker: don't spend a
        # minute loading the model to do nothing.
        logger.info("Nothing to do: every PDF already has a parquet.")
        return 0

    load_ocr_modules()
    logger.info("Loading OCR model (one-time cold start)...")
    cold_start = datetime.now()
    processor = PDFVisionProcessorVLLM(
        model_path=MODEL_PATH,
        logger=logger,
        gpu_memory_utilization=args.gpu_memory,
    )
    cold_start_secs = (datetime.now() - cold_start).total_seconds()
    logger.info(f"Model loaded in {cold_start_secs:.1f}s")

    processed = 0
    empty_rounds = 0
    since_rebuild = 0
    claimed_skip_set = set()  # per-worker memo of indices already taken/done

    while not _stop_requested:
        remaining = walltime_seconds_remaining()
        if remaining is not None and remaining < WALLTIME_SAFETY_MARGIN:
            logger.info(f"Walltime ending in {remaining}s, stopping gracefully")
            break

        # Periodic rebuild: refresh entries list to drop done items + pick up changes
        if since_rebuild >= REBUILD_AFTER_PROCESSED:
            logger.info(f"Rebuilding index after {since_rebuild} PDFs...")
            t0 = time.time()
            entries = build_remaining_entries(pdf_dirs, results_root, logger)
            claimed_skip_set = set()
            logger.info(f"Rebuild took {time.time() - t0:.1f}s")
            since_rebuild = 0

        found, idx = find_next_pdf(entries, claimed_skip_set)
        if found is None:
            empty_rounds += 1
            if empty_rounds >= 3:
                # Final rebuild before giving up
                logger.info("Index empty, doing final rebuild...")
                entries = build_remaining_entries(pdf_dirs, results_root, logger)
                claimed_skip_set = set()
                found, idx = find_next_pdf(entries, claimed_skip_set)
                if found is None:
                    logger.info("Confirmed empty after rebuild, exiting")
                    break
                empty_rounds = 0
            else:
                logger.info(f"No PDFs available, sleeping {QUEUE_EMPTY_GRACE}s")
                time.sleep(QUEUE_EMPTY_GRACE)
                continue
        empty_rounds = 0
        pdf, output_dir, claims_dir, stats_dir = found

        claim = try_claim_pdf(pdf, claims_dir, worker_id, logger)
        if claim is None:
            claimed_skip_set.add(idx)
            continue
        part_path = output_dir / f"{pdf.stem}.parquet"
        if part_path.exists():
            claim.unlink(missing_ok=True)
            claimed_skip_set.add(idx)
            continue

        pdf_start = datetime.now()
        folder_name = output_dir.parent.name
        logger.info(f"[{processed + 1}] [{folder_name}] Processing {pdf.name}")
        status = "ok"
        err_msg = None
        results = []
        total_pages = 0
        try:
            results, total_pages = process_one_pdf(pdf, processor, args.dpi,
                                                   args.batch_size, logger)
            tmp = part_path.with_suffix(".parquet.tmp")
            if results:
                pd.DataFrame(results).to_parquet(tmp, index=False)
            else:
                pd.DataFrame(columns=['filename', 'page', 'block_index',
                                      'category', 'bbox', 'text']).to_parquet(tmp, index=False)
                status = "empty"
            os.replace(tmp, part_path)
            logger.info(f"  Saved {len(results)} blocks ({datetime.now() - pdf_start})")
            processed += 1
            since_rebuild += 1
            claimed_skip_set.add(idx)
        except Exception as e:
            status = "error"
            err_msg = str(e)
            logger.error(f"  FAILED on {pdf.name}: {e}", exc_info=True)
        finally:
            duration_s = (datetime.now() - pdf_start).total_seconds()
            if not args.no_stats:
                try:
                    write_stats_sidecar(stats_dir, pdf.stem, {
                        "filename": pdf.name,
                        "folder": folder_name,
                        "worker_id": worker_id,
                        "hostname": socket.gethostname(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "status": status,
                        "error": err_msg,
                        "pages": total_pages,
                        "blocks": len(results),
                        "duration_seconds": round(duration_s, 2),
                        "start_time": pdf_start.isoformat(),
                        "end_time": datetime.now().isoformat(),
                    })
                except Exception as sce:
                    logger.warning(f"  Could not write stats sidecar: {sce}")
            claim.unlink(missing_ok=True)

    logger.info(f"Worker {worker_id} done. Processed {processed} PDFs this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
