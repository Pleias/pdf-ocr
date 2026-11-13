#!/usr/bin/env python3
"""
pdf-ocr — run the OCR pipeline on your own machine, no scheduler involved.

    pdf-ocr --pdf ./papers --out ./ocr --model ./DotsOCR

The Slurm launchers exist for corpora that need many GPUs. This entry point is
for the ordinary case: one machine, one H100, a folder of PDFs.

It is the same worker underneath, so you keep the useful properties: each PDF is
claimed before work starts and skipped if its parquet already exists, which
means you can stop it with Ctrl-C and run it again to carry on.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pdf-ocr",
        description="OCR a folder of PDFs to one parquet per document.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pdf", required=True,
                    help="a PDF file, or a folder of PDFs (subfolders are processed too)")
    ap.add_argument("--out", required=True, help="output root; parquets land under <out>/")
    ap.add_argument("--model", required=True,
                    help="local folder holding the dots.ocr weights and modeling code "
                         "(huggingface-cli download rednote-hilab/dots.ocr --local-dir ./DotsOCR)")
    ap.add_argument("--cache", default=None,
                    help="scratch for model/vLLM caches [default: <out>/.cache]")
    ap.add_argument("--dpi", type=int, default=150,
                    help="rasterisation DPI; bbox is expressed in these pixels")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="pages rasterised and queued at once; tuned for one H100 (80 GB)")
    ap.add_argument("--gpu-memory", type=float, default=0.95,
                    help="fraction of GPU memory vLLM may use; tuned for one H100 (80 GB)")
    ap.add_argument("--stats", action="store_true", help="write a per-PDF stats sidecar")
    ap.add_argument("--offline", action="store_true",
                    help="never contact the Hub (required on an air-gapped machine)")
    a = ap.parse_args(argv)

    pdf = Path(a.pdf).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    cache = Path(a.cache).expanduser().resolve() if a.cache else out / ".cache"
    if not pdf.exists():
        sys.exit(f"no such path: {pdf}")
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    # A single file is the common local case; the worker drains directories, so
    # stage a one-entry directory of symlinks rather than making the user mkdir.
    # The worker names the output folder after the input folder, so the staging
    # folder carries the PDF's real parent name, not the random temp name.
    tmp = None
    if pdf.is_file():
        tmp = tempfile.mkdtemp(prefix="pdf-ocr-", dir=str(cache))
        stage = Path(tmp) / (pdf.parent.name or "pdf")
        stage.mkdir()
        os.symlink(pdf, stage / pdf.name)
        pdf_dirs = str(stage)
    else:
        subs = sorted(d for d in pdf.iterdir() if d.is_dir())
        pdf_dirs = ",".join(str(d) for d in subs) if subs else str(pdf)

    # Must be set before worker is imported: it reads them at module scope.
    os.environ["DOTS_MODEL_PATH"] = str(Path(a.model).expanduser())
    os.environ["DOTS_CACHE_ROOT"] = str(cache)
    # Unlike a compute node, a laptop usually has internet. Let the Hub resolve
    # the model unless the user asks for strict offline.
    os.environ.setdefault("HF_HUB_OFFLINE", "1" if a.offline else "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if a.offline else "0")

    from . import worker  # noqa: E402  (env above must land first)

    argv2 = ["--pdf-dirs", pdf_dirs,
             "--results-root", str(out),
             "--log-dir", str(out / "logs"),
             "--dpi", str(a.dpi),
             "--batch-size", str(a.batch_size),
             "--gpu-memory", str(a.gpu_memory)]
    if not a.stats:
        argv2.append("--no-stats")
    try:
        return worker.main(argv2) or 0
    finally:
        if tmp:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
