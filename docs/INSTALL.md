# Install

Linux x86_64, Python 3.12, CUDA 12.8, one NVIDIA H100 (80 GB) per worker.

## uv

    uv venv --python 3.12 /path/to/venv
    VIRTUAL_ENV=/path/to/venv uv sync --frozen

## pip

    python3.12 -m venv /path/to/venv && . /path/to/venv/bin/activate
    pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
    pip install --no-deps .

## Offline machines

On a machine with internet:

    tools/make_wheelhouse.sh ./wheelhouse

Copy `./wheelhouse` and the repo over, then:

    python3.12 -m venv /path/to/venv && . /path/to/venv/bin/activate
    pip install --no-index --find-links ./wheelhouse -r ./wheelhouse/requirements-linux.txt
    pip install --no-index --find-links ./wheelhouse --no-deps .

Use the script, not `pip download`: run on another OS, pip silently drops the
Linux-only CUDA packages.

## Weights

    huggingface-cli download rednote-hilab/dots.ocr --local-dir ./DotsOCR

Any local folder name works. Offline: download elsewhere and copy it over.

## Pitfalls

- **Install everything from the lock in one go.** Installing vLLM on its own can
  pull transformers 5.x, which fails at load with
  `ValueError: aimv2 is already used by a Transformers config`.
- **flash-attn is required**, and pinned as a prebuilt wheel. Without it,
  start-up fails with `torch.OutOfMemoryError: … Tried to allocate 296.63 GiB`.
  If you change torch, change the flash-attn wheel to match.
