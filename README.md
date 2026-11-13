# pdf-ocr

PDF in, parquet out. Runs [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)
over a folder of PDFs and writes one parquet per document: every layout block of
every page, with its category, bounding box and text, in reading order.

## Requirements

Linux x86_64, Python 3.12, CUDA 12.8, one NVIDIA H100 (80 GB).

## Install

```bash
git clone <this repo> && cd pdf-ocr
uv venv --python 3.12 && uv sync --frozen
huggingface-cli download rednote-hilab/dots.ocr --local-dir ./DotsOCR
```

Without uv, offline machines, pitfalls: [docs/INSTALL.md](docs/INSTALL.md).

## Run

```bash
uv run pdf-ocr --pdf ./papers --out ./ocr --model ./DotsOCR
```

`--pdf` takes a file, a folder of PDFs, or a folder of folders. Output:
`./ocr/<folder>/ocr_results/<name>.parquet`, where `<folder>` is the PDF's folder
(a folder named `pdfs` takes its parent's name).

Interrupt and re-run at will: finished PDFs are skipped.

## Output

| column | |
|---|---|
| `filename` | source PDF |
| `page` | PDF page number, 1-based |
| `block_index` | reading order within the page |
| `category` | Text, Section-header, Footnote, Table, Picture, Caption, List-item, Page-header, Page-footer, Title, Formula |
| `bbox` | `[x0, y0, x1, y1]` in pixels at `--dpi` (default 150); divide by `dpi/72` for PDF points |
| `text` | Markdown; tables as HTML, formulas as LaTeX |

Pages with no blocks are absent.

## Slurm

Many GPUs over one corpus: [docs/SLURM.md](docs/SLURM.md).
