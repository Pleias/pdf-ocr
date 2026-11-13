# Adapted from dots.ocr, Copyright (c) 2025 rednote-hilab, under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions: The above copyright notice and this
# permission notice shall be included in all copies or substantial portions of
# the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.
"""
The DotsOCR inference layer: the prompt, PDF rasterisation, and one batched
vLLM call per group of page images.

The rendering path,
engine and sampling settings and prompt are the ones every reference corpus was
produced with; change any of them and you change the OCR. Sampling runs at
temperature 0.1 without a seed, so two runs over the same PDF agree closely but
not byte for byte.
"""
import io
import os
import tempfile
from typing import List

import fitz  # PyMuPDF
from PIL import Image

# The layout prompt dots.ocr was trained on. Do not edit: the model's output
# format (categories, bbox convention, tables as HTML) follows from it.
PROMPT_ALL = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object."""


def save_temp_image(image: Image.Image) -> str:
    """Write a page image to a temp JPEG (in $TMPDIR) and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.convert("RGB").save(tmp.name, "JPEG", quality=95)
        return tmp.name


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[str]:
    """Render every page at `dpi` and return temp image paths, in page order.

    Returns [] if the PDF cannot be opened or rendered. Pages are rendered to
    PNG, then stored as JPEG q95: that round trip is what the model has always
    been fed, so keep it.
    """
    img_paths: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img_paths.append(save_temp_image(img))
        doc.close()
    except Exception as e:
        print(f"Error converting PDF {pdf_path}: {e}")
        clean_temp_paths(img_paths)
        return []
    return img_paths


def clean_temp_paths(img_paths: List[str]) -> None:
    for p in img_paths:
        try:
            os.unlink(p)
        except OSError:
            pass


class PDFVisionProcessorVLLM:
    """Holds one vLLM engine and runs batches of page images through it."""

    def __init__(self, model_path: str, logger=None, gpu_memory_utilization: float = 0.95):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.log = (lambda msg: logger.info(msg)) if logger else print
        self.log("Loading vLLM engine...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=24000,
            tensor_parallel_size=1,
        )
        self.log("vLLM engine loaded")
        self.sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=24000)
        self.prompt = PROMPT_ALL
        # Only for the chat template.
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def process_images_batch(self, img_paths: List[str]) -> List[str]:
        """One generate() call over all images; returns raw model text per image."""
        prompts = []
        for img_path in img_paths:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": self.prompt},
            ]}]
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompts.append({"prompt": prompt_text,
                            "multi_modal_data": {"image": Image.open(img_path)}})
        self.log(f"  Processing {len(prompts)} pages in batch")
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]
