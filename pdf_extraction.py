"""Structured data out of a folder of PDFs, in one call.

    from pdf_extraction import extract_folder

    df = extract_folder("example_pdfs", prompt=MY_PROMPT, schema=MY_SCHEMA)

That is the whole of this workshop's pipeline, packaged. For each PDF in the
folder it decides, page by page, whether the embedded text layer can be
trusted or whether the page has to be re-read with an OCR model; it pulls out
the figures; it asks a language model for exactly the fields your schema
describes; and it returns one table.

`workshop.ipynb` builds every piece of this from scratch, in the open. This
module is the finished article, so that the first thing you see on the day is
a result rather than a pile of parts.

Requires an Ollama server on this machine, with both models pulled:

    ollama pull qwen3.5:9b
    ollama pull deepseek-ocr
"""
import base64
import glob
import io
import json
import os
import re

import ollama
import pandas as pd
import pymupdf
from PIL import Image

CHAT_MODEL = "qwen3.5:9b"      # reads text and images, follows a schema
OCR_MODEL = "deepseek-ocr"     # transcribes a page image, with layout

DEFAULT_DPI = 100              # enough to read fine print, cheap to send
SCAN_COVERAGE = 0.90           # an image covering this much of a page IS the page
MIN_FIGURE_AREA = 0.03         # smaller than this is a logo, not a figure

# Defaults sized for a 16 GB GPU. Raise all three together if you have more.
MAX_CHARS = 30_000             # of text per document
MAX_FIGURES = 8                # images per document
NUM_CTX = 16_384               # context window to ask Ollama for


# ---------------------------------------------------------------- talking to Ollama

def _chat(messages, schema, model, client, num_ctx):
    chat = (client or ollama).chat
    try:
        reply = chat(model=model, messages=messages, format=schema, think=False,
                     options={"temperature": 0, "num_ctx": num_ctx})
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach model {model!r}. Is the Ollama server running, "
            f"and have you run `ollama pull {model}`?") from exc
    return json.loads(reply.message.content)


def ocr_page(doc, page, dpi=DEFAULT_DPI):
    """Transcribe one page from its image, with region coordinates."""
    try:
        reply = ollama.generate(
            model=OCR_MODEL,
            prompt="<image>\n<|grounding|>Convert the document to markdown.",
            images=[render_page(doc, page, dpi=dpi)],
            options={"temperature": 0, "num_predict": 4096},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach model {OCR_MODEL!r}. Is the Ollama server "
            f"running, and have you run `ollama pull {OCR_MODEL}`?") from exc
    return reply.response or ""


# ------------------------------------------------------------------------- pages

def open_pdf(path):
    return pymupdf.open(path)


def page_text(doc, page):
    """The text layer already stored inside the PDF. Free and instant."""
    return doc[page - 1].get_text()


def render_page(doc, page, dpi=DEFAULT_DPI):
    """A page as an image, encoded for sending to a model."""
    return base64.b64encode(
        doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")).decode()


def page_is_scan(doc, page, coverage=SCAN_COVERAGE):
    """True if this page is a photograph of a page rather than real text.

    A scanned page holds one image object covering the whole page. Its text
    layer, if it has one at all, is whatever OCR ran years ago -- often wrong,
    and wrong without saying so, which is why we re-read those pages.
    """
    pg = doc[page - 1]
    page_area = pg.rect.width * pg.rect.height
    for xref, *_ in pg.get_images(full=True):
        placed = pg.get_image_rects(xref)
        if placed and max(r.width * r.height for r in placed) / page_area >= coverage:
            return True
    return False


# ----------------------------------------------------------------------- figures

def regions(ocr_text):
    """Every labelled region the OCR model reported, scaled 0-1000."""
    found = re.findall(r"(\w+)\s*\[\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]\]",
                       ocr_text)
    return [(label, *map(int, box)) for label, *box in found]


def figure_boxes(ocr_text):
    """One box per figure, each grown to take in its caption."""
    regs = regions(ocr_text)
    images = [r for r in regs if r[0] == "image"]
    captions = [r for r in regs if r[0] == "image_caption"]
    boxes = []
    for img in images:
        group = [img] + [c for c in captions
                         if c[1] < img[3] and c[3] > img[1]   # overlaps sideways
                         and abs(c[2] - img[4]) < 100]        # sits just below
        boxes.append(("figure",
                      min(g[1] for g in group), min(g[2] for g in group),
                      max(g[3] for g in group), max(g[4] for g in group)))
    return boxes


def crop_region(doc, page, box, dpi=150, pad=0.01):
    """Cut one 0-1000 box out of a rendered page."""
    im = Image.open(io.BytesIO(doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")))
    width, height = im.size
    _, x1, y1, x2, y2 = box
    return im.crop((int((x1 / 1000 - pad) * width), int((y1 / 1000 - pad) * height),
                    int((x2 / 1000 + pad) * width), int((y2 / 1000 + pad) * height)))


def figures_from_objects(doc, page):
    """Figures a born-digital PDF stores as objects, as PIL images."""
    pg = doc[page - 1]
    page_area = pg.rect.width * pg.rect.height
    found = []
    for xref, *_ in pg.get_images(full=True):
        placed = pg.get_image_rects(xref)
        share = max((r.width * r.height) / page_area for r in placed) if placed else 0
        if not MIN_FIGURE_AREA <= share < SCAN_COVERAGE:
            continue
        pix = pymupdf.Pixmap(doc, xref)
        if pix.colorspace is None:      # a stencil mask, not a picture
            continue
        found.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    return found


def as_image_data(fig, max_side=1024):
    """A figure, shrunk if it is huge, encoded the way a model wants it."""
    small = fig.copy()
    small.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------- documents

def read_document(doc, max_pages=None, dpi=DEFAULT_DPI, figures=True, log=None):
    """A whole PDF as (text, figures), each page read the way it needs to be."""
    last = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
    chunks, found = [], []
    for page in range(1, last + 1):
        if page_is_scan(doc, page):
            transcription = ocr_page(doc, page, dpi=dpi)
            chunks.append(f"--- page {page} ---\n{transcription}")
            if figures:
                # We already paid for the OCR, so reuse it to place the figures.
                found += [crop_region(doc, page, box)
                          for box in figure_boxes(transcription)]
            if log:
                log(f"    page {page}: scanned, re-read with {OCR_MODEL}")
        else:
            chunks.append(f"--- page {page} ---\n{page_text(doc, page)}")
            if figures:
                found += figures_from_objects(doc, page)
    return "\n\n".join(chunks), found


def record_key(schema):
    """The name of the array of records in a schema, if it has one.

    A schema for a whole document wraps its records in one named array --
    "species", "records", whatever you called it. We need that name to turn
    the answer into rows.
    """
    for name, spec in (schema.get("properties") or {}).items():
        if spec.get("type") == "array" and spec.get("items", {}).get("type") == "object":
            return name
    return None


def extract_document(path, prompt, schema, model=CHAT_MODEL, client=None,
                     figures=True, max_pages=None, max_chars=MAX_CHARS,
                     max_figures=MAX_FIGURES, num_ctx=NUM_CTX, log=None):
    """One PDF in, a list of records out."""
    doc = open_pdf(path)
    text, found = read_document(doc, max_pages=max_pages, figures=figures, log=log)

    if len(text) > max_chars and log:
        log(f"    text truncated to {max_chars} of {len(text)} characters "
            f"-- raise max_chars and num_ctx together to send it all")
    if len(found) > max_figures and log:
        log(f"    sending {max_figures} of {len(found)} figures "
            f"-- raise max_figures to send them all")

    message = {"role": "user", "content": prompt + text[:max_chars]}
    if found:
        message["images"] = [as_image_data(f) for f in found[:max_figures]]

    answer = _chat([message], schema, model, client, num_ctx)
    key = record_key(schema)
    records = answer.get(key, []) if key else [answer]
    return records if isinstance(records, list) else [records]


def extract_folder(folder, prompt, schema, model=CHAT_MODEL, client=None,
                   figures=True, max_pages=None, max_chars=MAX_CHARS,
                   max_figures=MAX_FIGURES, num_ctx=NUM_CTX, progress=True):
    """Every PDF in a folder, as one table.

    folder  -- a directory holding .pdf files
    prompt  -- what to extract, and what each field of your schema means
    schema  -- a JSON schema: the fields you want, wrapped in a named array

    Returns a DataFrame with one row per record and a `source` column saying
    which file it came from. A document that fails is reported and skipped,
    so one bad PDF does not cost you the rest of the run.
    """
    log = (lambda msg: print(msg, flush=True)) if progress else None
    paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"no PDFs in {folder!r}")

    frames = []
    for path in paths:
        name = os.path.basename(path)
        if log:
            log(f"{name}")
        try:
            records = extract_document(
                path, prompt, schema, model=model, client=client,
                figures=figures, max_pages=max_pages, max_chars=max_chars,
                max_figures=max_figures, num_ctx=num_ctx, log=log)
        except Exception as exc:
            if log:
                log(f"    FAILED -- {exc}")
            continue
        if log:
            log(f"    {len(records)} record(s)")
        if records:
            frame = pd.DataFrame(records)
            frame.insert(0, "source", name)
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
