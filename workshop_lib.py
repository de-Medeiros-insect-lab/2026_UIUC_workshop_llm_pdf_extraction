"""Helpers for the 2026 UIUC workshop on extracting data from PDFs
with open-weight models. Imported by workshop.ipynb."""
from __future__ import annotations

import base64
import time
from pathlib import Path

import fitz
import ollama

DEFAULT_DPI = 100
"""100 dpi is calibrated: 72 dpi loses taxonomic terms, 150 dpi costs ~22%
more time for one extra term."""


def open_pdf(path: str | Path) -> fitz.Document:
    """Open a PDF. Pages are addressed 1-based everywhere in this module."""
    return fitz.open(str(path))


def _check_page(doc: fitz.Document, page: int) -> int:
    if not 1 <= page <= doc.page_count:
        raise ValueError(
            f"page {page} out of range; document has {doc.page_count} pages "
            f"(pages are 1-based)"
        )
    return page - 1


def get_page_text(doc: fitz.Document, page: int) -> str:
    """The PDF's embedded text layer for a page.

    Free and instant, and correct for born-digital PDFs.

    On a scanned document this returns whatever OCR the scanner ran, which
    may be silently corrupt: the 1929 example renders "Curculionidae" as
    "Cureulionidse". It does not come back empty, so there is nothing to
    check for -- compare it against ocr_page() to see the damage.
    """
    return doc[_check_page(doc, page)].get_text()


def render_page(doc: fitz.Document, page: int, dpi: int = DEFAULT_DPI) -> str:
    """Render a page to a base64-encoded PNG, for sending to a vision model."""
    pix = doc[_check_page(doc, page)].get_pixmap(dpi=dpi)
    return base64.b64encode(pix.tobytes("png")).decode()


# --- Text-quality gate -----------------------------------------------------
#
# Old scans often DO have an embedded text layer -- it's just the output of
# whatever OCR the scanner ran decades ago (or last year, on bad settings),
# and it can be silently wrong. On the 1929 plate used in this workshop,
# "Curculionidae" comes back as "Cureulionidse", "Elytra" as "Elylra",
# "Fig. 20" as "Fi.q. 20", "265" as "2~5", and "sub-linear" as "suh-linear".
# None of that raises a Python exception and `if not text:` sees nothing
# wrong -- the string is non-empty, plausible-looking prose. A pipeline that
# hands this straight to a model produces confident, fluent, wrong data at
# scale, and nothing downstream flags it.
#
# This gate has to be a dumb regex, not a judgment call by the model. Show a
# capable model this exact corrupt text and ask "does this look OK?" and it
# will notice the damage, call it "minor OCR artifacts", and then propose a
# plausible-sounding correction anyway -- which is exactly the failure mode
# we're trying to avoid, just moved one step earlier. A regex that counts
# suspicious tokens can't be talked into rationalising anything. It is also
# cheap and instant, so it's worth running on every page before deciding
# whether to burn time and compute on OCR.

CHAT_MODEL = "qwen3.5:9b"
OCR_MODEL = "deepseek-ocr"

OCR_PROMPT = (
    "Transcribe ALL of the text on this page exactly as printed, verbatim. "
    "Do not summarise, do not omit anything, do not paraphrase. "
    "Output only the transcription."
)


def server_ready(timeout: float = 60.0) -> bool:
    """Poll until the Ollama server answers, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ollama.list()
            return True
        except Exception:
            time.sleep(1.0)
    return False


def ocr_page(doc, page: int, dpi: int = DEFAULT_DPI) -> str:
    """Re-read a page from its image with the dedicated OCR model.

    The embedded text layer on scanned documents is often silently corrupt --
    for example, "Curculionidae" may render as "Cureulionidse", or "Elytra"
    as "Elylra". A vision model applied to the image can recover the correct
    text where the text layer failed. The deepseek-ocr model is a lightweight,
    purpose-built OCR engine that specializes in page transcription.

    think is not passed: deepseek-ocr is not a reasoning model. For the chat
    model, transcription-style work MUST pass think=False -- on defaults it
    emits six figures of reasoning and returns no content at all.
    """
    image = render_page(doc, page, dpi=dpi)
    reply = ollama.chat(
        model=OCR_MODEL,
        messages=[{"role": "user", "content": OCR_PROMPT, "images": [image]}],
        options={"temperature": 0, "num_predict": 8192},
    )
    return reply.message.content or ""
