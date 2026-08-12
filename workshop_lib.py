"""Helpers for the 2026 UIUC workshop on extracting data from PDFs
with open-weight models. Imported by workshop.ipynb."""
from __future__ import annotations

import base64
from pathlib import Path

import fitz

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

    Free and instant. On scanned documents this is the output of whatever OCR
    the scanner ran, which may be silently corrupt -- see looks_corrupt().
    """
    return doc[_check_page(doc, page)].get_text()


def render_page(doc: fitz.Document, page: int, dpi: int = DEFAULT_DPI) -> str:
    """Render a page to a base64-encoded PNG, for sending to a vision model."""
    pix = doc[_check_page(doc, page)].get_pixmap(dpi=dpi)
    return base64.b64encode(pix.tobytes("png")).decode()
