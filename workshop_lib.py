"""Helpers for the 2026 UIUC workshop on extracting data from PDFs
with open-weight models. Imported by workshop.ipynb."""
from __future__ import annotations

import base64
import re
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

# Letter patterns that essentially never occur in English or in Latin
# taxonomic names, but are common OCR damage: tildes standing in for digits
# or letters, stray periods splitting a word, ampersands substituted for a
# letter, digits stitched into the middle of a word, unpronounceable runs of
# consonants where a scanner has mangled a whole word beyond recognition,
# the specific "-idae"/"-inae" -> "-idse"/"-inse" mangle that shows up
# constantly in insect family and subfamily names, stray bracket characters
# glued onto a word, and a letter running straight into a digit across a
# hyphenated line break.
#
# The first several of these came from just looking at known-bad text before
# ever touching real data. The last three were added only after running this
# gate against a real corrupt page (see tests/test_gate.py,
# test_real_corrupt_page_is_flagged) and inspecting corruption_report() to
# see which damaged words the original patterns were missing -- a reminder
# that a plausible-looking regex still has to be checked against real
# scans, not just against the examples you invented for it.
_SUSPECT_PATTERNS = [
    r"[a-zA-Z]~",        # 2~5, Curculionid~e
    r"~[a-zA-Z]",
    r"\d~",
    r"[a-z]\.[a-z]",     # Fi.q.
    r"[a-zA-Z][&][a-zA-Z]",   # Cureulioni&e
    r"[a-zA-Z]{2}[0-9]{1,2}[a-zA-Z]{1,}",  # digits embedded in words, e.g. Antenn6e
    r"\b[bcdfghjklmnpqrstvwxz]{4,}\b",     # 4+ consonants, no vowel
    r"i[dn]se\b",          # Curculionidae/Cryptorrhynchinae -> ...idse/...inse
    r"[\[\]]",             # a stray bracket fused into a word, e.g. [tenter
    r"[a-zA-Z]-{1,2}[0-9]",  # a letter running into a digit, e.g. l--4
]
_SUSPECT_RE = re.compile("|".join(_SUSPECT_PATTERNS))


def corruption_report(text: str) -> dict:
    """Describe how damaged a block of text looks.

    Deterministic on purpose. The model cannot be relied on to judge this:
    given corrupt text it will rationalise the damage and invent a
    correction, rather than reporting that the input can't be trusted.

    Returns a dict with:
      - "ratio": fraction of words that match a suspect pattern (1.0 for
        empty input, since there's nothing to trust either way)
      - "suspect_words": the actual offending tokens, for showing students
        what tripped the gate
      - "n_words": how many whitespace-separated tokens were examined
    """
    words = text.split()
    suspects = [w for w in words if _SUSPECT_RE.search(w)]
    ratio = (len(suspects) / len(words)) if words else 1.0
    return {"ratio": ratio, "suspect_words": suspects, "n_words": len(words)}


def looks_corrupt(text: str, threshold: float = 0.02) -> bool:
    """True when a page's text layer should not be trusted and needs OCR.

    Empty or whitespace-only text counts as corrupt: a scan with no text
    layer at all must route to OCR exactly like a badly-OCR'd one -- both
    mean "don't trust what get_page_text() gave you."

    The default threshold (2% of words flagged) is deliberately low. Clean
    scientific prose contains the occasional genuine oddity -- abbreviations,
    hyphenated compounds -- so the gate tolerates a little noise, but a truly
    damaged page clears this bar even so. Don't lower the threshold just to
    make one stubborn page pass; that starts waving through pages that
    really are corrupt, which defeats the point of having a gate at all.
    """
    if not text.strip():
        return True
    return corruption_report(text)["ratio"] > threshold
