"""Helpers for the 2026 UIUC workshop on extracting data from PDFs
with open-weight models. Imported by workshop.ipynb."""
from __future__ import annotations

import base64
import re
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

# Publisher cover pages and reference lists are full of URLs, DOIs, and
# email addresses -- and those legitimately contain exactly the letter/digit/
# punctuation combinations the suspect patterns below look for: dots between
# letters ("dx.doi.org"), digits fused into a "word" ("tnah16",
# "00222932908673050"), long consonant runs ("fas.harvard.edu"). None of
# that is OCR damage; it's just what a URL looks like. Strip it out before
# scoring anything, on both the born-digital and the scanned PDFs, or a
# perfectly clean cover page gets flagged as corrupt for no reason other
# than containing a working link.
_URL_EMAIL_RE = re.compile(
    r"https?://\S+|www\.\S+|\b10\.\d{4,9}/\S+|\b[\w.+-]+@[\w.-]+\.\w+\b",
    re.IGNORECASE,
)


def _strip_urls_and_emails(text: str) -> str:
    """Remove URLs, bare DOIs, and email addresses before scoring.

    They are never evidence of OCR damage, only of the page being a cover
    sheet or reference list, so they must not count toward the corruption
    ratio in either direction.
    """
    return _URL_EMAIL_RE.sub(" ", text)


# Markers that are strong enough on their own to prove a page is corrupt,
# independent of the ratio. A tilde or an ampersand glued between two letters
# never occurs in ordinary English or Latin text (once URLs are stripped),
# so even a single occurrence is decisive -- unlike the softer patterns
# below, whose signal is only meaningful once several of them turn up in the
# same page. Checked separately from the ratio because real corrupt pages
# in this collection put their damage in just a handful of words out of
# several hundred: diluting a single "Cureulioni&e." across a 400-word page
# drops its ratio below the threshold and would wave the page through.
_HIGH_CONFIDENCE_PATTERNS = [
    r"[a-zA-Z]~",        # 2~5, Curculionid~e
    r"~[a-zA-Z]",
    r"\d~",
    r"[a-zA-Z][&][a-zA-Z]",   # Cureulioni&e
]
_HIGH_CONFIDENCE_RE = re.compile("|".join(_HIGH_CONFIDENCE_PATTERNS))

# The full set of suspect patterns, used for the ratio-based fallback and
# for corruption_report()'s "suspect_words" listing. Includes the
# high-confidence markers above (a tilde-word is also a suspect word for
# reporting purposes) plus softer signals that are only meaningful in
# aggregate: stray periods splitting a word, digits stitched into the
# middle of a word, unpronounceable runs of consonants where a scanner has
# mangled a whole word beyond recognition, the specific "-idae"/"-inae" ->
# "-idse"/"-inse" mangle that shows up constantly in insect family and
# subfamily names, stray bracket characters glued onto a word, and a letter
# running straight into a digit across a hyphenated line break.
#
# The first several of these came from just looking at known-bad text before
# ever touching real data. The rest were added only after running this gate
# against every page of both example PDFs (see tests/test_gate.py) and
# inspecting corruption_report() to see which damaged words the earlier
# patterns were missing, and which clean words or URLs were being flagged
# by mistake -- a reminder that a plausible-looking regex still has to be
# checked against real documents, not just against the examples you
# invented for it.
_SUSPECT_PATTERNS = _HIGH_CONFIDENCE_PATTERNS + [
    r"[a-z]\.[a-z]",     # Fi.q.
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

    URLs, DOIs, and email addresses are stripped before scoring (see
    _strip_urls_and_emails) so a clean cover page full of links doesn't
    read as corrupt.

    Returns a dict with:
      - "ratio": fraction of (post-stripping) words that match a suspect
        pattern (1.0 for empty input, since there's nothing to trust
        either way)
      - "suspect_words": the actual offending tokens, for showing students
        what tripped the gate
      - "n_words": how many whitespace-separated tokens were examined,
        after URLs/DOIs/emails were removed
    """
    text = _strip_urls_and_emails(text)
    words = text.split()
    suspects = [w for w in words if _SUSPECT_RE.search(w)]
    ratio = (len(suspects) / len(words)) if words else 1.0
    return {"ratio": ratio, "suspect_words": suspects, "n_words": len(words)}


def looks_corrupt(text: str, threshold: float = 0.02) -> bool:
    """True when a page's text layer should not be trusted and needs OCR.

    Empty or whitespace-only text counts as corrupt: a scan with no text
    layer at all must route to OCR exactly like a badly-OCR'd one -- both
    mean "don't trust what get_page_text() gave you."

    The decision is two-part, because a plain ratio test is the wrong shape
    for this data:

    1. If any high-confidence marker is present (a tilde or ampersand fused
       into a word, after stripping URLs/DOIs/emails), the page is corrupt
       regardless of ratio. Corruption in this collection is sparse but
       decisive: a single "Cureulioni&e." or "Ba~rDzY~e." on an otherwise
       ordinary-looking 400-word page is conclusive proof the text layer
       cannot be trusted, and a ratio test alone dilutes that one damaged
       word into insignificance.
    2. Otherwise, fall back to the ratio test at `threshold` (2% of words
       flagged by default). Clean scientific prose contains the occasional
       genuine oddity -- abbreviations, hyphenated compounds -- so the gate
       tolerates a little noise here, but don't lower the threshold just to
       make one stubborn page pass; that starts waving through pages that
       really are corrupt, which defeats the point of having a gate at all.
    """
    if not text.strip():
        return True
    if _HIGH_CONFIDENCE_RE.search(_strip_urls_and_emails(text)):
        return True
    return corruption_report(text)["ratio"] > threshold


# --- Ollama helpers --------------------------------------------------------

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
