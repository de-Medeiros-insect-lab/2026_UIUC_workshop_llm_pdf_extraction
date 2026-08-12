"""Helpers for the 2026 UIUC workshop on extracting data from PDFs
with open-weight models. Imported by workshop.ipynb."""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import List, Optional

import fitz
import ollama
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

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


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": (
                "Return the PDF's embedded text layer for a page. Free and "
                "instant, but on scanned documents it may be poor-quality OCR "
                "with garbled words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer",
                             "description": "1-based page number"}
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ocr_page",
            "description": (
                "Re-read a page from its image with a dedicated OCR model. "
                "Slower but far more accurate on scans."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer",
                             "description": "1-based page number"}
                },
                "required": ["page"],
            },
        },
    },
]


def run_tool_loop(messages, tools, impls, model: str = CHAT_MODEL,
                  max_turns: int = 6, think: bool = True, chat=None):
    """Drive a multi-turn tool conversation.

    Returns (final_text, calls_made). `chat` is injectable for testing.

    think defaults to True and should stay True. Deciding whether a page's text
    is trustworthy is judgement work: with think=False this model reads the
    corrupt text, talks itself out of the problem, and answers anyway. With
    reasoning on it re-reads the page via ocr_page and gets it right.
    """
    chat = chat or ollama.chat
    messages = list(messages)
    calls_made: list[tuple[str, dict]] = []

    for _ in range(max_turns):
        reply = chat(model=model, messages=messages, tools=tools,
                     think=think, options={"temperature": 0})
        msg = reply.message
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": msg.tool_calls or []})

        if not msg.tool_calls:
            return (msg.content or ""), calls_made

        for call in msg.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            calls_made.append((name, args))
            impl = impls.get(name)
            if impl is None:
                result = f"ERROR: no such tool {name!r}"
            else:
                try:
                    result = impl(**args)
                except Exception as exc:            # feed errors back
                    result = f"ERROR: {exc}"
            messages.append({"role": "tool", "name": name,
                             "content": str(result)[:6000]})

    return (f"stopped: max_turns={max_turns} reached"), calls_made


MAX_PLAUSIBLE_MM = 300.0  # no described weevil is 30 cm long


class Trait(BaseModel):
    """One measured or described character.

    source_text is required on purpose: an extraction you cannot trace back to
    the page is not evidence. It is also the field that a paraphrasing model
    quietly destroys.
    """
    anatomical_part: str
    trait: str
    value: str
    units: Optional[str] = None
    source_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _plausible_measurement(self):
        """Cross-field, so it must be a model validator, not a field validator.

        `units` is declared after `value`, so during field validation of `value`
        the `units` value is not yet available in info.data -- a field_validator
        here would silently never fire.

        Currently only validates millimeters. This is a workshop scope limitation:
        real pipelines normalize units first. Centimeters, inches, and other units
        bypass this check entirely.
        """
        if self.units == "mm":
            try:
                number = float(self.value)
            except (TypeError, ValueError):
                return self
            if not 0 < number <= MAX_PLAUSIBLE_MM:
                raise ValueError(
                    f"{number} mm is not a plausible measurement "
                    f"(0 < x <= {MAX_PLAUSIBLE_MM})"
                )
        return self


class Species(BaseModel):
    """A species with its measured and described traits.

    The name field requires a binomial (genus + species). Failing on a bare genus
    is intentional: when a genus-only or family-level identification appears, it
    usually means the model failed to extract the actual binomial name. Accepting
    and quietly recording the incomplete name would hide that failure. Loud failure
    makes the problem visible.
    """
    name: str
    traits: List[Trait] = []

    @field_validator("name")
    @classmethod
    def _binomial(cls, v):
        """Validate that name contains at least genus and species.

        Rejects names with fewer than two words (genus-only, family names, etc).
        """
        if len(v.split()) < 2:
            raise ValueError(f"{v!r} is not a binomial (need genus + species)")
        return v


class Extraction(BaseModel):
    """The output of structured extraction: all species and their traits found
    in a text.

    Enforced by pydantic model validation: each Species must have a binomial
    name, each Trait must have source_text, and any measurement in millimeters
    must be plausible (0 < x <= 300).
    """
    species: List[Species] = []


EXTRACT_PROMPT = (
    "Extract every species described in this text, with their morphological "
    "traits. Copy the exact source sentence for each trait into source_text. "
    "Do not invent traits that are not stated.\n\n"
)


def extract(text: str, model: str = CHAT_MODEL, chat=None) -> Extraction:
    """Structured extraction, enforced by JSON-schema-constrained decoding.

    The format parameter enforces the response to match Extraction.model_json_schema(),
    making this the whole point of the section: replacing the 2025 approach of
    asking nicely for JSON and repairing it afterwards.

    think=False disables chain-of-thought reasoning. For extraction tasks, this
    model produces six figures of thinking and returns no content when enabled,
    so we disable it for one-shot extraction. (Compare ocr_page and run_tool_loop
    which keep think=True because they perform judgement: does this text layer
    need correction? Should I escalate to OCR?)

    chat is injectable for testing without a live model. If not provided, uses
    the real ollama.chat client.
    """
    chat = chat or ollama.chat
    reply = chat(
        model=model,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text}],
        format=Extraction.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    return Extraction.model_validate_json(reply.message.content)


def to_dataframe(extraction: Extraction) -> pd.DataFrame:
    """Flatten an Extraction to one row per trait, ready for analysis.

    Each row represents a single trait measurement or description, with columns:
    species, anatomical_part, trait, value, units, source_text.

    An empty Extraction (no species, or species with no traits) returns a
    DataFrame with zero rows but all six named columns, preserving schema
    consistency downstream.
    """
    rows = [
        {"species": sp.name, "anatomical_part": tr.anatomical_part,
         "trait": tr.trait, "value": tr.value, "units": tr.units,
         "source_text": tr.source_text}
        for sp in extraction.species for tr in sp.traits
    ]
    return pd.DataFrame(
        rows,
        columns=["species", "anatomical_part", "trait", "value", "units",
                 "source_text"],
    )
