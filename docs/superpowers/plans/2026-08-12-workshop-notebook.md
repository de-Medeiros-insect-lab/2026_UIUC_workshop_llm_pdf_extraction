# 2026 UIUC Workshop Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a half-day Colab notebook that extracts structured data from
modern and 1929 taxonomic PDFs using open-weight models served by Ollama.

**Architecture:** A tested Python module (`workshop_lib.py`) holds the plumbing —
PDF access, Ollama calls, the OCR gate, the tool loop, the extraction schema. The
notebook imports it and displays the source of teaching-critical functions with
`inspect.getsource`, so students read and edit real code without the notebook
becoming an untestable wall of definitions. Everything runs on one Ollama backend
with two models.

**Tech Stack:** Python 3.12, `ollama` (client ≥0.32), PyMuPDF 1.28, pydantic 2.13,
pandas, pytest. Models: `qwen3.5:9b`, `deepseek-ocr`.

## Global Constraints

- **Ollama client and server must be ≥0.32.** `qwen3.5` and `gemma4` will not
  load on 0.13.2 — the pull fails with a bare "download the latest version".
- **Render pages at 100 dpi.** 72 dpi loses 4 of 20 taxonomic terms; 150 dpi
  costs ~22% more time for one extra term.
- **Always pass `think=False` for transcription and OCR calls.** On default
  settings `qwen3.5:9b` produced 123,055 characters of thinking, hit
  `done_reason='length'`, and returned zero content.
- **All page indices refer to the prepared 8-page
  `example_pdfs/Marshall1929_AnnMagNatHist.pdf`**, not the 9-page original.
  Printed p. 264 is PDF page 2. Page numbers in the public API are **1-based**.
- **Never claim the model escalates to OCR by itself.** It does not; this was
  measured. The gate is deterministic code.
- **Sanitization wording stays generic** in commit messages and committed docs.
- Repo: `de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction`, branch
  `main`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `workshop_lib.py` | All tested plumbing imported by the notebook |
| `tests/test_pdf.py` | Page text + rendering |
| `tests/test_gate.py` | The corruption gate |
| `tests/test_loop.py` | Tool-loop driver, with a fake model |
| `tests/test_schema.py` | Extraction schema + validators |
| `tests/conftest.py` | Shared fixtures; `ollama` marker |
| `workshop.ipynb` | The taught notebook, sections 0–9 |
| `README.md` | Colab badge, model list, timings |
| `environment.yml` | Local (non-Colab) environment |

`workshop_lib.py` is a single module by design: students clone the repo and read
one file. It stays under ~300 lines; if it grows past that, split by
responsibility (`pdf.py`, `models.py`, `loop.py`), not by layer.

---

### Task 1: Repo scaffolding

**Files:**
- Create: `environment.yml`, `README.md`, `tests/conftest.py`
- Modify: `.gitignore`
- Remove from tracking: `pdf_data_extraction.ipynb`

**Interfaces:**
- Consumes: nothing
- Produces: `pytest` marker `ollama` for tests needing a live server; repo layout

- [ ] **Step 1: Untrack last year's notebook**

It stays on disk as reference but must not ship — a student could open it and
hit the AWS key section. It is preserved in the 2025 repo.

```bash
cd /Users/bruno/Documents/docs_macbookair2015/teaching/2026_UIC_workshop
git rm --cached pdf_data_extraction.ipynb
printf 'pdf_data_extraction.ipynb\n' >> .gitignore
```

- [ ] **Step 2: Write `environment.yml`**

```yaml
name: uic_workshop_2026
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pydantic>=2.13
  - pandas
  - pymupdf>=1.28
  - pillow
  - ipykernel
  - jupyter
  - requests
  - pytest
  - pip
  - pip:
      - ollama>=0.6
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
import pytest, fitz, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
LEGACY = REPO / "example_pdfs" / "Marshall1929_AnnMagNatHist.pdf"


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: needs a running Ollama server")


@pytest.fixture(scope="session")
def legacy_pdf():
    assert LEGACY.exists(), f"missing {LEGACY}"
    return LEGACY


@pytest.fixture(scope="session")
def legacy_doc(legacy_pdf):
    return fitz.open(legacy_pdf)
```

- [ ] **Step 4: Verify the fixture sees the expected document**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/ --collect-only -q
```
Expected: collects 0 tests, exits without import errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Add repo scaffolding: environment, test config, gitignore"
```

---

### Task 2: PDF access

**Files:**
- Create: `workshop_lib.py`
- Create: `tests/test_pdf.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `open_pdf(path: str | Path) -> fitz.Document`
  - `get_page_text(doc: fitz.Document, page: int) -> str` — 1-based
  - `render_page(doc: fitz.Document, page: int, dpi: int = 100) -> str` —
    1-based, returns base64 PNG
  - `DEFAULT_DPI: int = 100`

- [ ] **Step 1: Write the failing test**

`tests/test_pdf.py`:

```python
import base64
import pytest
from workshop_lib import open_pdf, get_page_text, render_page, DEFAULT_DPI


def test_default_dpi_is_100():
    assert DEFAULT_DPI == 100


def test_page_numbers_are_one_based(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    # PDF page 2 is printed page 264, the first content page
    assert "264" in get_page_text(doc, 2)[:40]


def test_page_five_is_the_corrupt_description(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    text = get_page_text(doc, 5)
    assert "Cureulionidse" in text      # the corrupt running header
    assert "Curculionidae" not in text  # the correct spelling is absent


def test_render_page_returns_base64_png(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    b64 = render_page(doc, 5, dpi=DEFAULT_DPI)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_dpi_changes_size(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    assert len(render_page(doc, 5, dpi=150)) > len(render_page(doc, 5, dpi=72))


def test_out_of_range_page_raises(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    with pytest.raises(ValueError):
        get_page_text(doc, 99)
    with pytest.raises(ValueError):
        get_page_text(doc, 0)
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_pdf.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'workshop_lib'`.

- [ ] **Step 3: Write the implementation**

`workshop_lib.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_pdf.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add workshop_lib.py tests/test_pdf.py
git commit -m "Add PDF page access and rendering helpers"
```

---

### Task 3: The corruption gate

This is the heart of Section 7 and the one piece the model cannot be trusted to
do. Keep it deterministic, cheap, and explainable.

**Files:**
- Modify: `workshop_lib.py`
- Create: `tests/test_gate.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `looks_corrupt(text: str, threshold: float = 0.02) -> bool`
  - `corruption_report(text: str) -> dict` with keys `ratio`, `suspect_words`,
    `n_words`

- [ ] **Step 1: Write the failing test**

`tests/test_gate.py`:

```python
from workshop_lib import looks_corrupt, corruption_report, open_pdf, get_page_text

CLEAN = (
    "Legs approximately equal in length; femora slender, sub-linear and not "
    "toothed, the hind pair not exceeding the apex of the elytra; tibiae "
    "compressed, curved at the base, sub-carinate dorsally."
)
DIRTY = (
    "new South American Cureulionidse. 267 Legs approximately equal in length; "
    "femora slender, suh-linear and not toothed; Elylra oblong; Fi.q. 20; 2~5"
)


def test_clean_text_passes():
    assert looks_corrupt(CLEAN) is False


def test_dirty_text_is_flagged():
    assert looks_corrupt(DIRTY) is True


def test_report_names_the_suspects():
    suspects = " ".join(corruption_report(DIRTY)["suspect_words"])
    assert "2~5" in suspects
    assert "Fi.q." in suspects or "Fi.q" in suspects


def test_report_counts_words():
    r = corruption_report(CLEAN)
    assert r["n_words"] > 20
    assert r["ratio"] == 0.0


def test_empty_text_is_corrupt():
    # a scan with no text layer at all must route to OCR
    assert looks_corrupt("") is True
    assert looks_corrupt("   \n  ") is True


def test_threshold_is_adjustable():
    # one bad token in a long clean passage passes at a loose threshold
    text = CLEAN + " 2~5"
    assert looks_corrupt(text, threshold=0.5) is False
    assert looks_corrupt(text, threshold=0.0) is True


def test_real_corrupt_page_is_flagged(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    assert looks_corrupt(get_page_text(doc, 5)) is True
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_gate.py -v
```
Expected: FAIL — `ImportError: cannot import name 'looks_corrupt'`.

- [ ] **Step 3: Write the implementation**

Append to `workshop_lib.py`:

```python
import re

# Letter patterns that essentially never occur in English or in Latin
# taxonomic names, but are common OCR damage.
_SUSPECT_PATTERNS = [
    r"[a-zA-Z]~",        # 2~5, Curculionid~e
    r"~[a-zA-Z]",
    r"\d~",
    r"[a-z]\.[a-z]",     # Fi.q.
    r"[a-zA-Z][&][a-zA-Z]",   # Cureulioni&e
    r"[a-zA-Z]{2}[0-9]{1,2}[a-zA-Z]{2}",  # digits embedded in words
    r"\b[bcdfghjklmnpqrstvwxz]{4,}\b",     # 4+ consonants, no vowel
]
_SUSPECT_RE = re.compile("|".join(_SUSPECT_PATTERNS))


def corruption_report(text: str) -> dict:
    """Describe how damaged a block of text looks.

    Deterministic on purpose. The model cannot be relied on to judge this:
    given corrupt text it will rationalise the damage and invent a correction.
    """
    words = text.split()
    suspects = [w for w in words if _SUSPECT_RE.search(w)]
    ratio = (len(suspects) / len(words)) if words else 1.0
    return {"ratio": ratio, "suspect_words": suspects, "n_words": len(words)}


def looks_corrupt(text: str, threshold: float = 0.02) -> bool:
    """True when a page's text layer should not be trusted.

    Empty or whitespace-only text counts as corrupt: a scan with no text layer
    must route to OCR just like a badly-OCR'd one.
    """
    if not text.strip():
        return True
    return corruption_report(text)["ratio"] > threshold
```

- [ ] **Step 4: Run the tests**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_gate.py -v
```
Expected: 7 passed. If `test_real_corrupt_page_is_flagged` fails, print
`corruption_report(get_page_text(doc, 5))` and widen `_SUSPECT_PATTERNS` until
the real page trips the gate — do not lower the threshold below 0.02, which
would flag clean pages.

- [ ] **Step 5: Commit**

```bash
git add workshop_lib.py tests/test_gate.py
git commit -m "Add deterministic text-quality gate"
```

---

### Task 4: Ollama helpers

**Files:**
- Modify: `workshop_lib.py`
- Create: `tests/test_ollama.py`

**Interfaces:**
- Consumes: `render_page`, `DEFAULT_DPI`
- Produces:
  - `CHAT_MODEL: str = "qwen3.5:9b"`, `OCR_MODEL: str = "deepseek-ocr"`
  - `server_ready(timeout: float = 60.0) -> bool`
  - `ocr_page(doc, page, dpi=DEFAULT_DPI) -> str`

- [ ] **Step 1: Write the failing test**

`tests/test_ollama.py`:

```python
import pytest
from workshop_lib import CHAT_MODEL, OCR_MODEL, server_ready, ocr_page, open_pdf


def test_model_names():
    assert CHAT_MODEL == "qwen3.5:9b"
    assert OCR_MODEL == "deepseek-ocr"


@pytest.mark.ollama
def test_server_ready():
    assert server_ready(timeout=5) is True


@pytest.mark.ollama
def test_ocr_recovers_what_the_text_layer_destroyed(legacy_pdf):
    """The point of the whole OCR section: the image beats the text layer."""
    doc = open_pdf(legacy_pdf)
    out = ocr_page(doc, 5)
    assert "Curculionid" in out       # correct spelling recovered
    assert "Cureulionidse" not in out  # the corrupt form is gone
    assert len(out) > 1500             # a full page, not a summary
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_ollama.py -v -m ""
```
Expected: FAIL — `ImportError: cannot import name 'CHAT_MODEL'`.

- [ ] **Step 3: Write the implementation**

Append to `workshop_lib.py`:

```python
import time
import ollama

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
```

- [ ] **Step 4: Run the tests**

```bash
/usr/local/bin/ollama serve > /tmp/ollama.log 2>&1 &
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_ollama.py -v -m ""
```
Expected: 3 passed. The OCR test takes ~15 s cold.

- [ ] **Step 5: Commit**

```bash
git add workshop_lib.py tests/test_ollama.py
git commit -m "Add Ollama server check and OCR helper"
```

---

### Task 5: The tool loop

**Files:**
- Modify: `workshop_lib.py`
- Create: `tests/test_loop.py`

**Interfaces:**
- Consumes: `CHAT_MODEL`
- Produces:
  - `TOOL_SCHEMAS: list[dict]`
  - `run_tool_loop(messages, tools, impls, model=CHAT_MODEL, max_turns=6, chat=None) -> tuple[str, list[tuple[str, dict]]]`
    returning `(final_text, calls_made)`. `chat` is injectable so the loop is
    testable without a model.

- [ ] **Step 1: Write the failing test**

`tests/test_loop.py`. The fake `chat` lets us test loop mechanics
deterministically — no model, no GPU, no flakiness.

```python
import pytest
from types import SimpleNamespace
from workshop_lib import run_tool_loop, TOOL_SCHEMAS


def _reply(content="", tool_calls=None):
    calls = [
        SimpleNamespace(function=SimpleNamespace(name=n, arguments=a))
        for n, a in (tool_calls or [])
    ]
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=calls or None)
    )


def test_returns_text_when_no_tools_requested():
    fake = lambda **kw: _reply(content="done")
    text, calls = run_tool_loop([{"role": "user", "content": "hi"}], [], {},
                                chat=fake)
    assert text == "done"
    assert calls == []


def test_executes_a_tool_then_finishes():
    replies = iter([
        _reply(tool_calls=[("get_page_text", {"page": 5})]),
        _reply(content="the answer"),
    ])
    fake = lambda **kw: next(replies)
    impls = {"get_page_text": lambda page: f"text of {page}"}
    text, calls = run_tool_loop([{"role": "user", "content": "go"}],
                                TOOL_SCHEMAS, impls, chat=fake)
    assert text == "the answer"
    assert calls == [("get_page_text", {"page": 5})]


def test_tool_errors_are_fed_back_not_raised():
    replies = iter([
        _reply(tool_calls=[("ocr_page", {"page": 99})]),
        _reply(content="recovered"),
    ])
    fake = lambda **kw: next(replies)

    def boom(page):
        raise ValueError("page 99 out of range")

    text, calls = run_tool_loop([{"role": "user", "content": "go"}],
                                TOOL_SCHEMAS, {"ocr_page": boom}, chat=fake)
    assert text == "recovered"
    assert calls == [("ocr_page", {"page": 99})]


def test_unknown_tool_is_reported_back():
    replies = iter([
        _reply(tool_calls=[("no_such_tool", {})]),
        _reply(content="ok"),
    ])
    fake = lambda **kw: next(replies)
    text, _ = run_tool_loop([{"role": "user", "content": "go"}], [], {},
                            chat=fake)
    assert text == "ok"


def test_max_turns_is_enforced():
    fake = lambda **kw: _reply(tool_calls=[("get_page_text", {"page": 1})])
    impls = {"get_page_text": lambda page: "x"}
    text, calls = run_tool_loop([{"role": "user", "content": "go"}],
                                TOOL_SCHEMAS, impls, max_turns=3, chat=fake)
    assert len(calls) == 3
    assert "max_turns" in text


def test_tool_schemas_describe_both_tools():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"get_page_text", "ocr_page"}
    ocr = next(t for t in TOOL_SCHEMAS
               if t["function"]["name"] == "ocr_page")
    assert "page" in ocr["function"]["parameters"]["properties"]
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_loop.py -v
```
Expected: FAIL — `ImportError: cannot import name 'run_tool_loop'`.

- [ ] **Step 3: Write the implementation**

Append to `workshop_lib.py`:

```python
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
                  max_turns: int = 6, chat=None):
    """Drive a multi-turn tool conversation.

    Returns (final_text, calls_made). `chat` is injectable for testing.

    Note: the model decides *which* tool to call, but do not rely on it to
    judge whether text is good enough -- it will not escalate to OCR on its
    own. Gate that in code with looks_corrupt().
    """
    chat = chat or ollama.chat
    messages = list(messages)
    calls_made: list[tuple[str, dict]] = []

    for _ in range(max_turns):
        reply = chat(model=model, messages=messages, tools=tools,
                     think=False, options={"temperature": 0})
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
```

- [ ] **Step 4: Run the tests**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_loop.py -v
```
Expected: 6 passed, in under a second — no model involved.

- [ ] **Step 5: Commit**

```bash
git add workshop_lib.py tests/test_loop.py
git commit -m "Add tool-loop driver with injectable chat for testing"
```

---

### Task 6: Extraction schema and validators

**Files:**
- Modify: `workshop_lib.py`
- Create: `tests/test_schema.py`

**Interfaces:**
- Consumes: `CHAT_MODEL`
- Produces:
  - `Trait`, `Species`, `Extraction` (pydantic models)
  - `extract(text: str, model=CHAT_MODEL, chat=None) -> Extraction`
  - `to_dataframe(extraction: Extraction) -> pandas.DataFrame`

- [ ] **Step 1: Write the failing test**

`tests/test_schema.py`:

```python
import pytest
from pydantic import ValidationError
from workshop_lib import Trait, Species, Extraction, to_dataframe


def test_trait_requires_the_source_sentence():
    with pytest.raises(ValidationError):
        Trait(anatomical_part="elytra", trait="length")


def test_valid_trait():
    t = Trait(anatomical_part="elytra", trait="length", value="2.1",
              units="mm", source_text="Elytra 2.1 mm long")
    assert t.units == "mm"


def test_absurd_measurement_is_rejected():
    """Syntactically valid, semantically impossible -- the real failure mode."""
    with pytest.raises(ValidationError):
        Trait(anatomical_part="body", trait="length", value="5000",
              units="mm", source_text="body 5000 mm long")


def test_non_numeric_value_allowed_when_unitless():
    t = Trait(anatomical_part="pronotum", trait="colour", value="dull black",
              units=None, source_text="Colour dull black")
    assert t.value == "dull black"


def test_species_name_must_look_binomial():
    with pytest.raises(ValidationError):
        Species(name="Huarucus", traits=[])
    ok = Species(name="Huarucus cacti", traits=[])
    assert ok.name.startswith("Huarucus")


def test_to_dataframe_flattens_one_row_per_trait():
    ex = Extraction(species=[
        Species(name="Huarucus cacti", traits=[
            Trait(anatomical_part="elytra", trait="length", value="2.1",
                  units="mm", source_text="a"),
            Trait(anatomical_part="rostrum", trait="shape", value="curved",
                  units=None, source_text="b"),
        ]),
    ])
    df = to_dataframe(ex)
    assert len(df) == 2
    assert list(df.columns[:2]) == ["species", "anatomical_part"]
    assert set(df["species"]) == {"Huarucus cacti"}
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_schema.py -v
```
Expected: FAIL — `ImportError: cannot import name 'Trait'`.

- [ ] **Step 3: Write the implementation**

Append to `workshop_lib.py`:

```python
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

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
    name: str
    traits: List[Trait] = []

    @field_validator("name")
    @classmethod
    def _binomial(cls, v):
        if len(v.split()) < 2:
            raise ValueError(f"{v!r} is not a binomial (need genus + species)")
        return v


class Extraction(BaseModel):
    species: List[Species] = []


EXTRACT_PROMPT = (
    "Extract every species described in this text, with their morphological "
    "traits. Copy the exact source sentence for each trait into source_text. "
    "Do not invent traits that are not stated.\n\n"
)


def extract(text: str, model: str = CHAT_MODEL, chat=None) -> Extraction:
    """Structured extraction, enforced by JSON-schema-constrained decoding."""
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
    """One row per trait, ready for analysis."""
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
```

- [ ] **Step 4: Run the tests**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_schema.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Run the whole suite**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/ -v -m "not ollama"
```
Expected: all non-Ollama tests pass.

- [ ] **Step 6: Commit**

```bash
git add workshop_lib.py tests/test_schema.py
git commit -m "Add extraction schema with plausibility validators"
```

---

### Task 7: End-to-end pipeline check

Proves the gate actually changes the outcome — the claim Section 7 rests on.

**Files:**
- Modify: `workshop_lib.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above
- Produces: `page_text_trusted(doc, page, dpi=DEFAULT_DPI) -> tuple[str, str]`
  returning `(text, source)` where `source` is `"text_layer"` or `"ocr"`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:

```python
import pytest
from workshop_lib import page_text_trusted, open_pdf


@pytest.mark.ollama
def test_corrupt_page_is_routed_to_ocr_and_repaired(legacy_pdf):
    doc = open_pdf(legacy_pdf)
    text, source = page_text_trusted(doc, 5)
    assert source == "ocr"
    assert "Curculionid" in text
    assert "Cureulionidse" not in text


@pytest.mark.ollama
def test_clean_page_uses_the_free_path(legacy_pdf):
    """Page 1 is the born-digital publisher cover: no OCR needed."""
    doc = open_pdf(legacy_pdf)
    _, source = page_text_trusted(doc, 1)
    assert source == "text_layer"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_pipeline.py -v -m ""
```
Expected: FAIL — `ImportError: cannot import name 'page_text_trusted'`.

- [ ] **Step 3: Write the implementation**

Append to `workshop_lib.py`:

```python
def page_text_trusted(doc, page: int, dpi: int = DEFAULT_DPI):
    """Return text you can trust, plus where it came from.

    The decision is deterministic and lives here, in code, rather than in the
    model's judgement -- measured behaviour is that the model notices the
    corruption, talks itself out of it, and invents a correction.
    """
    text = get_page_text(doc, page)
    if looks_corrupt(text):
        return ocr_page(doc, page, dpi=dpi), "ocr"
    return text, "text_layer"
```

- [ ] **Step 4: Run the tests**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/test_pipeline.py -v -m ""
```
Expected: 2 passed. If page 1 trips the gate, inspect
`corruption_report(get_page_text(doc, 1))` — the publisher cover contains URLs,
which may need excluding from the suspect patterns.

- [ ] **Step 5: Commit**

```bash
git add workshop_lib.py tests/test_pipeline.py
git commit -m "Add trusted-text routing and end-to-end pipeline test"
```

---

### Task 8: Notebook sections 0–5

**Files:**
- Create: `workshop.ipynb`

**Interfaces:**
- Consumes: all of `workshop_lib`
- Produces: notebook cells for sections 0–5

- [ ] **Step 1: Create the notebook with the Colab setup cell**

Cell 1 (code) must work on a fresh Colab VM. `ollama serve` runs detached or
the cell never returns.

```python
# Colab setup. On a fresh VM this takes ~2 minutes; keep reading while it runs.
import os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    !curl -fsSL https://ollama.com/install.sh | sh
    !pip -q install ollama pymupdf pydantic pandas
    !git clone -q https://github.com/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction.git
    os.chdir("2026_UIUC_workshop_llm_pdf_extraction")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from workshop_lib import server_ready, CHAT_MODEL, OCR_MODEL
assert server_ready(120), "Ollama did not start"
print("Ollama is up")
```

- [ ] **Step 2: Add the GPU check cell**

Students must learn immediately, not in section 4, that they need to pair up.

```python
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True)
if gpu.returncode == 0:
    print("GPU:", gpu.stdout.strip())
else:
    print("NO GPU on this runtime.")
    print("Colab often refuses free GPUs at busy times of day.")
    print("Please pair up with a neighbour who has one -- the models are far")
    print("too slow on CPU for a workshop.")
```

- [ ] **Step 3: Add the model pull cell**

```python
# ~14 GB total. Start this now; we will talk while it downloads.
!ollama pull {CHAT_MODEL}
!ollama pull {OCR_MODEL}
!ollama list
```

- [ ] **Step 4: Add sections 2 and 5 (messages, system prompt, thinking)**

Section 2 markdown must keep the role-play framing and cite Shanahan, M.,
McDonell, K. & Reynolds, L. *Role play with large language models.* Nature 623,
493–498 (2023). Code cell:

```python
import ollama

reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "system",
         "content": "You are an expert coleopterist. Answer in two sentences."},
        {"role": "user",
         "content": "What is a rostrum, and which beetles have one?"},
    ],
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
```

Section 5 (thinking) code cell, showing the reasoning separately:

```python
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": "A beetle is 4.2 mm long and 1.4 mm wide. "
                          "What is the length-to-width ratio?"}],
    think=True,
    options={"temperature": 0},
)
print("--- reasoning ---");  print(reply.message.thinking)
print("--- answer ---");     print(reply.message.content)
```

Add a markdown warning immediately after, stating that `think=True` on a
transcription task made this model emit 123,055 characters of reasoning and
return no answer at all, and that transcription calls must pass `think=False`.

**Hands-on 1** (end of section 2). Students write their own system prompt and
see how strongly it steers a 9B model — the role-play point from Shanahan et al.
Give them this template with the upper-case parts to replace:

```python
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "system", "content": "WRITE A ROLE FOR THE MODEL HERE"},
        {"role": "user",   "content": "ASK YOUR QUESTION HERE"},
    ],
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
```

Markdown prompt for discussion: run the same question with an empty system
prompt and with a detailed one. Small models need firmer steering than the
hosted models used in 2025 — the difference is much larger here.

- [ ] **Step 5: Add section 3 (PDFs are not text)**

```python
from workshop_lib import open_pdf, get_page_text, render_page
from IPython.display import Image, display
import base64

modern = open_pdf("example_pdfs/deMedeiros2013Zootaxa.pdf")
legacy = open_pdf("example_pdfs/Marshall1929_AnnMagNatHist.pdf")

print("MODERN, born-digital:")
print(repr(get_page_text(modern, 1)[:200]))
print("\nLEGACY, a 1929 scan -- note this is NOT empty:")
print(repr(get_page_text(legacy, 5)[:200]))
```

Then display the rendered page beside it so students see the mismatch:

```python
display(Image(data=base64.b64decode(render_page(legacy, 5, dpi=100))))
```

- [ ] **Step 6: Execute the notebook end-to-end locally**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/jupyter nbconvert \
  --to notebook --execute workshop.ipynb --output /tmp/out.ipynb \
  --ExecutePreprocessor.timeout=1800
```
Expected: completes with no exception. The Colab-only branch is skipped because
`IN_COLAB` is False.

- [ ] **Step 7: Commit**

```bash
git add workshop.ipynb
git commit -m "Add notebook sections 0-5"
```

---

### Task 9: Notebook sections 6–9

**Files:**
- Modify: `workshop.ipynb`

**Interfaces:**
- Consumes: `extract`, `to_dataframe`, `run_tool_loop`, `page_text_trusted`,
  `looks_corrupt`, `corruption_report`

- [ ] **Step 1: Add section 4 (OCR) with the side-by-side comparison**

```python
from workshop_lib import ocr_page, corruption_report

dirty = get_page_text(legacy, 5)
clean = ocr_page(legacy, 5)

for label, text in [("TEXT LAYER", dirty), ("DEEPSEEK-OCR", clean)]:
    print(f"--- {label} ---")
    print(" ".join(text.split())[:180])
    print()
```

Markdown after it must state the measured result: at 100 dpi both models
recovered 19 of 20 checked taxonomic terms against the text layer's 16, and
DeepSeek-OCR's prompt-token cost stays flat at 961 across 100 and 150 dpi while
the general model's grows — which is why a purpose-built OCR model exists.

- [ ] **Step 2: Add section 6 (structured output) and hands-on 2**

```python
from workshop_lib import extract, to_dataframe

result = extract(clean)
df = to_dataframe(result)
df.head(15)
```

Immediately before this, one markdown cell plus one code cell showing **the old
way**, so students recognise it in the wild and understand why it is gone. In
2025 this section spent four cells escalating a prompt — add an example,
strengthen the system prompt, add XML tags — and then repaired the output with
`json-repair`. Show the unconstrained version failing or drifting:

```python
# The 2025 approach: ask nicely and hope. Run it a couple of times.
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": "List two traits of this beetle as JSON:\n\n"
                          + clean[:1500]}],
    think=False, options={"temperature": 0},
)
print(reply.message.content[:400])
```

Markdown: with `format=` the schema is enforced during decoding, so
non-conforming output cannot be produced at all. `json-repair` is still worth
knowing for providers that lack schema support — but you no longer need it here.

Then show a validator firing, so students see the schema catching a real error:

```python
from workshop_lib import Trait
from pydantic import ValidationError

try:
    Trait(anatomical_part="body", trait="length", value="5000",
          units="mm", source_text="body 5000 mm long")
except ValidationError as e:
    print("Rejected, as it should be:\n", e)
```

- [ ] **Step 3: Add section 7, staged as the measured failure then the fix**

Stage 1 — give the model both tools and let it fail, live:

```python
from workshop_lib import run_tool_loop, TOOL_SCHEMAS, ocr_page

impls = {
    "get_page_text": lambda page: get_page_text(legacy, page),
    "ocr_page":      lambda page: ocr_page(legacy, page),
}
system = ("You extract data from scanned historical taxonomic literature. "
          "The embedded text layer is old OCR and is frequently corrupt. "
          "Never guess or silently repair a garbled word: if the text looks "
          "corrupt, call ocr_page and use its output instead.")

answer, calls = run_tool_loop(
    [{"role": "system", "content": system},
     {"role": "user", "content":
      "On page 5, what is the family name in the running header?"}],
    TOOL_SCHEMAS, impls)

print("tools the model chose:", calls)
print(answer)
```

Markdown after it: the model calls `get_page_text`, *notices* the corruption,
rationalises it as a minor artifact, and answers from the bad text anyway —
often inventing a spelling. This was reproduced across four different system
prompts. Prompt engineering does not fix it.

Stage 2 — put the decision in code:

```python
from workshop_lib import looks_corrupt, page_text_trusted

print("gate says corrupt?", looks_corrupt(get_page_text(legacy, 5)))
print(corruption_report(get_page_text(legacy, 5))["suspect_words"][:8])

text, source = page_text_trusted(legacy, 5)
print("source used:", source)
print(" ".join(text.split())[:160])
```

Markdown: the lesson is *where the decision lives*, not that agents are magic.
The model is good at choosing among tools and bad at judging whether its input
is trustworthy. Keep that judgement deterministic.

Hands-on 3: students tune `looks_corrupt`'s threshold and patterns against
pages 2–8 and report which pages route to OCR.

- [ ] **Step 4: Add section 8 (cloud) and section 9 (where to go)**

```python
# Same code, one different string. Requires a free ollama.com account;
# add OLLAMA_API_KEY to Colab Secrets first.
from google.colab import userdata
import os
os.environ["OLLAMA_API_KEY"] = userdata.get("OLLAMA_API_KEY")

CLOUD_MODEL = "qwen3.5:cloud"
answer, calls = run_tool_loop(
    [{"role": "system", "content": system},
     {"role": "user", "content":
      "On page 5, what is the family name in the running header?"}],
    TOOL_SCHEMAS, impls, model=CLOUD_MODEL)
print("tools the cloud model chose:", calls)
print(answer)
```

Markdown must pose the open question honestly: the 9B model never escalated to
OCR. Does the larger one? Run it and see.

Section 9's portability demo. Same server, same model, a different client — so
students see that this code is not locked to Ollama:

```python
from openai import OpenAI
from workshop_lib import Extraction, EXTRACT_PROMPT

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Tools and schemas port cleanly.
parsed = client.chat.completions.parse(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": EXTRACT_PROMPT + clean[:4000]}],
    response_format=Extraction,
    temperature=0,
).choices[0].message.parsed
print(parsed.species[0].name)
```

Then show precisely where the abstraction leaks — this was measured, not
assumed:

```python
reply = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content":
               "A beetle is 4.2 mm long and 1.4 mm wide. Ratio?"}],
    temperature=0)
msg = reply.choices[0].message
print("content:", msg.content)
print("reasoning_content present?", "reasoning_content" in (msg.model_extra or {}))
print("model_extra keys:", list((msg.model_extra or {}).keys()))
```

Markdown: tool calling and structured output port cleanly — `.parse()` even
handles the schema massaging for you. Reasoning does not: it arrives under a
bare `reasoning` key inside `model_extra`, which is neither OpenAI's convention
nor the `reasoning_content` used by DeepSeek and vLLM, and the typed SDK cannot
see it. That is what will break when you change providers next year.

Section 9 closes with reproducibility (pin the weights *and* the Ollama version
— these models will not load on 0.13.2), data sovereignty for unpublished
specimen records and sensitive localities, and zero marginal cost for iteration.

- [ ] **Step 5: Execute the notebook end-to-end**

```bash
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/jupyter nbconvert \
  --to notebook --execute workshop.ipynb --output /tmp/out.ipynb \
  --ExecutePreprocessor.timeout=3600
```
Expected: completes. The cloud cell will fail without a key — wrap it in
`try/except` and print an instruction rather than raising.

- [ ] **Step 6: Commit**

```bash
git add workshop.ipynb
git commit -m "Add notebook sections 6-9"
```

---

### Task 10: README, facilitator notes, Colab badge, and push

**Files:**
- Create: `README.md`, `FACILITATOR.md`

**Interfaces:**
- Consumes: everything
- Produces: the public entry point and the instructor's running sheet

- [ ] **Step 0: Write `FACILITATOR.md`**

The spec's pacing decisions live here, not in the notebook.

```markdown
# Facilitator notes

## Timing (~240 min)

| # | Section | min |
| --- | --- | --- |
| 0 | Why this changed since 2025 | 5 |
| 1 | Ollama on Colab | 20 |
| 2 | Messages + system prompt — hands-on 1 | 25 |
| 3 | PDFs are not text | 30 |
| 4 | OCR for legacy literature | 30 |
| 5 | Thinking | 15 |
| 6 | Structured output — hands-on 2 | 35 |
| 7 | Tool use & the agentic loop — capstone | 45 |
| 8 | Scaling up to a cloud model | 20 |
| 9 | Where to go from here | 15 |

## Pacing

Long-running cells are the main hazard. Structured extraction on a full paper
takes ~3 min. **Have students start the run, then talk over it** — do not let
the room sit in silence watching a spinner. The same applies to the ~14 GB model
pull in section 1: start it, then teach section 0 while it downloads.

## Before you begin

- Ask everyone to set Runtime → Change runtime type → T4 **before** running
  anything, and to raise a hand if they cannot get a GPU.
- Pair up anyone without a GPU immediately. Colab refuses free GPUs at busy
  times, and CPU is far too slow for the OCR and vision sections.

## Section 7 is a staged failure

The point of the capstone is that the model *fails* first. Do not fix it early.
Let students watch it read corrupt text, notice the corruption, talk itself out
of it, and invent a spelling. Then move the decision into code. The lesson is
what to delegate and what to keep deterministic.

## Live demo opportunity

Muse Glimmer (30B) will not run on free Colab — 18 GB against a T4's 16 GB. If
you have a 32 GB machine to hand, running it locally makes a good contrast for
section 8.
```

- [ ] **Step 1: Write `README.md`**

```markdown
# Extracting structured data from PDFs with open models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction/blob/main/workshop.ipynb)

Workshop materials, UIUC 2026. Everything runs free on Google Colab with
open-weight models — no API keys, no credit card.

**Click the badge above.** The first cell installs Ollama and clones this repo.

## Models

| model | role | size |
| --- | --- | --- |
| `qwen3.5:9b` | reasoning, extraction, tool use | 6.6 GB |
| `deepseek-ocr` | page transcription | 6.7 GB |

## What you need

A Google account, and a Colab runtime **with a GPU**
(Runtime → Change runtime type → T4). Colab sometimes refuses free GPUs at busy
times; if that happens, pair up with a neighbour.

## Running locally instead

    mamba env create -f environment.yml
    mamba activate uic_workshop_2026
    pytest tests/ -m "not ollama"

Requires Ollama ≥0.32 — earlier versions cannot load these models.

## The 2025 version

This workshop previously used Anthropic's Claude:
[2025_ESA_workshop_claude_pdfs](https://github.com/de-Medeiros-insect-lab/2025_ESA_workshop_claude_pdfs).
```

- [ ] **Step 2: Verify the full suite passes**

```bash
/usr/local/bin/ollama serve > /tmp/ollama.log 2>&1 &
/Users/bruno/miniforge3/envs/uic_workshop_2026/bin/python -m pytest tests/ -v -m ""
```
Expected: all pass.

- [ ] **Step 3: Commit and push**

```bash
git add README.md
git commit -m "Add README with Colab badge"
git push -u origin main
```

- [ ] **Step 4: Verify the badge resolves**

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://colab.research.google.com/github/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction/blob/main/workshop.ipynb"
```
Expected: 200.

---

## Post-implementation validation (before the workshop)

These need a real Colab runtime and cannot be done from this machine.

- [ ] Open the badge link on a **free-tier T4** and run every cell top to bottom.
- [ ] Record actual T4 timings; update the README table if they differ materially
      from the M1 Max figures in the spec.
- [ ] Confirm `qwen3.5:9b` and `deepseek-ocr` can both be used in one session on
      16 GB VRAM, and measure the swap penalty.
- [ ] Test whether `qwen3.5:cloud` escalates to `ocr_page` where the 9B did not.
      If it does, promote that to a headline moment in Section 8.
- [ ] Confirm ~20 students can each complete the Section 8 cloud call inside the
      free tier's limits; if not, demote it to an instructor demo.
