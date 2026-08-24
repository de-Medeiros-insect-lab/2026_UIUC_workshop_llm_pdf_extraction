"""Structured data out of a folder of PDFs, in one call.

    from pdf_extraction import extract_folder

    df = extract_folder("example_pdfs", prompt=MY_PROMPT, schema=MY_SCHEMA)

For each PDF in the folder this asks a reasoning model one question first — can
this document's own text layer be trusted, or do the pages have to be re-read
from their images? — and then reads the document whichever way the answer says,
pulls out the figures, asks for exactly the fields your schema describes, and
returns one table.

The work is done in three passes — every judgement, then every transcription,
then every extraction — so that only one model is ever resident. Two models of
this size do not fit on a 16 GB GPU together, and it is the second one to load
that fails.

That first question is the whole point. A scanned paper usually *has* embedded
text, left behind by an OCR pass that ran years ago, and it is often wrong in
ways nothing in the file admits to. No rule about odd characters survives
contact with the next scanner, so the judgement goes to a model that can
reason, and it tells you what it decided and why.

`workshop.ipynb` builds every piece of this from scratch, in the open. This
module is the finished article, meant to be pointed at your own folders long
after the workshop.

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

CHAT_MODEL = "qwen3.5:9b"      # reads text and images, reasons, follows a schema
OCR_MODEL = "deepseek-ocr"     # transcribes a page image, with layout

DEFAULT_DPI = 100              # enough to read fine print, cheap to send
MIN_FIGURE_AREA = 0.03         # smaller than this is a logo, not a figure
MAX_FIGURE_AREA = 0.90         # bigger than this is a scan of the whole page

SAMPLE_PAGES = 3               # pages of text layer shown to the model to judge
SAMPLE_CHARS = 1_500           # per sampled page
MIN_TEXT = 100                 # less text than this and there is nothing to judge
STRATEGY_CTX = 8_192           # small input, but reasoning needs somewhere to go

# Sized for a 16 GB GPU. Raise max_chars and num_ctx together if you have more.
MAX_CHARS = 30_000             # of document text sent to the model
MAX_FIGURES = 8                # images sent per document
NUM_CTX = 16_384               # context window to ask Ollama for


# ---------------------------------------------------------------- talking to models

def _failed(model, exc):
    """The error the server actually gave, plus a guess at what to do."""
    detail = str(exc)
    low = detail.lower()
    if "memory" in low or "resource" in low:
        hint = ("\n  The GPU could not fit it. Another model is probably still "
                "resident -- lower num_ctx, or free it first.")
    elif "not found" in low or "no such" in low or "404" in low:
        hint = f"\n  Run `ollama pull {model}`."
    elif "connect" in low or "refused" in low:
        hint = "\n  The Ollama server is not answering. Is `ollama serve` running?"
    else:
        hint = ""
    return RuntimeError(f"{model} failed: {detail}{hint}")


def installed_models():
    """What the server has, or None if the server cannot be reached."""
    try:
        listing = (ollama.list() or {})
    except Exception:
        return None
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models", [])
    names = set()
    for entry in models or []:
        name = getattr(entry, "model", None)
        if name is None and isinstance(entry, dict):
            name = entry.get("model")
        if name:
            names.add(name)
            names.add(name.split(":")[0])   # deepseek-ocr as well as :latest
    return names


def require_models(*wanted):
    """Fail before doing any work if a model we are going to need is absent."""
    have = installed_models()
    if have is None:
        raise RuntimeError(
            "Cannot reach the Ollama server. Is `ollama serve` running?")
    missing = [n for n in wanted if n not in have and n.split(":")[0] not in have]
    if missing:
        raise RuntimeError(
            "Not installed: " + ", ".join(missing) + "\n  Run "
            + "; ".join(f"`ollama pull {n}`" for n in missing)
            + f"\n  Installed: {', '.join(sorted(n for n in have if ':' in n))}")


def unload(model, client=None):
    """Drop a model from memory now, so the next one has room to load.

    Best effort: if the server will not, we carry on and let the next call
    report whatever goes wrong.
    """
    try:
        (client or ollama).generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass


class NoAnswer(RuntimeError):
    """The model replied, but with nothing in the content to parse.

    Carries the reasoning it did produce, so the caller can hand it back.
    """

    def __init__(self, message, thinking="", done_reason=None):
        super().__init__(message)
        self.thinking = thinking or ""
        self.done_reason = done_reason


def _chat(messages, schema, model, client, num_ctx, think=False):
    chat = (client or ollama).chat
    try:
        reply = chat(model=model, messages=messages, format=schema, think=think,
                     options={"temperature": 0, "num_ctx": num_ctx})
    except Exception as exc:
        raise _failed(model, exc) from exc

    content = (reply.message.content or "").strip()
    if not content:
        why = getattr(reply, "done_reason", None)
        detail = f" (done_reason={why})" if why else ""
        if reply.message.thinking:
            detail += (f", after {len(reply.message.thinking)} characters of "
                       f"reasoning -- num_ctx={num_ctx} left no room for the answer")
        raise NoAnswer(f"{model} returned no answer{detail}",
                       thinking=reply.message.thinking, done_reason=why)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{model} did not return JSON: {content[:200]!r}") from exc


RESUME_PROMPT = (
    "You worked through this already and ran out of room before answering. "
    "Your reasoning is below, and it is finished -- it may stop mid-sentence, "
    "in which case draw your conclusion from as far as it got.\n\n"
    "<thinking>\n{thinking}\n</thinking>\n\n"
    "Do not reason any further. Give the answer now, in the required format, "
    "based on the reasoning above."
)


def _answer(messages, schema, model, client, num_ctx, think=False, log=None):
    """Ask until there is an answer, not just reasoning.

    Ollama cannot reserve output space for the answer -- there is no thinking
    budget (ollama/ollama#10925) -- so a model given think=True can reason all
    the way to the end of its context and come back with nothing at all.

    Three tries, each a different guess at why the last one gave nothing:

      1. **thinking**, as the caller asked for it.
      2. **inserted thinking** -- its own reasoning handed back as finished
         text, with the context doubled and thinking off. This assumes the
         reasoning was sound and there was simply no room left to answer, so
         it keeps the expensive part instead of redoing it.
      3. **no thinking** -- the original question, answered straight. This
         assumes the reasoning itself was the problem, so it drops it, and it
         goes back to the context size we started with rather than asking a
         16 GB GPU for one it cannot allocate.

    A caller that did not ask for thinking gets one try: there is no reasoning
    to re-use, and stage 3 would just repeat stage 1.
    """
    stalled = None
    try:
        return _chat(messages, schema, model, client, num_ctx, think=think)
    except NoAnswer as exc:
        if not think:
            raise
        stalled = exc
        if log:
            log(f"    {exc}")

    if stalled.thinking:
        if log:
            log(f"    handing its reasoning back and asking for the answer "
                f"alone, with num_ctx={num_ctx * 2}")
        resumed = list(messages) + [
            {"role": "user",
             "content": RESUME_PROMPT.format(thinking=stalled.thinking)}]
        try:
            return _chat(resumed, schema, model, client, num_ctx * 2, think=False)
        except NoAnswer as exc:
            if log:
                log(f"    {exc}")

    if log:
        log("    asking once more with no reasoning at all")
    return _chat(messages, schema, model, client, num_ctx, think=False)


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
        raise _failed(OCR_MODEL, exc) from exc
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


# --------------------------------------------------- letting the model choose

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_ocr": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["needs_ocr", "reason"],
    "additionalProperties": False,
}

STRATEGY_PROMPT = (
    "Below is the text layer stored inside a PDF, sampled from a few of its "
    "pages. Decide whether it can be trusted as a transcription of the "
    "document.\n\n"
    "Set needs_ocr to true if the text looks like the output of an old and "
    "poor OCR pass: garbled words, impossible spellings of ordinary or "
    "taxonomic names, letters swapped for punctuation, words run together, "
    "lines missing. Those pages should be re-read from their images instead. "
    "Set needs_ocr to false if the text reads cleanly, the way text stored "
    "directly in a PDF does.\n\n"
    "Explain yourself in reason, in one sentence, quoting the words that "
    "decided it.\n\n"
)


def sample_pages(page_count, how_many=SAMPLE_PAGES):
    """A few page numbers spread through a document."""
    if page_count <= how_many:
        return list(range(1, page_count + 1))
    step = page_count / (how_many + 1)
    return sorted({min(page_count, max(1, round(step * (i + 1))))
                   for i in range(how_many)})


def choose_strategy(doc, model=CHAT_MODEL, client=None,
                    num_ctx=STRATEGY_CTX, log=None):
    """Ask a reasoning model how this document should be read.

    Returns {"needs_ocr": bool, "reason": str}. If the model cannot be reached
    we re-read the pages: slower, but never quietly wrong.
    """
    sample = "\n\n".join(
        f"--- page {p} ---\n{page_text(doc, p)[:SAMPLE_CHARS]}"
        for p in sample_pages(doc.page_count))

    if len(sample.strip()) < MIN_TEXT:
        return {"needs_ocr": True,
                "reason": "the PDF has no usable text layer at all"}
    messages = [{"role": "user", "content": STRATEGY_PROMPT + sample}]
    try:
        return _answer(messages, STRATEGY_SCHEMA, model, client, num_ctx,
                       think=True, log=log)
    except Exception as exc:
        if log:
            log(f"    could not get a judgement ({exc})")
            log("    re-reading the pages to be safe")
        return {"needs_ocr": True, "reason": "could not get a judgement"}


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
        if not MIN_FIGURE_AREA <= share <= MAX_FIGURE_AREA:
            continue
        # extract_image gives the bytes as stored, whatever the colour depth:
        # line art at 1 bit per pixel comes back just as happily as a photo.
        found.append(Image.open(io.BytesIO(doc.extract_image(xref)["image"])))
    return found


def as_image_data(fig, max_side=1024):
    """A figure, shrunk if it is huge, encoded the way a model wants it."""
    small = fig.copy()
    small.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------- documents

def transcribe(doc, max_pages=None, dpi=DEFAULT_DPI, cache_path=None, log=None):
    """Every page of a document, re-read from its image. {page: text}

    With a cache_path, each page is written as soon as it is transcribed and
    read back on a later run, so a run that dies at page 40 of 60 does not
    start over. This is the expensive pass, and the one most likely to be
    interrupted.
    """
    pages = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            pages = {int(page): text for page, text in json.load(fh).items()}

    last = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
    for page in range(1, last + 1):
        if page in pages:
            if log:
                log(f"    page {page}: kept from an earlier run")
            continue
        pages[page] = ocr_page(doc, page, dpi=dpi)
        if log:
            log(f"    page {page}: {len(pages[page])} chars")
        if cache_path:
            with open(cache_path, "w") as fh:
                json.dump(pages, fh)
    return {page: pages[page] for page in range(1, last + 1) if page in pages}


def assemble(doc, transcript=None, max_pages=None, figures=True):
    """A document as (text, figures). No model calls: everything is in hand.

    transcript is the output of transcribe() for a scan, or None to use the
    text and the figure objects the PDF carries itself.
    """
    last = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
    chunks, found = [], []
    for page in range(1, last + 1):
        if transcript is None:
            chunks.append(f"--- page {page} ---\n{page_text(doc, page)}")
            if figures:
                found += figures_from_objects(doc, page)
        else:
            text = transcript.get(page, "")
            chunks.append(f"--- page {page} ---\n{text}")
            if figures:
                # We already paid for the OCR, so reuse it to place the figures.
                found += [crop_region(doc, page, box) for box in figure_boxes(text)]
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


def ask(text, figures, prompt, schema, model=CHAT_MODEL, client=None,
        max_chars=MAX_CHARS, max_figures=MAX_FIGURES, num_ctx=NUM_CTX, log=None):
    """One extraction call: text and figures in, a list of records out."""
    if len(text) > max_chars and log:
        log(f"    text truncated to {max_chars} of {len(text)} characters "
            f"-- raise max_chars and num_ctx together to send it all")
    if len(figures) > max_figures and log:
        log(f"    sending {max_figures} of {len(figures)} figures "
            f"-- raise max_figures to send them all")

    message = {"role": "user", "content": prompt + text[:max_chars]}
    if figures:
        message["images"] = [as_image_data(f) for f in figures[:max_figures]]

    answer = _answer([message], schema, model, client, num_ctx, log=log)
    key = record_key(schema)
    records = answer.get(key, []) if key else [answer]
    return records if isinstance(records, list) else [records]


def extract_document(path, prompt, schema, model=CHAT_MODEL, client=None,
                     needs_ocr=None, figures=True, max_pages=None,
                     max_chars=MAX_CHARS, max_figures=MAX_FIGURES,
                     num_ctx=NUM_CTX, cache_path=None, log=None):
    """One PDF in, a list of records out.

    needs_ocr is None to let the model decide, or True/False when you already
    know — which, for your own material, you usually do.
    """
    doc = open_pdf(path)

    if needs_ocr is None:
        verdict = choose_strategy(doc, model=model, client=client, log=log)
        needs_ocr = verdict["needs_ocr"]
        if log:
            route = f"re-reading every page with {OCR_MODEL}" if needs_ocr \
                    else "using the text stored in the PDF"
            log(f"    {route} — {verdict['reason']}")

    transcript = None
    if needs_ocr:
        unload(model, client)                 # make room for the OCR model
        require_models(OCR_MODEL)
        transcript = transcribe(doc, max_pages=max_pages,
                                cache_path=cache_path, log=log)
        unload(OCR_MODEL, client)             # and give the GPU back

    text, found = assemble(doc, transcript, max_pages=max_pages, figures=figures)
    return ask(text, found, prompt, schema, model=model, client=client,
               max_chars=max_chars, max_figures=max_figures, num_ctx=num_ctx,
               log=log)


def extract_folder(folder, prompt, schema, model=CHAT_MODEL, client=None,
                   needs_ocr=None, figures=True, max_pages=None,
                   max_chars=MAX_CHARS, max_figures=MAX_FIGURES,
                   num_ctx=NUM_CTX, cache_dir=None, progress=True):
    """Every PDF in a folder, as one table.

    folder  -- a directory holding .pdf files
    prompt  -- what to extract, and what each field of your schema means
    schema  -- a JSON schema: the fields you want, wrapped in a named array

    Returns a DataFrame with one row per record and a `source` column saying
    which file it came from. A document that fails is reported and skipped, so
    one bad PDF does not cost you the rest of the run.

    The work runs in three passes -- judge every document, transcribe every
    document that needs it, then extract from all of them -- so each model is
    loaded once for the whole folder instead of once per document. Two models
    this size will not sit on a 16 GB GPU together.

    For a long run, pass cache_dir="somewhere": transcribed pages are written
    as they are read and finished records as they are extracted, both read back
    instead of being redone. Delete the folder when you change prompt or schema.
    """
    log = (lambda msg: print(msg, flush=True)) if progress else None
    paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"no PDFs in {folder!r}")
    require_models(model)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def cache_file(path, suffix):
        return (os.path.join(cache_dir, os.path.basename(path) + suffix)
                if cache_dir else None)

    # Anything already extracted needs none of the three passes.
    records = {}
    pending = []
    for path in paths:
        done = cache_file(path, ".records.json")
        if done and os.path.exists(done):
            with open(done) as fh:
                records[path] = json.load(fh)
            if log:
                log(f"{os.path.basename(path)}: {len(records[path])} record(s) "
                    f"from an earlier run")
        else:
            pending.append(path)

    docs = {path: open_pdf(path) for path in pending}

    # Pass 1 -- how should each document be read? The chat model loads once.
    plans = {}
    if pending and log:
        log("deciding how to read each document")
    for path in pending:
        name = os.path.basename(path)
        if needs_ocr is not None:
            plans[path] = needs_ocr
            if log:
                log(f"  {name}: told to "
                    f"{'re-read the pages' if needs_ocr else 'use the stored text'}")
            continue
        try:
            verdict = choose_strategy(docs[path], model=model, client=client, log=log)
        except Exception as exc:
            if log:
                log(f"  {name}: FAILED -- {exc}")
            plans[path] = None
            continue
        plans[path] = verdict["needs_ocr"]
        if log:
            route = "re-reading every page" if verdict["needs_ocr"] \
                    else "using the stored text"
            log(f"  {name}: {route} — {verdict['reason']}")

    # Pass 2 -- all the transcription together. The OCR model loads once.
    transcripts = {}
    if any(plans.get(p) for p in pending):
        unload(model, client)
        require_models(OCR_MODEL)
        for path in pending:
            if not plans.get(path):
                continue
            name = os.path.basename(path)
            if log:
                log(f"re-reading {name} with {OCR_MODEL}")
            try:
                transcripts[path] = transcribe(
                    docs[path], max_pages=max_pages,
                    cache_path=cache_file(path, ".ocr.json"), log=log)
            except Exception as exc:
                if log:
                    log(f"    FAILED -- {exc}")
                plans[path] = None
        unload(OCR_MODEL, client)

    # Pass 3 -- extraction. The chat model loads once more.
    for path in pending:
        if plans.get(path) is None:
            continue
        name = os.path.basename(path)
        if log:
            log(f"extracting from {name}")
        text, found = assemble(docs[path], transcripts.get(path),
                               max_pages=max_pages, figures=figures)
        try:
            got = ask(text, found, prompt, schema, model=model, client=client,
                      max_chars=max_chars, max_figures=max_figures,
                      num_ctx=num_ctx, log=log)
        except Exception as exc:
            if log:
                log(f"    FAILED -- {exc}")
            continue
        records[path] = got
        if log:
            log(f"    {len(got)} record(s)")
        done = cache_file(path, ".records.json")
        if done:
            with open(done, "w") as fh:
                json.dump(got, fh, indent=1)

    frames = []
    for path in paths:
        got = records.get(path)
        if got:
            frame = pd.DataFrame(got)
            frame.insert(0, "source", os.path.basename(path))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
