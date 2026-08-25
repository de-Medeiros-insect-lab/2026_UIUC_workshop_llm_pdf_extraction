"""Structured data out of a folder of PDFs, in one call.

    from pdf_extraction import extract_folder

    df = extract_folder("example_pdfs", prompt=MY_PROMPT, schema=MY_SCHEMA)

For each PDF in the folder this asks a reasoning model one question first — can
this document's own text layer be trusted, or do the pages have to be re-read
from their images? — and then writes the document out as a folder of markdown
pages, each in reading order with its figures linked where they appeared:

    processed/Marshall1929_AnnMagNatHist/
        page-001.md ... page-008.md
        figures/p006-fig01.png
        document.json

Go and read those. They are what the model is given, and keeping a caption next
to the figure it describes is the whole reason for reading a page in order
rather than collecting all the prose and then all the pictures.

From there it asks for exactly the fields your schema describes and returns one
table. Pages already written are never redone, so an interrupted run resumes,
and deleting a page has just that page re-read.

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


def ocr_elements(ocr_text):
    """The OCR model's regions in the order it read them: (label, box, text).

    deepseek-ocr writes a marker for each region and then that region's
    content, so the text between one marker and the next belongs to the first.
    """
    marks = list(re.finditer(
        r"(\w+)\s*\[\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]\]", ocr_text))
    found = []
    for i, mark in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(ocr_text)
        box = ("region", *(int(mark.group(k)) for k in range(2, 6)))
        found.append((mark.group(1), box, ocr_text[mark.end():stop].strip()))
    return found


def page_elements(doc, page, transcript=None, dpi=150):
    """One page as ("text", str) and ("figure", Image) pieces, in reading order.

    This is what lets a caption stay with its figure: we keep the order the
    page was laid out in rather than collecting all the text and then all the
    pictures.
    """
    if transcript is not None:                      # a scan: the OCR model
        pieces = []                                 # already read it in order
        for label, box, text in ocr_elements(transcript):
            if label == "image":
                pieces.append(("figure", crop_region(doc, page, box, dpi=dpi)))
            elif text:
                pieces.append(("text", text))
        return pieces

    pg = doc[page - 1]                              # born-digital: sort by
    page_area = pg.rect.width * pg.rect.height      # where things sit
    placed = []
    for x0, y0, x1, y1, text, _, kind in pg.get_text("blocks", sort=True):
        if kind == 0 and text.strip():
            placed.append((y0, ("text", text.strip())))
    for xref, *_ in pg.get_images(full=True):
        rects = pg.get_image_rects(xref)
        if not rects:
            continue
        share = max((r.width * r.height) / page_area for r in rects)
        if not MIN_FIGURE_AREA <= share <= MAX_FIGURE_AREA:
            continue
        figure = as_png_mode(Image.open(io.BytesIO(doc.extract_image(xref)["image"])))
        placed.append((rects[0].y0, ("figure", figure)))
    return [piece for _, piece in sorted(placed, key=lambda item: item[0])]


PNG_MODES = {"1", "L", "LA", "P", "RGB", "RGBA"}

def as_png_mode(image):
    """A version of an image that PNG can actually hold.

    Print-ready figures are often CMYK, and PNG has no CMYK: Pillow refuses to
    write it rather than approximating. Anything PNG cannot represent becomes
    RGB here, once, so that saving it and sending it both work later.
    """
    return image if image.mode in PNG_MODES else image.convert("RGB")


def as_image_data(fig, max_side=1024):
    """A figure, shrunk if it is huge, encoded the way a model wants it."""
    small = as_png_mode(fig).copy()
    small.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------- documents

FIGURE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def process_pdf(path, out_dir="processed", needs_ocr=False, max_pages=None,
                dpi=DEFAULT_DPI, log=None):
    """Turn a PDF into a folder you can open and read.

        processed/Marshall1929_AnnMagNatHist/
            page-001.md ... page-008.md
            figures/p006-fig01.png
            document.json

    Each page is markdown in reading order, with its figures linked where they
    appeared, so a caption stays next to the picture it belongs to. Open one in
    any markdown viewer and you see what the model is about to be given.

    Pages already written are left alone, so an interrupted run picks up where
    it stopped -- and deleting one page has just that page redone.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.join(out_dir, name)
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)

    doc = open_pdf(path)
    last = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
    for page in range(1, last + 1):
        page_file = os.path.join(folder, f"page-{page:03d}.md")
        if os.path.exists(page_file):
            if log:
                log(f"    page {page}: already done")
            continue

        transcript = ocr_page(doc, page, dpi=dpi) if needs_ocr else None
        lines, figures = [], 0
        for kind, value in page_elements(doc, page, transcript):
            if kind == "text":
                lines.append(value)
            else:
                figures += 1
                relative = f"figures/p{page:03d}-fig{figures:02d}.png"
                value.save(os.path.join(folder, relative))
                lines.append(f"![figure]({relative})")
        with open(page_file, "w") as fh:
            fh.write(f"# page {page}\n\n" + "\n\n".join(lines) + "\n")
        if log:
            log(f"    page {page}: {sum(len(l) for l in lines)} chars, "
                f"{figures} figure(s)")

    with open(os.path.join(folder, "document.json"), "w") as fh:
        json.dump({"source": os.path.basename(path),
                   "pages": last,
                   "read_with": OCR_MODEL if needs_ocr else "the PDF's own text"},
                  fh, indent=1)
    return folder


def load_pages(folder, pages=None):
    """A processed folder back as (text, figures), ready to send to a model.

    pages is None for all of them, or a range or list of page numbers.

    Ollama attaches images to a message rather than placing them in the text,
    so each figure link becomes a numbered marker where it stood, and the
    figures come back in that same order. The model cannot see the picture in
    position, but it can see where the picture was.
    """
    wanted = None if pages is None else {int(p) for p in pages}
    chunks, figures = [], []

    for page_file in sorted(glob.glob(os.path.join(folder, "page-*.md"))):
        page = int(os.path.basename(page_file)[5:8])
        if wanted is not None and page not in wanted:
            continue
        with open(page_file) as fh:
            text = fh.read()

        def mark(match):
            figures.append(Image.open(os.path.join(folder, match.group(1))))
            return f"[figure {len(figures)} appears here]"

        chunks.append(FIGURE_LINK.sub(mark, text))
    return "\n\n".join(chunks), figures


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
                     needs_ocr=None, out_dir="processed", pages=None,
                     max_pages=None, max_chars=MAX_CHARS,
                     max_figures=MAX_FIGURES, num_ctx=NUM_CTX, log=None):
    """One PDF in, a list of records out, by way of a folder you can inspect.

    needs_ocr is None to let the model decide, or True/False when you already
    know -- which, for your own material, you usually do.
    """
    if needs_ocr is None:
        verdict = choose_strategy(open_pdf(path), model=model, client=client,
                                  log=log)
        needs_ocr = verdict["needs_ocr"]
        if log:
            route = f"re-reading every page with {OCR_MODEL}" if needs_ocr \
                    else "using the text stored in the PDF"
            log(f"    {route} — {verdict['reason']}")

    if needs_ocr:
        unload(model, client)                 # make room for the OCR model
        require_models(OCR_MODEL)
    folder = process_pdf(path, out_dir=out_dir, needs_ocr=needs_ocr,
                         max_pages=max_pages, log=log)
    if needs_ocr:
        unload(OCR_MODEL, client)             # and give the GPU back

    text, figures = load_pages(folder, pages=pages)
    return ask(text, figures, prompt, schema, model=model, client=client,
               max_chars=max_chars, max_figures=max_figures, num_ctx=num_ctx,
               log=log)


def extract_folder(folder, prompt, schema, model=CHAT_MODEL, client=None,
                   needs_ocr=None, out_dir="processed", pages=None,
                   max_pages=None, max_chars=MAX_CHARS,
                   max_figures=MAX_FIGURES, num_ctx=NUM_CTX, progress=True):
    """Every PDF in a folder, as one table.

    folder  -- a directory holding .pdf files
    prompt  -- what to extract, and what each field of your schema means
    schema  -- a JSON schema: the fields you want, wrapped in a named array

    Returns a DataFrame with one row per record and a `source` column saying
    which file it came from. A document that fails is reported and skipped, so
    one bad PDF does not cost you the rest of the run.

    Every PDF is written to out_dir first, as a folder of markdown pages with
    the figures linked where they appeared -- go and read them, that is what
    the model is given. Pages already there are not redone, so an interrupted
    run resumes and a page you delete is the only one re-read.

    The work runs in three passes -- judge every document, process every
    document, extract from all of them -- so each model is loaded once for the
    whole folder. Two models this size will not sit on a 16 GB GPU together.
    """
    log = (lambda msg: print(msg, flush=True)) if progress else None
    paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not paths:
        raise FileNotFoundError(f"no PDFs in {folder!r}")
    require_models(model)

    # Pass 1 -- how should each document be read? The chat model loads once.
    if log:
        log("deciding how to read each document")
    plans = {}
    for path in paths:
        name = os.path.basename(path)
        if needs_ocr is not None:
            plans[path] = needs_ocr
            if log:
                log(f"  {name}: told to "
                    f"{'re-read the pages' if needs_ocr else 'use the stored text'}")
            continue
        try:
            verdict = choose_strategy(open_pdf(path), model=model, client=client,
                                      log=log)
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

    # Pass 2 -- write every document out. The OCR model loads once, if at all.
    scanned = [p for p in paths if plans.get(p)]
    if scanned:
        unload(model, client)
        require_models(OCR_MODEL)
    folders = {}
    for path in paths:
        if plans.get(path) is None:
            continue
        name = os.path.basename(path)
        if log:
            log(f"processing {name}")
        try:
            folders[path] = process_pdf(path, out_dir=out_dir,
                                        needs_ocr=plans[path],
                                        max_pages=max_pages, log=log)
        except Exception as exc:
            if log:
                log(f"    FAILED -- {exc}")
    if scanned:
        unload(OCR_MODEL, client)

    # Pass 3 -- extraction. The chat model loads once more.
    frames = []
    for path in paths:
        if path not in folders:
            continue
        name = os.path.basename(path)
        if log:
            log(f"extracting from {name}")
        text, figures = load_pages(folders[path], pages=pages)
        try:
            records = ask(text, figures, prompt, schema, model=model,
                          client=client, max_chars=max_chars,
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
