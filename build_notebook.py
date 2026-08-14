"""Generate workshop.ipynb: five sessions, self-contained, no imports from a
local module. Run:  python build_notebook.py"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ════════════════════════════════════════════════════════════ TITLE
md("""
# Extracting structured data from PDFs with open models

Five sessions. Everything runs free on Colab, on models you could also run on
your own machine.

1. **Setup** — talking to a language model from Python
2. **Preparing documents** — OCR and figure extraction
3. **Structured extraction** — getting tables out of prose
4. **An agent** — letting the model choose its own approach
5. **Scaling up** — a bigger model, and a whole folder of PDFs

**Set your runtime to a GPU before you start:** Runtime → Change runtime type → T4.
""")

# ════════════════════════════════════════════════════════ SESSION 1
md("""
---
# Session 1 — Setup, and talking to a model

**What we're doing:** installing Ollama, downloading a language model, and
sending it our first messages from Python.

**Why:** everything later in the day is this same call with more structure
around it. Get comfortable with it now, while the parts are still small.
""")

code("""
# Colab setup. Takes ~2 minutes on a fresh machine.
import os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # zstd unpacks Ollama's installer; pciutils lets it find the GPU.
    !DEBIAN_FRONTEND=noninteractive apt-get -qq install -y zstd pciutils > /dev/null
    !curl -fsSL https://ollama.com/install.sh | sh
    !pip -q install ollama pymupdf pandas openai pillow
    !git clone -q https://github.com/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction.git
    os.chdir("2026_UIUC_workshop_llm_pdf_extraction")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import time, base64, json, glob, re
import ollama, pymupdf, pandas as pd
from PIL import Image
import io

CHAT_MODEL = "qwen3.5:9b"      # reads and reasons
OCR_MODEL  = "deepseek-ocr"    # transcribes page images
""")

md("""
The server starts in the background, so our first request can arrive before
it is listening. A short polling loop is more pleasant than a crash — and
we will reuse it every time we restart.
""")

code("""
def server_ready(timeout=120):
    \"\"\"Wait until Ollama answers, or give up.\"\"\"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ollama.list()
            return True
        except Exception:
            time.sleep(1)
    return False

assert server_ready(), "Ollama did not start"
print("Ollama is up")
""")

md("""
### Check you actually have a GPU

Two separate questions: does this machine have a GPU, and is Ollama using it?
A model can quietly land on the CPU and simply be very slow.
""")

code("""
def gpu_report():
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"],
                             capture_output=True, text=True)
        print("GPU:", smi.stdout.strip() if smi.returncode == 0 else "none")
    except FileNotFoundError:
        print("GPU: none")
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    rows = [r for r in ps.stdout.strip().splitlines()[1:] if r.strip()]
    print("Ollama is running:", rows or "nothing loaded yet")

gpu_report()
""")

md("""
No GPU? Pair up with a neighbour. There is no CPU path today — the models are
too slow on one to follow along.
""")

code("""
# ~6.6 GB. Start it now.
!ollama pull {CHAT_MODEL}
""")

md("""
### Your first message

A conversation is a list of messages. Each has a `role` and `content`.
`temperature=0` makes the model as repeatable as it can be.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "What is a rostrum on a beetle?"}],
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
""")

md("""
### The system prompt

A `system` message sets who the model is being. Language models are best
understood as role-playing machines: say clearly what role you want, and you
get noticeably better answers.

> Shanahan, M., McDonell, K. & Reynolds, L. Role play with large language
> models. *Nature* **623**, 493–498 (2023).
""")

code("""
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
""")

md("""
**Check that answer against what you know.**

A 9B model is fluent about taxonomy and not reliable about it — it may well
have just told you something confidently wrong. Fluency and accuracy are
separate things.

This is why the rest of the day extracts text *from documents* rather than
asking the model what it knows.
""")

md("""
### Hands-on 1

Change the role and the question. Try an empty system prompt, then a very
detailed one, and compare.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "system", "content": "WRITE A ROLE HERE"},
        {"role": "user",   "content": "ASK YOUR QUESTION HERE"},
    ],
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
""")

md("""
### Thinking

This model can reason before answering. `think=True` returns that reasoning
separately from the answer.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": "A beetle is 4.2 mm long and 1.4 mm wide. "
                          "What is the length-to-width ratio?"}],
    think=True,
    options={"temperature": 0},
)
print("--- reasoning ---\\n", reply.message.thinking)
print("--- answer ---\\n", reply.message.content)
""")

md("""
Reasoning is not always a win. On a task with nothing to reason about — copying
text out of an image, say — it can ramble for pages and never answer. We use
`think=False` for that. In Session 4 the setting reverses, and reasoning is the
only thing that makes the job work.

**Recap.** You installed Ollama, pulled a model, and sent it messages. You set
a role with a system prompt, saw the model state something false with complete
confidence, and switched its reasoning on and off. Next: getting text out of
real documents.
""")

# ════════════════════════════════════════════════════════ SESSION 2
md("""
---
# Session 2 — Preparing documents

**What we're doing:** turning two PDFs into text and images we can feed to a
model — one born-digital, one a 1929 scan.

**Why:** a PDF is not a text file. What you can get out of it, and how much you
can trust it, depends entirely on how it was made. Get this wrong and
everything downstream is quietly wrong too.
""")

code("""
# ~6.7 GB. Start it now, it downloads while we talk.
!ollama pull {OCR_MODEL}
""")

md("""
We wrap page access in small functions because we call them constantly for the
rest of the day, and because page numbering is a trap worth handling once:
these use **1-based** pages, like the numbers printed on paper, while the
library underneath counts from 0.
""")

code("""
DEFAULT_DPI = 100   # enough to read fine print; higher is slower for no gain

def open_pdf(path):
    return pymupdf.open(path)

def get_page_text(doc, page):
    \"\"\"The text layer already stored inside the PDF. Free and instant.\"\"\"
    if not 1 <= page <= doc.page_count:
        raise ValueError(f"page {page} out of range (1-{doc.page_count})")
    return doc[page - 1].get_text()

def render_page(doc, page, dpi=DEFAULT_DPI):
    \"\"\"Draw a page as an image, encoded for sending to a model.\"\"\"
    if not 1 <= page <= doc.page_count:
        raise ValueError(f"page {page} out of range (1-{doc.page_count})")
    return base64.b64encode(
        doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")).decode()

modern = open_pdf("example_pdfs/deMedeiros2013Zootaxa.pdf")   # 2013, born-digital
legacy = open_pdf("example_pdfs/Marshall1929_AnnMagNatHist.pdf")  # 1929, scanned
print(modern.page_count, "pages |", legacy.page_count, "pages")
""")

md("""
### Two documents, two kinds of text
""")

code("""
print("MODERN, page 2:")
print(repr(get_page_text(modern, 2)[:180]))
print()
print("LEGACY, page 5:")
print(repr(get_page_text(legacy, 5)[:180]))
""")

md("""
The legacy text is not empty. It is not obviously broken either. Look closely:
the family name reads `Cureulionidse`. The scanner's OCR ran years ago and got
it wrong, and nothing in the file says so.

That is the dangerous case. An empty page announces itself; a page of
confident nonsense does not.
""")

code("""
from IPython.display import Image as ShowImage, display
display(ShowImage(data=base64.b64decode(render_page(legacy, 5))))
""")

md("""
### Figures: two completely different jobs

In a born-digital PDF, figures are **objects**. You pull them out exactly, for
free.
""")

code("""
for pno in range(1, modern.page_count + 1):
    imgs = modern[pno - 1].get_images(full=True)
    if imgs:
        print(f"page {pno}: {len(imgs)} embedded image(s)")

xref = modern[2].get_images(full=True)[0][0]
pix = pymupdf.Pixmap(modern, xref)
pix.save("figure_zootaxa.png")
print("saved figure_zootaxa.png", pix.width, "x", pix.height)
display(ShowImage(filename="figure_zootaxa.png", width=380))
""")

md("""
In a scan there are no objects — the whole page is one photograph. Page 6 of
the 1929 paper has a drawing of *Huarucus cacti* on it, but there is nothing
to extract. We have to **find** it.
""")

code("""
print("objects on legacy page 6:",
      len(legacy[5].get_images(full=True)), "-- the page scan itself")
""")

md("""
### OCR with layout

`deepseek-ocr` transcribes a page image, and asked the right way it also
reports *where* things are. Note this uses `ollama.generate`, not
`ollama.chat` — the layout information only comes through `generate`.
""")

code("""
def ocr_page(doc, page, dpi=DEFAULT_DPI):
    \"\"\"Transcribe a page from its image, with region coordinates.

    Coordinates come back scaled 0-1000, labelled text / image /
    image_caption. No think= here: this model does not reason, it transcribes.
    \"\"\"
    reply = ollama.generate(
        model=OCR_MODEL,
        prompt="<image>\\n<|grounding|>Convert the document to markdown.",
        images=[render_page(doc, page, dpi=dpi)],
        options={"temperature": 0, "num_predict": 4096},
    )
    return reply.response or ""

out = ocr_page(legacy, 6)
print(out[:600])
""")

md("""
Compare the corrupt line from the text layer with what OCR reads off the image.
""")

code("""
print("text layer :", get_page_text(legacy, 5)[:60].replace(chr(10), " "))
print("OCR        :", ocr_page(legacy, 5)[:80].replace(chr(10), " "))
""")

md("""
### Cropping the figure

The regions are just coordinates in the text. We parse them out and cut the
image — a small function, because Session 5 does this over a whole folder.
""")

code("""
def regions(ocr_text):
    \"\"\"Every labelled region: [(label, x1, y1, x2, y2), ...], scaled 0-1000.\"\"\"
    found = re.findall(r"(\\w+)\\s*\\[\\[\\s*(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\s*\\]\\]",
                       ocr_text)
    return [(lab, *map(int, box)) for lab, *box in found]

def crop_region(doc, page, box, dpi=150, pad=0.01):
    \"\"\"Cut one 0-1000 box out of a rendered page.\"\"\"
    im = Image.open(io.BytesIO(
        doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")))
    W, H = im.size
    _, x1, y1, x2, y2 = box
    return im.crop((int((x1/1000 - pad) * W), int((y1/1000 - pad) * H),
                    int((x2/1000 + pad) * W), int((y2/1000 + pad) * H)))

for r in regions(out):
    print(r)
""")

code("""
figures = [r for r in regions(out) if r[0] in ("image", "image_caption")]
if figures:
    x1 = min(f[1] for f in figures); y1 = min(f[2] for f in figures)
    x2 = max(f[3] for f in figures); y2 = max(f[4] for f in figures)
    crop = crop_region(legacy, 6, ("figure", x1, y1, x2, y2))
    crop.save("figure_huarucus.png")
    display(crop)
""")

md("""
**Recap.** Born-digital PDFs give you clean text and figures as objects. Scans
give you pixels and, often, a text layer that is wrong without saying so. OCR
recovers the text and tells you where each region sits, which is enough to cut
figures out of a page that contains no figure objects.

Next: turning this text into data.
""")

# ════════════════════════════════════════════════════════ SESSION 3
md("""
---
# Session 3 — Structured extraction

**What we're doing:** getting a table of species and traits out of prose.

**Why:** a paragraph is not data. To compare hundreds of descriptions you need
the same fields, with the same names, every time.

**Start the cell below now** — it OCRs the whole 1929 paper and takes a few
minutes. We will talk while it runs.
""")

code("""
# Long-running: OCR every page of the scanned paper and keep the text.
legacy_ocr = {}
for pno in range(2, legacy.page_count + 1):
    legacy_ocr[pno] = ocr_page(legacy, pno)
    print(f"page {pno} done ({len(legacy_ocr[pno])} chars)")

with open("legacy_ocr.json", "w") as fh:
    json.dump(legacy_ocr, fh, indent=1)
print("saved legacy_ocr.json")
""")

md("""
### Asking for JSON is not enough

The obvious approach: ask for JSON.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": "List the species and their traits as JSON.\\n\\n"
                          + get_page_text(modern, 2)[:1500]}],
    format="json",
    think=False,
    options={"temperature": 0},
)
print(reply.message.content[:400])
""")

md("""
That is valid JSON, but the model chose the field names. Run it again and they
may change. You cannot build a table on that.

### A schema

A JSON schema is a dictionary describing the shape you want. Pass it as
`format=` and the model cannot produce anything else.
""")

code("""
SPECIES_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string"},
                    "traits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anatomical_part": {"type": "string"},
                                "trait":           {"type": "string"},
                                "value":           {"type": "string"},
                                "units":           {"type": "string"},
                                "source_text":     {"type": "string"},
                            },
                            "required": ["anatomical_part", "trait", "value",
                                         "source_text"],
                        },
                    },
                },
                "required": ["name", "traits"],
            },
        }
    },
    "required": ["species"],
}
""")

md("""
`source_text` is required on purpose: every extracted value should be traceable
back to the sentence it came from.

We wrap the call in a function because Session 5 runs it over a whole folder.
""")

code("""
EXTRACT_PROMPT = (
    "Extract every species described in this text, with their morphological "
    "traits. Copy the exact source sentence for each trait into source_text. "
    "Do not invent traits that are not stated.\\n\\n"
)

def extract(text, schema=SPECIES_SCHEMA, model=CHAT_MODEL, client=None):
    \"\"\"Pull structured data out of text.

    client is None for the local server. Session 5 passes a cloud client
    instead, which is the only change needed to run this on a far bigger model.
    \"\"\"
    chat = (client or ollama).chat
    reply = chat(
        model=model,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text}],
        format=schema,
        think=False,
        options={"temperature": 0},
    )
    return json.loads(reply.message.content)

def to_table(data):
    rows = [{"species": sp["name"], **t}
            for sp in data.get("species", []) for t in sp.get("traits", [])]
    return pd.DataFrame(rows)
""")

code("""
data = extract(get_page_text(modern, 2) + get_page_text(modern, 4))
df = to_table(data)
print(f"{df['species'].nunique()} species, {len(df)} rows")
df.head(10)
""")

md("""
### The same, from the scan

Now the OCR text from the cell you started earlier.
""")

code("""
legacy_text = "\\n".join(legacy_ocr[p] for p in sorted(legacy_ocr))
legacy_df = to_table(extract(legacy_text[:12000]))
print(f"{legacy_df['species'].nunique()} species, {len(legacy_df)} rows")
legacy_df.head(10)
""")

md("""
### What a schema cannot do

The schema fixes the shape. It says nothing about whether the numbers are
sensible.
""")

code("""
def implausible(row):
    \"\"\"A check the schema cannot express.\"\"\"
    if row.get("units") == "mm":
        try:
            mm = float(row["value"])
        except (ValueError, TypeError):
            return None
        if not 0 < mm <= 300:
            return f"{mm} mm is not plausible"
    return None

print(implausible({"units": "mm", "value": "5000"}))
for _, row in df.iterrows():
    if (problem := implausible(row)):
        print(row["species"], row["trait"], "->", problem)
""")

md("""
**Recap.** Asking for JSON gets you JSON with unpredictable field names. A
schema passed as `format=` fixes the fields, the types and what is required,
and the model cannot deviate. It cannot tell you whether an answer is *true* —
that check is yours to write.

Next: letting the model decide how to read a page.
""")

# ════════════════════════════════════════════════════════ SESSION 4
md("""
---
# Session 4 — An agent that chooses

**What we're doing:** giving the model two tools — cheap text, expensive OCR —
and letting it pick.

**Why:** you will not always know which pages of which document need OCR. The
alternative to deciding yourself is describing the tools well enough that the
model decides.
""")

md("""
Tools are described as JSON, much like the schema. The description is what the
model reads to choose — write it for the model, not for yourself.
""")

code("""
TOOLS = [
    {"type": "function",
     "function": {
         "name": "get_page_text",
         "description": ("Return the PDF's embedded text layer for a page. "
                         "Free and instant, but on scanned documents it can be "
                         "poor-quality OCR with garbled words."),
         "parameters": {"type": "object",
                        "properties": {"page": {"type": "integer",
                                                "description": "1-based page number"}},
                        "required": ["page"]}}},
    {"type": "function",
     "function": {
         "name": "ocr_page",
         "description": ("Re-read a page from its image with a dedicated OCR "
                         "model. Slower, far more accurate on scans."),
         "parameters": {"type": "object",
                        "properties": {"page": {"type": "integer",
                                                "description": "1-based page number"}},
                        "required": ["page"]}}},
]
""")

md("""
The model can only *ask* for a tool; we run it and hand back the result. That
back-and-forth is the loop below. It is a function because we call it several
times with different settings.
""")

code("""
def run_tool_loop(messages, tools, impls, think=True, max_turns=6):
    \"\"\"Let the model call tools until it has an answer.

    Returns (answer, calls_it_made).
    \"\"\"
    messages = list(messages)
    calls = []
    for _ in range(max_turns):
        reply = ollama.chat(model=CHAT_MODEL, messages=messages, tools=tools,
                            think=think, options={"temperature": 0})
        msg = reply.message
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": msg.tool_calls or []})
        if not msg.tool_calls:
            return (msg.content or ""), calls
        for call in msg.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            calls.append((name, args))
            try:
                result = impls[name](**args)
            except Exception as exc:
                result = f"ERROR: {exc}"        # tell the model, don't crash
            messages.append({"role": "tool", "name": name,
                             "content": str(result)[:6000]})
    return f"stopped after {max_turns} turns", calls

impls = {"get_page_text": lambda page: get_page_text(legacy, page),
         "ocr_page":      lambda page: ocr_page(legacy, page)}

SYSTEM = ("You read scanned historical literature. The embedded text layer of "
          "a scan is often corrupt: watch for garbled words and impossible "
          "spellings of taxonomic names. If the text looks corrupt, re-read "
          "the page with ocr_page before trusting it.")
QUESTION = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             "On page 5, what is the family name printed in the running "
             "header at the top? Report it exactly."}]
""")

code("""
answer, calls = run_tool_loop(QUESTION, TOOLS, impls, think=False)
print("tools used:", calls)
print(answer)
""")

md("""
It read the corrupt text, noticed something was off, decided it was close
enough, and answered anyway — often inventing a spelling that is in neither the
text nor the page.

Now change one argument.
""")

code("""
answer, calls = run_tool_loop(QUESTION, TOOLS, impls, think=True)
print("tools used:", calls)
print(answer)
""")

md("""
With reasoning on it re-reads the page and gets it right. Judging whether your
input is trustworthy *is* reasoning work — it is the same setting that wasted
its time back in Session 1, applied to a task that actually needs it.

### Why not just write a rule?

Tempting: scan the text for odd characters and call OCR when you find them.
It works on the page you tuned it on. There is no finite list of ways OCR can
be wrong, so it fails on the next document — silently, which is the worst way.

In practice you either already know which of your PDFs are scans, or you ask a
model that can reason.
""")

md("""
### Hands-on 2

Run the loop over several pages and watch which ones it sends to OCR.
""")

code("""
for pno in [2, 5, 6]:
    q = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content":
          f"What is printed at the very top of page {pno}? Report it exactly."}]
    answer, calls = run_tool_loop(q, TOOLS, impls, think=True)
    print(f"p{pno}: {[c[0] for c in calls]} -> {' '.join(answer.split())[:90]}")
""")

md("""
**Recap.** You described two tools, ran the loop that lets a model use them,
and saw that the same code fails or succeeds depending on whether reasoning is
switched on. Deciding *where* a decision lives — in your code or in the model —
is the judgement that carries over to your own pipelines.

Next: doing this at scale.
""")

# ════════════════════════════════════════════════════════ SESSION 5
md("""
---
# Session 5 — Scaling up

**What we're doing:** swapping in a much larger model, then running the whole
pipeline over every PDF in a folder.

**Why:** the shape of the code is the same for two documents or four thousand.
What changes is that you save results as you go, so a failure halfway through
does not cost you the whole run.
""")

md("""
### Get your own API key

The models we have used so far fit on this machine. Bigger ones do not, but
Ollama runs them on their servers and the code barely changes.

You need a key of your own. It is free and takes two minutes:

1. Go to **[ollama.com](https://ollama.com)** and create an account. No card
   required.
2. Go to **[ollama.com/settings/keys](https://ollama.com/settings/keys)** →
   **Create key**. Copy it. You will not be shown it again.
3. Back in Colab, click the **🔑 key icon** in the left sidebar.
4. **+ Add new secret**. Name it exactly `OLLAMA_API_KEY`, paste your key into
   the value box, and turn on **Notebook access**.

**Never paste a key into a cell.** Anything typed into a notebook gets shared
when the notebook does — with your collaborators, in your repository, in the
copy you email to a student. Colab Secrets keeps it out of the file.
""")

code("""
from ollama import Client

def cloud_client():
    \"\"\"A client pointed at Ollama's servers instead of this machine.

    Same interface as the local one, so everything we have written already
    works against it -- we only have to say where to send the request and
    who is asking.
    \"\"\"
    try:
        from google.colab import userdata
        key = userdata.get("OLLAMA_API_KEY")
    except Exception:
        key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError(
            "No key found. Add OLLAMA_API_KEY to Colab Secrets (key icon, "
            "left sidebar) and switch on Notebook access for this notebook.")
    return Client(host="https://ollama.com",
                  headers={"Authorization": "Bearer " + key})

# Browse what is available at https://ollama.com/search?c=cloud
# Note: no "-cloud" suffix when talking to the servers directly.
CLOUD_MODEL = "gpt-oss:120b"

try:
    cloud = cloud_client()
    data = extract(get_page_text(modern, 2)[:4000],
                   model=CLOUD_MODEL, client=cloud)
    display(to_table(data).head())
except Exception as exc:
    print("Cloud step skipped:", exc)
""")

md("""
That is the whole difference: same prompt, same schema, same `extract`
function — a different client and a model roughly a hundred times larger.

The free tier is metered, so keep cloud calls small. Everything else today
runs on the machine in front of you.
""")

md("""
### Every PDF in a folder

One JSON file per document, written as we go.
""")

code("""
def process_pdf(path, model=CHAT_MODEL, max_chars=12000):
    \"\"\"Read a PDF, extract structured data, return it.\"\"\"
    doc = open_pdf(path)
    text = "\\n".join(get_page_text(doc, p)
                     for p in range(1, doc.page_count + 1))
    return extract(text[:max_chars], model=model)

os.makedirs("results", exist_ok=True)

for path in sorted(glob.glob("example_pdfs/*.pdf")):
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = f"results/{name}.json"
    if os.path.exists(out_path):
        print("skip (already done):", name)
        continue
    try:
        data = process_pdf(path)
        with open(out_path, "w") as fh:
            json.dump(data, fh, indent=1)
        n = sum(len(s.get("traits", [])) for s in data.get("species", []))
        print(f"{name}: {len(data.get('species', []))} species, {n} traits")
    except Exception as exc:
        print(f"{name}: FAILED -- {exc}")
""")

md("""
Skipping files that already have output means you can re-run the cell after a
crash and it picks up where it stopped. At any real scale this matters more
than anything else in the loop.

### One table
""")

code("""
frames = []
for path in sorted(glob.glob("results/*.json")):
    with open(path) as fh:
        data = json.load(fh)
    t = to_table(data)
    t.insert(0, "source", os.path.basename(path).replace(".json", ""))
    frames.append(t)

combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
print(combined.shape)
combined.to_csv("results/combined.csv", index=False)
combined.head(15)
""")

md("""
### Hands-on 3 — your own data

Upload a few of your own PDFs, write a schema for what *you* want out of them,
and run the same loop.

1. Click the folder icon 📁 in the left sidebar and upload PDFs into
   `my_pdfs/` (create it in the cell below).
2. Edit `MY_SCHEMA` for the fields you want.
3. Edit `MY_PROMPT` to say what to extract.
4. Run.

Start with two or three pages of one document while you are getting the prompt
right — not your whole library.
""")

code("""
os.makedirs("my_pdfs", exist_ok=True)
print("Upload PDFs into my_pdfs/ using the folder icon in the sidebar.")

MY_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "FIELD_ONE":   {"type": "string"},
                    "FIELD_TWO":   {"type": "string"},
                    "source_text": {"type": "string"},
                },
                "required": ["FIELD_ONE", "source_text"],
            },
        }
    },
    "required": ["records"],
}

MY_PROMPT = (
    "DESCRIBE WHAT TO EXTRACT HERE. Be specific. Copy the exact sentence "
    "each value came from into source_text.\\n\\n"
)

def my_extract(text, model=CHAT_MODEL):
    reply = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": MY_PROMPT + text}],
        format=MY_SCHEMA,
        think=False,
        options={"temperature": 0},
    )
    return json.loads(reply.message.content)

for path in sorted(glob.glob("my_pdfs/*.pdf")):
    doc = open_pdf(path)
    text = "\\n".join(get_page_text(doc, p)
                      for p in range(1, min(doc.page_count, 3) + 1))
    try:
        result = my_extract(text[:8000])
        print(os.path.basename(path))
        display(pd.DataFrame(result.get("records", [])))
    except Exception as exc:
        print(f"{os.path.basename(path)}: {exc}")
""")

md("""
**Recap.** The pipeline is a loop: read a document, extract with a schema, save
the result, move on. Saving per document and skipping finished work is what
makes it survivable at scale. Swapping to a larger model is a one-line change.

**Where to go from here**

- Test on a handful of documents you have already scored by hand, and measure
  how often the model agrees with you, before trusting a large run.
- Keep `source_text`. An extraction you cannot trace is not evidence.
- Pin your model *and* your Ollama version if you publish a method — open
  weights can be archived alongside your data, and a hosted model cannot.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

with open("workshop.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")

print(f"wrote workshop.ipynb: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in cells)} markdown)")
