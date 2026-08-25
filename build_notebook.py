"""Generate workshop.ipynb: five sessions, self-contained, no imports from a
local module.

    python build_notebook.py            # write the notebook
    python build_notebook.py --check    # say whether it is already up to date

This script and the notebook carry the same content, so only one of them can be
the copy you edit.

Editing here is safe: cells whose source has not changed keep the outputs they
already had, so a rebuild does not wipe the results of a run. Editing in Colab
instead means a rebuild would revert your work -- so after a Colab session, run
--check before you run anything else.
"""
import hashlib
import json
import os
import sys

NOTEBOOK = "workshop.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
<a href="https://colab.research.google.com/github/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction/blob/manual_changes/workshop.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

# ══════════════════ EXTRACTING STRUCTURED DATA FROM PDFS WITH OPEN MODELS
md("""
# Extracting structured data from PDFs with open models

In this workshop, we will learn how to interact programmatically with large language models (LLMs) and then apply this to the task of extracting structured data from PDFs.

Everything here runs free on Google Colab using a GPU with 16GB of RAM, which is enough for low-end LLMs. If you have this memory on your machine, you can run locally too by downloading this as a python notebook. You can run python notebooks using [Jupyter](https://jupyter.org/), among other software.

We will focus on a method using open-source models as minimizing required computing power, so you are able to run this yourself relatively cheaply. Many of the steps are unnecessary if you use highly sophisticated (and expensive) models with a good [agent harness](https://learn.microsoft.com/en-us/agent-framework/concepts/harness?pivots=programming-language-csharp) like [Claude code](https://claude.com/product/claude-code).

We will go through 5 steps:

1. **Setup** — talking to a language model from Python
2. **Preparing documents** — OCR and figure extraction
3. **Structured extraction** — getting tables out of unstructured text
4. **An agent** — creating an AI agent that chooses the best approach
5. **Scaling up** — a bigger model in the cloud, and a whole folder of PDFs



**To get started, set your runtime to a GPU before you start:** Runtime → Change runtime type → T4.
Then click *Connect*
""")

# ══════════════════════════════════════════════════════════════ SESSION 1
md("""
---
# Session 1 — Setup, and talking to a model

## 1.1 - Interacting with Ollama through the desktop app

We asked you to install the [Ollama Desktop app](https://ollama.com/) before the workshop. Let's first interact with it locally on your computer, and later we will do the same programmatically through this python notebook.

Ollama is one of the programs that you can use to deploy large language models with open weights. These are [open-source models](https://en.wikipedia.org/wiki/Open-source_artificial_intelligence) that you can run for free on your own device, or rent cloud computing to run. These include the models trained by Chinese companies and highly talked about in the media, but some US-based companies also release open source models (OpenAI has [a few, now a bit outdated](https://ollama.com/library/gpt-oss), and meta [just released a new one](https://ollama.com/library/muse-glimmer)).

You do not need Ollama to run open models, but Ollama offers a convenient engine that automatically figures out the best way to deploy a model on your hardware. For pure python, you can check the [Transformers library and models hosted by Huggingface](https://huggingface.co/docs/transformers/en/index). We will have Ollama running as a server in the background in this workshop and use python to communicate with this server. As you learn more about using LLMs, you can explore the Transformers library, [vLLM](https://vllm.ai/), [unsloth](https://unsloth.ai/) and other ways to do this.

Let's start opening Ollama on your own computer. Click on *New chat*, choose the model *deepseek-r1:1.5b*, and start chatting. The interface should be familiar to you, similar to other web-based chatbots.
* We are using *deepseek-r1:1.5b* because it is a very small model that can run on pretty much any of your computers, whether or not you have a powerful GPU. Let's start with this one so everyone is on the same page.
* **DO NOT USE A CLOUD MODEL YET**
""")

md("""
## 1.2 - Interacting with Ollama through the command line

Now that you are comfortable with the graphical user interface, let's chat with Ollama through a command line, which is one level closer to what we will do here. While Ollama is open, open a terminal (for Windows, Windows PowerShell; for Macs, Terminal; for Ubuntu, GNOME Shell, etc).

Type the following command:
```{bash}
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{ "model": "deepseek-r1:1.5b", "messages": [ { "role": "user", "content": "Which insect is the best? One must choose one and one only" } ] }'
```
""")

md("""
### What you will see back

You will get a very long line in [JSON format](https://en.wikipedia.org/wiki/JSON). Here I am showing results that I got when preparing the workshop, broken over several lines, with some parts numbered for us to discuss them:

```json
{
  "id": "chatcmpl-364",
  "object": "chat.completion",
  "created": 1786808743,
  "model": "deepseek-r1:1.5b",                 // 1️⃣
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",                   // 2️⃣
        "content": "Ants are the best ...",    // 3️⃣
        "reasoning": "Alright, so I need ..."  // 4️⃣
      },
      "finish_reason": "stop"                  // 5️⃣
    }
  ],
  "usage": { "prompt_tokens": 16, "completion_tokens": 638, "total_tokens": 654 }   // 6️⃣
}
```

(The `//` lines are our annotations — real JSON has no comments.)

| | field | what it is |
| :-- | :-- | :-- |
| 1️⃣ | `model` | the name of the model used. |
| 2️⃣ | `role` | `assistant`, `user` or `system`; in this case, a response, so `assistant`. |
| 3️⃣ | `content` | **the answer**. The only part a chat window shows you. |
| 4️⃣ | `reasoning` | the model thinking out loud first, kept separate. Only reasoning models return this. |
| 5️⃣ | `finish_reason` | `stop` means it finished. `length` means it hit a limit mid-sentence and your answer is cut off. Always check this. |
| 6️⃣ | `usage` | Number of tokens used, important for cost and model context. |
""")

md("""
### 3️⃣ &nbsp;The answer

> Ants are the best because they excel in pollination, carrying and spreading
> pollens throughout most of their lifetime in their home regions. This makes
> them highly effective in ensuring plant reproduction.
""")

md("""
### 4️⃣ &nbsp;The reasoning

> Alright, so I need to figure out which insect is the best. I guess the user is asking for the most effective or the best known insect. Let me start by thinking about insects in general—there are over 30, each with different roles and uses, from ants to moles.
>
> I remember that ants are generally noted for their work in pollination. They carry fruit all year around and are crucial for pollination in their native regions. Moles, on the other hand, I think are good at defense and scaring off neighbors, which probably helps them survive. They can be aggressive and make aggressive behavior.
>
> Wait, but if I'm comparing ants and moles, which one is better in what ways? The user said to choose one only, but that's confusing because they're two different insects with distinct roles. Maybe the user is asking which kind is the best or perhaps comparing based on specific aspects.
>
> Let me think about size. Moles are generally smaller than ants, so ants are larger. But ants are more effective in certain ways. Maybe I should consider their reproductive success in pollination, which ants do more than moles. Also, ants might be more aggressive, so if the user values that, perhaps ants are better.
>
> Another angle is their behavior. Moles are known for being aggressive, which can be advantageous in some environments. However, they might not handle certain tasks as efficiently as ants. On the flip side, if the focus is on defense, moles might be better, but their reputation as scaring off neighbors is something ants don't have.
>
> I'm a bit torn because there's no universal "best." The answer will depend on the criteria. If it's based on ability to pollinate, ants are better. If it's based on defense, moles might be better. But since the user is asking for an insect, not a group, perhaps they want a single answer. Maybe the ants are superior at pollination.
>
> Alternatively, maybe the question is about the best insect for garden use, considering factors like defense, size, and functionality. In that case, moles could be better for defense but less effective in pollination. But if considering a mix, like defense and pollination, maybe it's a balance.
>
> I'm also thinking about the user's perspective. They might be looking for an insect that's efficient in their specific niche, like for pollination, or maybe in defense. Without knowing exactly, it's a bit unclear. But I'll conclude that while there might not be a single best insect, between the two, considering pollination, ants are superior. So I'll say ants are the best because they excel in pollination and are effective.
>
> Wait, but the user might be looking for something else, like functional traits or specific environmental roles. So perhaps the answer could vary based on the criteria. But sticking with pollination seems solid, so ants are my choice for that.

## Three things to take from this response:

- A chat window shows you the response 3️⃣ only. Working
  programmatically you get a lot more information, including all reasoning and lots of metadata.
- Reasoning is necessary for some tasks, but it is expensive.
- Reasoning has limits: small models may still hallucinate.
""")

md("""
## 1.3 - Interacting with Ollama through a cloud Python notebook

From now on, we will still be running Ollama, but not on your own computer. This colab notebook runs on a cloud instance on a Google server. The free tier is limited to GPUs with 16 GB of memory (so this is the maximum model size + data we can use). We will install Ollama on the server, download models and interact with them.

**What we're doing:** installing Ollama, downloading a language model, and
sending it our first messages from Python.

**Why:** everything later in the day is this same call with more structure
around it. We will get comfortable with it now, while the parts are still small.
""")

code("""
# Colab setup. Takes ~2 minutes on a fresh machine.
import os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
REPO = "2026_UIUC_workshop_llm_pdf_extraction"
BRANCH = "main"   # switch to your working branch to test unmerged changes

if IN_COLAB:
    # zstd unpacks Ollama's installer; pciutils lets it find the GPU.
    !DEBIAN_FRONTEND=noninteractive apt-get -qq install -y zstd pciutils > /dev/null
    !curl -fsSL https://ollama.com/install.sh | sh
    !pip -q install ollama pymupdf pandas pillow
    if os.path.basename(os.getcwd()) != REPO:   # so this cell is safe to re-run
        if not os.path.isdir(REPO):
            !git clone -q --branch {BRANCH} https://github.com/de-Medeiros-insect-lab/{REPO}.git
        os.chdir(REPO)
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import time, base64, json, glob, re
import ollama, pymupdf, pandas as pd
from PIL import Image
from IPython.display import display, Image as ShowImage
import io
""")

md("""
The code below just checks whether the server is ready so we do not get an error when using it.
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
### If your runtime disconnects

It happens. Colab times out, or the GPU is taken back. There are two cases and
they are not equally bad.

**The kernel restarted, the machine is still yours.** Everything you downloaded
and everything you wrote to disk is still there. Click the cell you were
working in and choose **Runtime → Run before**. It re-runs the notebook from
the top, but the slow parts are already done: `ollama pull` sees the models on
disk and returns at once, and the OCR skips every page already written into
`processed/`. Expect a few minutes, mostly model calls, not the twenty you
spent the first time.

**Colab gave you a different machine.** Then the disk went too — no models, no
`processed/`, no repository. There is nothing to resume; run the setup cell at
the top and start again. If you are working on something you cannot afford to
lose, mount your Google Drive and write `processed/` there instead.

Either way, **Run before is always safe**. Every cell in this notebook,
including the hands-on ones, runs top to bottom without needing you to fill
anything in first.
""")

md("""
### Check you actually have a GPU

Models can run on a CPU, but then they get very slow. So let's check whether you have a GPU.
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
    try:
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
        rows = [r for r in ps.stdout.strip().splitlines()[1:] if r.strip()]
        print("Ollama is running:", rows or "nothing loaded yet")
    except FileNotFoundError:
        print("Ollama is running: the ollama command was not found -- "
              "did the setup cell above finish?")

gpu_report()
""")

md("""
If you do not have a GPU, **STOP** and ask the instructors. You may have forgotten to choose the T4 runtime type.

Now we will download our chat model. For convenience, we will save our chat model name in a variable so we can reuse it. We will use the model [Qwen 3.5](https://qwen.ai/blog?id=qwen3.5) with 9 billion parameters, which takes ~6.5 GB of RAM. It is a quite powerful model for a small size, with capabilities of chatting, reasoning, using tools, and taking images as input. More powerful versions of Qwen have more parameters, but need more RAM.

The terminal command to ask ollama to download a model is `ollama pull`. In a python
notebook we can run a terminal command by starting the line with `!`, and paste the
value of a python variable into it with `{}` — so `!ollama pull {CHAT_MODEL}` runs
`ollama pull qwen3.5:9b`. The download is ~6.5 GB, so start it and keep reading.
""")

code("""
CHAT_MODEL = "qwen3.5:9b"
!ollama pull {CHAT_MODEL}
""")

md("""
### Your first message

A conversation is a list of messages. Each has a `role` and `content`.
`temperature` sets how noisy a model will be. Higher temperature may make responses more creative because they are more variable. In our context, we typically want `temperature=0`, which makes the model as predictable and repeatable as it can be.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": "What is the best insect, and why?"}],
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
""")

md("""
### The system prompt

A `system` message sets who the model is being. Language models are best
understood as role-playing machines:

> Shanahan, M., McDonell, K. & Reynolds, L. [Role play with large language models](https://www.nature.com/articles/s41586-023-06647-8). *Nature* **623**, 493–498 (2023).

Say clearly what role you want, and the model will follow. They know lots of roles from the training data!

Let's try sending a message with a strict system constraint. We will use a very slim system prompt, but the chatbots your are more used to interacting with[ have a lot more there.](https://platform.claude.com/docs/en/release-notes/system-prompts)

Note that we are using `think=False`: the model will just answer directly without reasoning first.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "system",
         "content": "You are an expert coleopterist. Answer in two short sentences."},
        {"role": "user",
         "content": "What is the best insect, and why?"},
    ],
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
""")

md("""
**Check that answer against what you know.**

A 9B model is fluent about many things but not very reliable — it may well
have just told you something confidently wrong.

This is why we will extract text *from documents* rather than
asking the model what it knows. Both using larger models and providing stronger contextualization will decrease the chances that the model will make stuff up.
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
separately from the answer. Let's try the same question as above:
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[
        {"role": "system",
         "content": "You are an expert coleopterist. Answer in two short sentences."},
        {"role": "user",
         "content": "What is the best insect, and why?"},
    ],
    think="low",
    options={"temperature": 0},
)

print("--- reasoning ---\\n", reply.message.thinking)
print("--- answer ---\\n", reply.message.content)
""")

md("""
Reasoning is not always prerable. On a task with nothing to reason about — copying
text out of an image, say — it can ramble for pages and never answer. We use
`think=False` for that. For more complicated tasks, it may help tremendously.

**Recap.** You installed Ollama, pulled a model, and sent it messages. You set
a role with a system prompt, evaluated the model responses, and switched its reasoning on and off. Next: getting text out of
real documents.
""")

# ══════════════════════════════════════════════════════════════ SESSION 2
md("""
---
# Session 2 — Preparing documents

**What we're doing:** turning two PDFs into text and images we can feed to a
model — one born-digital, one a 1929 scan.

**Why:** a PDF is not a text file. What you can get out of it, and how much you
can trust it, depends entirely on how it was made.

Very powerful models (e.g. Claude models) can read PDFs natively, but use a lot of tokens for it. If we prepare our PDFs beforehand as text and images, we can get high-quality responses with a much lower cost.

It is very common for old scans to have used outdated OCR tools, so it is usually a good idea to re-run OCR. Here we will use Ollama to do OCR using a modern model.

OCR models are highly specialized - they only do OCR and nothing else. So they are usually pretty good even being small. And models from Chinese companies tend to be very good in multiple languages. Here we will use [deepseek OCR](https://github.com/deepseek-ai/DeepSeek-OCR) as our OCR model. It is not the latest version, but it is already excellent and supported by Ollama.
""")

code("""
# ~6.7 GB. Start it now, it downloads while we talk.
OCR_MODEL  = "deepseek-ocr"    # transcribes page images
!ollama pull {OCR_MODEL}
""")

md("""
We will now define several small functions that we will call constantly throughout the day. We will not explain in detail how each one works, only their general goal. Feel free to use these functions in your own code, and get your favorite chatbot to explain them to you if you are curious!

**open_pdf()**: opens a pdf file using the [pymupdf library](https://pymupdf.readthedocs.io/en/latest/)

**get_page_text()**: gets a specific page from a loaded pdf.

**render_page()**: encodes a pdf page in the format required by an LLM

**DEFAULT_DPI**: the resolution we will be using. We are limiting to 100 dpi here so we limit the memory (and number of tokens, and computing time, etc). Probably sufficient for most text and figures.

The two PDFs we will use are taxonomic publications, one born-digital and the other historical and digitized. You can find them in the [github repository](https://github.com/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction)
""")

code("""
DEFAULT_DPI = 100   # enough to read fine print; higher is slower for little gain

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
The legacy text is not empty, but its content is hidden  Look closely:
the family name reads `Cureulionidse`. The scanner's OCR ran years ago and got
it wrong, and nothing in the file makes this clear.

That is a dangerous and common case. Better never trust the OCR that comes with a historical PDF.
""")

code("""
from IPython.display import Image as ShowImage, display
display(ShowImage(data=base64.b64decode(render_page(legacy, 5))))
""")

md("""
### Figures: two completely different jobs

In a born-digital PDF, figures are **objects**. You can pull them out using code. Here we use functions from the pymupdf library:
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
display(ShowImage(data=base64.b64decode(render_page(legacy,6))))
""")

md("""
### OCR with layout

`deepseek-ocr` transcribes a page image, and asked the right way it also
reports *where* things are.

When using the ollama library, we need `ollama.generate`, not
`ollama.chat` to get the layout in addition to the text.

Because OCR models are very specialized, they need a very specific prompt. In the case of deepseek OCR, this is documented in their github page. To get the text from a page and its layout, we need the following prompt: `"<image>\\n<|grounding|>Convert the document to markdown."`

Let's do that for one page of the historical PDF:
""")

code("""
legacy_p6 = ollama.generate(
        model=OCR_MODEL,
        prompt="<image>\\n<|grounding|>Convert the document to markdown.",
        images=[render_page(legacy, 6, dpi=100)],
        options={"temperature": 0, "num_predict": 4096},
    )
print(legacy_p6.response)
""")

md("""
Now let's compare this with the original OCR for the same page:
""")

code("""
print("text layer :", get_page_text(legacy, 6).replace(chr(10), " "))
""")

md("""
Now that we know this works, we will define a function that will get the OCR for a given page. This will help us repeat it many times.
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
""")

md("""
### Cropping the figure

The regions are just coordinates in the text. We have to manually parse them out and cut the
image.

The functions below will open a page and cut figures that deepseek OCR identified. Again, we will not go over the details, feel free to reuse the code!
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

for r in regions(legacy_p6.response):
    print(r)
""")

code("""
figures = [r for r in regions(legacy_p6.response) if r[0] in ("image", "image_caption")]
if figures:
    x1 = min(f[1] for f in figures); y1 = min(f[2] for f in figures)
    x2 = max(f[3] for f in figures); y2 = max(f[4] for f in figures)
    crop = crop_region(legacy, 6, ("figure", x1, y1, x2, y2))
    crop.save("figure_huarucus.png")
    display(crop)
""")

md("""
### Reading a page in order

We can pull the text out of a page, and we can pull the figures out of a page.
But we have been pulling them out *separately*, and that throws something away:
which caption belongs to which figure.

Page 5 of the modern paper carries two figures and two captions. Hand a model
all the text and then all the pictures and it sees four things, with no way to
know that FIGURE 3 describes the first picture and FIGURE 4 the second.

So let's read a page the way you would: top to bottom, taking each piece as it
comes.

**ocr_elements()**: splits the OCR model's answer into its regions, in the order it read them. The text after each marker belongs to that marker.

**page_elements()**: one page as a list of `("text", ...)` and `("figure", ...)`, in reading order. On a scan the OCR model has already done the ordering for us. On a born-digital page we sort the text blocks and the image rectangles by where they sit.
""")

code("""
MIN_FIGURE_AREA = 0.03   # smaller than this is a logo, not a figure
MAX_FIGURE_AREA = 0.90   # bigger than this is a scan of the whole page

def ocr_elements(ocr_text):
    \"\"\"The OCR model's regions, in the order it read them: (label, box, text).\"\"\"
    marks = list(re.finditer(
        r"(\\w+)\\s*\\[\\[\\s*(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\s*\\]\\]", ocr_text))
    found = []
    for i, mark in enumerate(marks):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(ocr_text)
        box = ("region", *(int(mark.group(k)) for k in range(2, 6)))
        found.append((mark.group(1), box, ocr_text[mark.end():stop].strip()))
    return found

def page_elements(doc, page, transcript=None, dpi=150):
    \"\"\"A page as ("text", str) and ("figure", Image) pieces, in reading order.\"\"\"
    if transcript is not None:                    # a scan: already in order
        pieces = []
        for label, box, text in ocr_elements(transcript):
            if label == "image":
                pieces.append(("figure", crop_region(doc, page, box, dpi=dpi)))
            elif text:
                pieces.append(("text", text))
        return pieces

    pg = doc[page - 1]                            # born-digital: sort by
    page_area = pg.rect.width * pg.rect.height    # where things sit
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
        # extract_image gives the bytes as stored, whatever the colour depth:
        # line art at 1 bit per pixel comes back as readily as a photograph.
        figure = Image.open(io.BytesIO(doc.extract_image(xref)["image"]))
        placed.append((rects[0].y0, ("figure", figure)))
    return [piece for _, piece in sorted(placed, key=lambda item: item[0])]
""")

md("""
Page 5 of the modern paper, and page 6 of the scan, read in order:
""")

code("""
def show_elements(pieces):
    for kind, value in pieces:
        if kind == "text":
            print(f"  text   {' '.join(value.split())[:72]}")
        else:
            print(f"  figure {value.size[0]}x{value.size[1]}")

print("MODERN, page 5 (figure objects, instant):")
show_elements(page_elements(modern, 5))

print("\\nLEGACY, page 6 (the OCR we already ran, then we cut the figure out):")
show_elements(page_elements(legacy, 6, transcript=legacy_p6.response))
""")

md("""
### Writing the whole document out

Now we can do this for every page of a document and **save the result to a
folder**, so that what the model gets is something you can open and read:

```
processed/deMedeiros2013Zootaxa/
    page-001.md ... page-007.md
    figures/p005-fig01.png
    document.json
```

Each page becomes markdown, in reading order, with its figures linked where
they appeared. A caption sits under its own figure, exactly as in the paper.

This also means we only ever pay for OCR once. A page already written is left
alone, so if the runtime dies you carry on where you stopped — and if one page
came out badly, delete it and only that page is read again.
""")

code("""
def process_pdf(path, out_dir="processed", needs_ocr=False):
    \"\"\"Turn a PDF into a folder of markdown pages you can open and read.\"\"\"
    name = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.join(out_dir, name)
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)

    doc = open_pdf(path)
    for page in range(1, doc.page_count + 1):
        page_file = os.path.join(folder, f"page-{page:03d}.md")
        if os.path.exists(page_file):
            continue                      # done on an earlier run

        transcript = ocr_page(doc, page) if needs_ocr else None
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
            fh.write(f"# page {page}\\n\\n" + "\\n\\n".join(lines) + "\\n")
        print(f"  page {page}: {figures} figure(s)")
    return folder

# The modern paper is quick -- its text is already there.
print("deMedeiros2013Zootaxa:")
modern_folder = process_pdf("example_pdfs/deMedeiros2013Zootaxa.pdf")

# The scan needs an OCR call per page. Start it and keep listening.
print("Marshall1929_AnnMagNatHist:")
legacy_folder = process_pdf("example_pdfs/Marshall1929_AnnMagNatHist.pdf",
                            needs_ocr=True)
""")

md("""
Open the folder icon 📁 in the left sidebar and look inside `processed/`. Click
a `page-*.md` file. That is the document as the model will receive it.

Here is page 5 of the modern paper — two figures, each followed by its own
caption:
""")

code("""
print(open(f"{modern_folder}/page-005.md").read())
""")

md("""
### Your turn — a PDF of your own

Bring in a paper you actually care about and see what comes out of it.

1. Click the **folder icon 📁** in the left sidebar, open the `example_pdfs`
   folder, and drag your PDF into it. Wait for the upload to finish.
2. In the cell below, put your file name in `MY_PDF` variable and the page you want in
   `MY_PAGE`.
3. Run it. You get the PDF's own text layer, a fresh OCR transcription of the
   same page, and any figures on it.

Things worth looking for:

- Is there a text layer at all? A pure scan may give you nothing.
- Where do the text layer and the fresh OCR disagree? Those disagreements are
  where a silently wrong extraction would come from.
- Did the figures come out? If your paper is born-digital and none did, they
  may be *drawn* in the PDF as vector art rather than stored as image objects —
  there is nothing to pull out, so set `MY_NEEDS_OCR = True` and let the OCR
  model find them in the pixels instead.
- Do the captions sit under the right figures in the markdown?
""")

code("""
MY_PDF = "example_pdfs/CHANGE_THIS.pdf"   # your file, uploaded into example_pdfs/

if not os.path.exists(MY_PDF):
    print(f"{MY_PDF} not found. PDFs currently in example_pdfs/:")
    for f in sorted(glob.glob("example_pdfs/*.pdf")):
        print("   ", f)
else:
    mine = open_pdf(MY_PDF)
    print(f"{MY_PDF}: {mine.page_count} pages\\n")

    # Is its own text trustworthy, or does it need re-reading? Look and decide.
    print("--- the PDF's own text, page 1 ---")
    print(get_page_text(mine, 1)[:600] or "(empty -- a pure scan)")

    print("\\n--- the same page, re-read by the OCR model ---")
    print(ocr_page(mine, 1)[:600])

    # Set this once you have looked: True re-reads every page, False trusts
    # the text the PDF already carries.
    MY_NEEDS_OCR = False

    print(f"\\n--- writing processed/{os.path.basename(MY_PDF)[:-4]}/ ---")
    my_folder = process_pdf(MY_PDF, needs_ocr=MY_NEEDS_OCR)
    print("open it in the sidebar and read a page")
""")

md("""
**Recap.** Born-digital PDFs give you clean text and figures as objects. Scans
give you pixels and, often, a text layer that is wrong without saying so. OCR
recovers the text and tells you where each region sits.

The piece that matters most is the *order*. Reading a page top to bottom keeps
each caption with the figure it describes, and writing the result to a folder
means you can look at exactly what the model will be given — before you give it.

Next: turning those pages into data.
""")

# ══════════════════════════════════════════════════════════════ SESSION 3
md("""
---
# Session 3 — Structured extraction

Now we know how to interact with Ollama with 2 kinds of models:

- A multimodal model that can read text and pictures and give us a response.

- A specialized OCR model that reads the image of a page and returns its layout and text.

**What we're doing:**  Now we will take a pdf as input and use an LLM to extract structured data from it.

**Why:** a paragraph is not data. To compare hundreds of morphological descriptions (or pollination papers, or anything you are interested in!), you need
the same fields, with the same names, every time.

We will start by using our OCR model to extract the text from our legacy paper. We will save this text to a file in JSON format, keeping the pagination:
""")

code("""
def load_pages(folder, pages=None):
    \"\"\"A processed folder back as (text, figures), ready to send to a model.

    Ollama attaches images to a message rather than placing them in the text,
    so each figure link becomes a numbered marker where it stood, and the
    figures are handed over in that same order.
    \"\"\"
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

        chunks.append(re.sub(r"!\\[[^\\]]*\\]\\(([^)]+)\\)", mark, text))
    return "\\n\\n".join(chunks), figures

text, figures = load_pages(modern_folder, pages=[5])
print(text)
print("figures handed over:", [f.size for f in figures])
""")

md("""
### How to constrain the output

Let's start by asking the model to constrain the output to a specific format.  Let's try this on page 2 of our modern pdf and ask for JSON format. After finished, let's discuss the response. Did you get the same as I did?
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": "List the species and their traits as JSON.\\n\\n"
                          + get_page_text(modern, 2)}],
    format="json",
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
""")

md("""
### A schema

A better way to constrain the response is to use a SCHEMA.

A [JSON schema](https://en.wikipedia.org/wiki/JSON#Metadata_and_schema) is a dictionary describing the shape you want. It includes which variables you want, what type they are, etc.

Using ollama, we can pass schemas using
`format=` and the model will constrain its response to the schema. Let's start with a simple schema that extracts the name, author, synonyms and minimum and maximum length of each species.

The schema below only makes species name and author mandatory (it is possible the other data is missing). It tells the type to expect for each field (types can be *string*, *number*, *array* (an unnamed list of things, like a Python list) and *object* (a named list of things, like a Python dictionary)).

We use `"additionalProperties": False` to prevent the model from adding additional traits.
""")

code("""
SPECIES_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "author": {"type": "string"},
        "is_new": {"type": "boolean"},
        "max_length": {"type": "number"},
        "n_photos": {"type": "integer"},
        "n_in_photo": {"type": "integer"},
        "synonyms": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["name", "author"],
    "additionalProperties": False
}
""")

md("""
Let's now try our prompt again but with this schema. For that, we just replace `format=json` with `format=SPECIES_SCHEMA`, the variable where we saved our schema.
""")

code("""
reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": '''List the species and their traits as JSON.
               max_length is the maximum length mentioned for the species
               is_new is whether the species is described here
               n_photos is the number of photos used to illustrate the species
               n_in_photo is number of individual beetles in Figure 4 only (ignore for other figures)
               synonyms are all synonyms mentioned in the text

               '''
                          + get_page_text(modern, 2)}],
    format=SPECIES_SCHEMA,
    think=False,
    options={"temperature": 0},
)
print(reply.message.content)
""")

md("""
We can now use the python library `json` to load this json-formatted object as a python object including strings, numbers, lists and dictionaries.

One wrinkle, and it is worth a minute. `json.loads` wants JSON and nothing else.
A model that wraps its answer in a code fence, or explains itself first, or —
as we saw in session 1 — spends its whole reply reasoning and never gets to the
answer, will hand you something `json.loads` refuses at the first character.
So we will use a small wrapper that digs the JSON out and says something useful
when there is none.
""")

code("""
def parse_json(text):
    \"\"\"The JSON in a reply, even if the model wrapped it in something else.\"\"\"
    text = (text or "").strip()
    if not text:
        raise ValueError("the model returned nothing to parse -- it may have "
                         "spent the whole reply reasoning")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"no JSON in the reply: {text[:120]!r}") from None
        return json.loads(text[start:end + 1])

json_result = parse_json(reply.message.content)
json_result
""")

md("""
And we can easily transform JSON into tables using Python:
""")

code("""
pd.DataFrame([json_result])
""")

md("""
### The whole paper, figures included

Now that we have done this for one page, let's do a whole paper. A real extraction takes the whole document, and
a taxonomic paper keeps a lot of its data in the plates — so we send the figures
too. `qwen3.5:9b` is multimodal, which means a message can carry images
alongside its text.

We need to change two things:

**A whole paper has many species, so the schema has to say so.** `SPECIES_SCHEMA`
describes *one* species. Now we need a larger schema that includes `SPECIES_SCHEMA` explicitly saying there will be many species.

**We need a large anough context window.** A page of text
is a few hundred tokens; a whole paper plus its figures is tens of thousands.
Ollama gives you a modest default context. If we do not adjust the context, it will fail but not throw any obvious error: just truncate the output. We need to set  `num_ctx` to a larger number to accomodate the context. For very large papers, you may need a bigger model, more GPU memory, or both.

This cell additionally defines a few convenience function to make out work easier. As before, we will not read their code in details, just know what they do:

**load_pages()** reads a processed folder back: the text with a marker where
each figure stood, and the figures themselves in that same order.

**as_image_data()** decreases resolution of images, if needed, and encodes them as the bytes that LLMs understand.
""")

code("""
PAPER_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {"type": "array", "items": SPECIES_SCHEMA}
    },
    "required": ["species"],
    "additionalProperties": False,
}

def as_image_data(fig, max_side=1024):
    \"\"\"A figure, shrunk if it is huge, encoded the way a model wants it.\"\"\"
    small = fig.copy()
    small.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

WHOLE_PAPER_PROMPT = (
    '''List every species this paper describes, using both the text and the figures.
    name is the species name.
    author is the taxonomic authority for that name -- the person who described it -- not the author of this paper; for a species described as new here, use the paper's own authors.
    max_length is the largest body length in mm given for the species.
    is_new is whether the species is described here
    n_photos is the number of photos used to illustrate the species
    synonyms are all synonyms mentioned in the text
    Do not invent values that are not stated.

    '''
)

text, figures = load_pages(modern_folder)
print(f"sending {len(text)} characters of text and {len(figures)} figures")

reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": WHOLE_PAPER_PROMPT + text,
               "images": [as_image_data(f) for f in figures]}],
    format=PAPER_SCHEMA,
    think=False,
    options={"temperature": 0, "num_ctx": 16384},
)

species = parse_json(reply.message.content)["species"]
print(f"{len(species)} species\\n")
pd.DataFrame(species)
""")

md("""
Read that table against the paper. Because we used a schema, the shape is guaranteed - it will only have what we asked for. But there may be content errors.  Worth checking before you trust any of it:

- Did it find every species, and only species that are really described here?
- Are the lengths the ones in the text, or ones it assembled from nearby
  numbers? A ratio, a scale bar and a body length all look alike to a model.
- Is `author` a taxonomic authority, or did it fall back on "sp. n."? Compare
  with what came back before we spelled the field out in the prompt.
""")

md("""
### Two additional functions

Every extraction we have run has the same shape:
- text (and maybe figures) in
- a schema to constrain the shape of the output
- a prompt to say what we want back
- JSON output
- converstion to a table

We will now define two functions that do all of that.

**`extract()`**: takes the text, the schema and the prompt, and hands back the
parsed JSON. `figures=` adds images to the message. `client=` sends the request
to a different server, which is the only change Session 5 needs to run all of
this on a much bigger model.

**`to_table()`**: pulls the array of records out of the result and makes a
DataFrame of it. The array's name is part of *your* schema, so it is an
argument — `"species"` for `PAPER_SCHEMA`, something else for yours.

Together they turn the cell above into one line:
`to_table(extract(*load_pages(modern_folder)))`
""")

code("""
JSON_ONLY = (
    "\\n\\nReply with JSON only -- no prose, no markdown, no code fence -- "
    "matching exactly this schema:\\n{schema}\\n\\n"
)

def extract(text, schema=PAPER_SCHEMA, prompt=WHOLE_PAPER_PROMPT, figures=None,
            system=None, model=CHAT_MODEL, client=None, num_ctx=16384,
            schema_in_prompt=False):
    \"\"\"Text (and figures) in, structured data out.

    client is None for the server on this machine. Session 5 passes a client
    pointed at Ollama's servers instead, which is the only change needed to
    run any of this on a far bigger model.

    system sets the role the model should answer in, the way session 1 did.

    schema_in_prompt writes the schema into the prompt as well as passing it as
    format=. Locally that is redundant -- format= is enforced. Ollama's cloud
    does not enforce it, so there asking is all you have.
    \"\"\"
    if schema_in_prompt:
        prompt = prompt + JSON_ONLY.format(schema=json.dumps(schema, indent=1))

    # Roughly four characters to a token, and roughly a thousand tokens an
    # image. Ollama drops whatever will not fit without saying so, which is the
    # one failure you cannot see in the answer, so say it here.
    budget = (len(prompt) + len(text)) // 4 + 1000 * len(figures or [])
    if budget > num_ctx:
        print(f"WARNING: about {budget:,} tokens going into a context of "
              f"{num_ctx:,}. The overflow will be dropped silently. Raise "
              f"num_ctx, send fewer pages, or send fewer figures.")

    message = {"role": "user", "content": prompt + text}
    if figures:
        message["images"] = [as_image_data(fig) for fig in figures]

    messages = [{"role": "system", "content": system}] if system else []
    messages.append(message)

    chat = (client or ollama).chat
    reply = chat(
        model=model,
        messages=messages,
        format=schema,
        think=False,
        options={"temperature": 0, "num_ctx": num_ctx},
    )
    return parse_json(reply.message.content)

def to_table(data, key="species"):
    \"\"\"The array of records inside an extraction result, as a table.

    Our schemas wrap the records in one named array -- "species" in
    PAPER_SCHEMA. Pass key= if you named yours something else.

    Anything the schema asks for beside that array -- a fact about the document
    rather than about one record -- becomes a column of its own, repeated down
    the rows, so it travels with the records it belongs to.
    \"\"\"
    rows = pd.DataFrame(data.get(key, []))
    for field, value in data.items():
        if field == key:
            continue
        rows[field] = (json.dumps(value) if isinstance(value, (list, dict))
                       else value)
    return rows
""")

# ════════════════════════════════════════════════════════ TRY IT YOURSELF
md("""
# Try it yourself

Now let's practice. Use the placeholders below to set the schema for each species, the global schema, and the prompt with the details of what you want. Try to extract some information of your choice from the historical pdf.

**Tip:** writing well-formatted schemas can be hard. Use Claude, chatGPT, copilot or whatever you like to double-chekc your schema!
""")

code("""
# ---------------------------------------------------------------- 1. one record
# The fields you want for each thing you are extracting -- the equivalent of
# SPECIES_SCHEMA above. Rename the placeholders and set each type to "string"
# or "number".
MY_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "FIELD_ONE":   {"type": "string"},
        "FIELD_TWO":   {"type": "string"},
        "FIELD_THREE": {"type": "number"},
    },
    "required": ["FIELD_ONE"],       # only the fields that must always be there
    "additionalProperties": False,
}

# ------------------------------------------------------------ 2. the whole answer
# A paper holds many of them, so wrap the record schema in an array -- the
# equivalent of PAPER_SCHEMA above. Rename "records" if you like, and pass the
# same name to to_table() at the bottom.
MY_SCHEMA = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": MY_ITEM_SCHEMA}
    },
    "required": ["records"],
    "additionalProperties": False,
}

# ------------------------------------------------------------------ 3. the prompt
# Say what to extract, and say what every field means. Leaving a field
# unexplained is how we got "sp. n." in the author column earlier.
MY_PROMPT = (
    "DESCRIBE WHAT TO EXTRACT HERE. "
    "FIELD_ONE is ... FIELD_TWO is ... FIELD_THREE is ... "
    "Do not invent values that are not stated in the text.\\n\\n"
)

# ------------------------------------------------- 4. run it on the 1929 paper
# Straight out of the folder we wrote in session 2 -- text in reading order,
# figures where they appeared.
legacy_text, legacy_figures = load_pages(legacy_folder)
print(f"sending {len(legacy_text)} characters and {len(legacy_figures)} figures")

reply = ollama.chat(
    model=CHAT_MODEL,
    messages=[{"role": "user",
               "content": MY_PROMPT + legacy_text,
               "images": [as_image_data(f) for f in legacy_figures]}],
    format=MY_SCHEMA,
    think=False,
    options={"temperature": 0, "num_ctx": 16384},
)

my_records = parse_json(reply.message.content)["records"]
print(f"{len(my_records)} records\\n")
to_table({"records": my_records}, key="records")
""")

md("""
**Recap.** Asking for JSON gets you JSON with unpredictable field names. A
schema passed as `format=` fixes the fields, the types and what is required,
and the model cannot deviate. It cannot tell you whether an answer is *true* —
that check is yours to write.

Next: letting the model decide how to read a page.
""")

# ══════════════════════════════════════════════════════════════ SESSION 4
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

Now let's try with reasoning
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
**Recap.** You described two tools, ran the loop that lets a model use them,
and saw that the same code fails or succeeds depending on whether reasoning is
switched on. Deciding *where* a decision lives — in your code or in the model —
is the judgement that carries over to your own pipelines.

Next: doing this at scale.
""")

# ══════════════════════════════════════════════════ WHERE TO GO FROM HERE
md("""
---
# Where to go from here

That is the whole method: read a document the way it was laid out, decide how
much you can trust what comes out, ask for the fields you want, and check the
answer against the source.

**A bigger model, if you need one.** Everything here fits on a 16 GB GPU
because it had to. The same code runs against something much larger without
changing shape — a bigger model on a machine with more memory, or a hosted one,
which in Ollama means pointing a `Client` at a different host and passing it as
`client=`. We are not doing that today, and one warning if you do: Ollama's
cloud does not enforce `format=`, so the schema you pass is a request rather
than a guarantee and you must check what comes back.

**Things worth doing before you trust a big run**

- Test on a handful of documents you have already scored by hand, and measure
  how often the model agrees with you.
- Keep the source text. An extraction you cannot trace is not evidence.
- Pin your model *and* your Ollama version if you publish a method — open
  weights can be archived alongside your data, and a hosted model cannot.
- Never paste an API key into a notebook. It travels with the file, to your
  collaborators and into your repository.

**Now open `hands-on.ipynb`** and run this on documents of your own.
""")


METADATA = {
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "gpuType": "T4", "include_colab_link": True},
    "accelerator": "GPU",
}


def cell_id(source, taken):
    """A short id that stays the same as long as the cell's text does."""
    base = "c" + hashlib.sha1(source.encode()).hexdigest()[:11]
    ident, n = base, 1
    while ident in taken:          # two cells can hold identical source
        n += 1
        ident = f"{base}-{n}"
    taken.add(ident)
    return ident


def body(cell):
    """A cell's source, ignoring blank lines at either end."""
    return "".join(cell["source"]).strip("\n")


def keep_outputs(new_cells, path=NOTEBOOK):
    """Carry ids, metadata and outputs over from the notebook on disk.

    Matched on the source text, so a cell you did not touch keeps the output it
    already had, and a cell you rewrote comes back empty -- which is honest,
    since its old output no longer belongs to it.
    """
    previous = {}
    if os.path.exists(path):
        with open(path) as fh:
            for cell in json.load(fh).get("cells", []):
                previous.setdefault(body(cell), []).append(cell)

    taken = set()
    for cell in new_cells:
        source = body(cell)
        matches = previous.get(source)
        if matches:
            was = matches.pop(0)
            cell["metadata"] = was.get("metadata", {})
            if cell["cell_type"] == "code":
                cell["outputs"] = was.get("outputs", [])
                cell["execution_count"] = was.get("execution_count")
            ident = was.get("id") or was.get("metadata", {}).get("id")
            if ident and ident not in taken:
                cell["id"] = ident
                taken.add(ident)
                continue
        cell["id"] = cell_id(source, taken)
    return new_cells


def notebook():
    return {"cells": keep_outputs(cells), "metadata": METADATA,
            "nbformat": 4, "nbformat_minor": 5}


def check(path=NOTEBOOK):
    """Report whether the notebook on disk already says what this script says."""
    if not os.path.exists(path):
        print(f"{path} does not exist")
        return False
    with open(path) as fh:
        theirs = [(c["cell_type"], body(c)) for c in json.load(fh)["cells"]]
    ours = [(c["cell_type"], body(c)) for c in cells]
    if theirs == ours:
        print(f"{path} is up to date ({len(ours)} cells)")
        return True

    print(f"{path} and this script have drifted apart "
          f"({len(theirs)} cells on disk, {len(ours)} here)")
    for i, (mine, yours) in enumerate(zip(ours, theirs)):
        if mine != yours:
            print(f"first difference at cell {i}:")
            print(f"  script:   {mine[1][:70]!r}")
            print(f"  notebook: {yours[1][:70]!r}")
            break
    return False


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)

    nb = notebook()
    with open(NOTEBOOK, "w") as fh:
        json.dump(nb, fh, indent=2, ensure_ascii=False)
    kept = sum(1 for c in nb["cells"] if c.get("outputs"))
    print(f"wrote {NOTEBOOK}: {len(nb['cells'])} cells "
          f"({sum(c['cell_type'] == 'code' for c in nb['cells'])} code, "
          f"{sum(c['cell_type'] == 'markdown' for c in nb['cells'])} markdown); "
          f"{kept} kept their outputs")
