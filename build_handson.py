"""Generate hands-on.ipynb from workshop.ipynb.

    python build_handson.py

The hands-on notebook opens with every import, constant and function the
workshop builds, collected into one block, so a student can run their own
documents through the same pipeline without stepping through five sessions
again. That block is extracted from workshop.ipynb rather than written twice:
change a function there and re-run this, or the two will drift apart.

Everything that is not a definition -- the model calls, the demonstrations, the
variables that only make sense for the two example papers -- is left behind.
"""
import ast
import json

WORKSHOP = "workshop.ipynb"
HANDS_ON = "hands-on.ipynb"

# Constants belonging to the walk-through, not to the pipeline.
SKIP_NAMES = {"QUESTION", "SYSTEM", "MY_PDF", "MY_NEEDS_OCR"}


def definitions(path=WORKSHOP):
    """Every import, function and constant the workshop defines, in order."""
    nb = json.load(open(path))
    kept = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in "".join(cell["source"]).splitlines():
            stripped = line.lstrip()
            # Shell magics are not python. Replace rather than drop them, or an
            # `if` whose whole body was magics stops parsing and the cell's
            # imports go missing with it.
            lines.append(" " * (len(line) - len(stripped)) + "pass"
                         if stripped.startswith(("!", "%")) else line)
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.ClassDef)):
                kept.append(ast.get_source_segment("\n".join(lines), node))
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                # constants only: SHOUTED names, and none of the walk-through's
                if (names and all(n.isupper() for n in names)
                        and not any(n.startswith("MY_") for n in names)
                        and not any(n in SKIP_NAMES for n in names)):
                    kept.append(ast.get_source_segment("\n".join(lines), node))
    return "\n\n".join(kept)


cells = []


def md(text, cid):
    cells.append({"cell_type": "markdown", "metadata": {"id": cid}, "id": cid,
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text, cid):
    cells.append({"cell_type": "code", "metadata": {"id": cid}, "id": cid,
                  "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Hands-on — your own documents

You have seen the whole method. Now run it on papers you care about.

Five things to fill in, in order: **where your PDFs are**, a **system prompt**,
a **user prompt**, and a **schema**. Then run the last cell and read the table.

Nothing here is new. The first code block is every function the workshop built,
collected in one place so you do not have to run five sessions again.
""", "ho-title")

md("""
## Setup

Same as the workshop: Ollama, the two models, this repository. If you have just
come from `workshop.ipynb` in the same runtime, this is quick — the models are
already on disk.
""", "ho-setup-md")

code('''
import glob, os, subprocess, sys, time

IN_COLAB = "google.colab" in sys.modules
REPO = "2026_UIUC_workshop_llm_pdf_extraction"
BRANCH = "main"   # switch to your working branch to test unmerged changes

if IN_COLAB:
    !DEBIAN_FRONTEND=noninteractive apt-get -qq install -y zstd pciutils > /dev/null
    !curl -fsSL https://ollama.com/install.sh | sh
    !pip -q install ollama pymupdf pandas pillow
    if os.path.basename(os.getcwd()) != REPO:   # so this cell is safe to re-run
        if not os.path.isdir(REPO):
            !git clone -q --branch {BRANCH} https://github.com/de-Medeiros-insect-lab/{REPO}.git
        os.chdir(REPO)
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import ollama

def server_ready(timeout=120):
    """Wait until Ollama answers, or give up."""
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

!ollama pull qwen3.5:9b
!ollama pull deepseek-ocr
''', "ho-setup-cd")

md("""
## Everything the workshop built

One block, no surprises: the imports, the constants and every function from
`workshop.ipynb`. Run it and read on.
""", "ho-defs-md")

code(definitions(), "ho-defs-cd")

md("""
## 1. Your documents

Click the folder icon 📁 in the left sidebar and upload **three to five PDFs**
into `my_pdfs/`. Start small — a vague prompt over four thousand documents is an
expensive way to find out the prompt was vague.

Run the cell below. It prints a sample of the text each PDF carries, so you can
judge it yourself — which is session 2's whole point. A scan usually *has* text;
it is just the output of an OCR pass from years ago, and it can be wrong without
saying so. Read the samples and look for the tell: impossible spellings, letters
swapped for punctuation, words run together.
""", "ho-folder-md")

code('''
MY_FOLDER = "my_pdfs"          # where your PDFs are

os.makedirs(MY_FOLDER, exist_ok=True)
my_papers = sorted(glob.glob(os.path.join(MY_FOLDER, "*.pdf")))

if not my_papers:
    print(f"No PDFs in {MY_FOLDER}/ yet. Upload some with the folder icon "
          f"in the sidebar, then run this cell again.")
for path in my_papers:
    doc = open_pdf(path)
    sample = get_page_text(doc, min(2, doc.page_count))[:400]
    print(f"=== {os.path.basename(path)} ({doc.page_count} pages) ===")
    print(sample.strip() or "(no text at all -- this one is certainly a scan)")
    print()
''', "ho-folder-cd")

md("""
### Which ones do you not trust?

List the files whose text looked wrong. Those get re-read page by page with the
OCR model; the rest are taken as they are. A file with no text at all always
gets re-read, whatever you say here.

Leave the list empty if they all looked clean — then this is quick.
""", "ho-rescan-md")

code('''
RESCAN = [
    # "the_1929_scan.pdf",       # <- file names, one per line
]

my_folders = {}
for path in my_papers:
    name = os.path.basename(path)
    doc = open_pdf(path)
    empty = len(get_page_text(doc, min(2, doc.page_count)).strip()) < 100
    needs_ocr = name in RESCAN or empty
    print(f"{name}: {'re-reading every page' if needs_ocr else 'using the stored text'}"
          f"{' (nothing there to use)' if empty and name not in RESCAN else ''}")
    my_folders[path] = process_pdf(path, needs_ocr=needs_ocr)

if my_folders:
    print("\\nwritten to:", *my_folders.values(), sep="\\n  ")
''', "ho-rescan-cd")

md("""
## 2. The system prompt

Who should the model be while it reads? This is the role, not the task —
session 1's point that these are role-playing machines. Be specific about the
field and about care: a model told it is a careful taxonomist behaves
differently from one told nothing.
""", "ho-system-md")

code('''
MY_SYSTEM = (
    "You are a careful DESCRIBE THE SPECIALITY HERE. You read primary "
    "literature and record only what the text actually says. When something "
    "is not stated, you leave it out rather than guessing."
)
''', "ho-system-cd")

md("""
## 3. The user prompt

What do you want out of each document? Say what to extract, and say what every
field of your schema means — a field you leave unexplained is a field the model
will fill with something plausible. This is where "sp. n." ended up in an author
column in session 3.
""", "ho-prompt-md")

code('''
MY_PROMPT = (
    "DESCRIBE WHAT TO EXTRACT HERE.\\n"
    "FIELD_ONE is ...\\n"
    "FIELD_TWO is ...\\n"
    "FIELD_THREE is ...\\n"
    "Do not invent values that are not stated in the text.\\n\\n"
)
''', "ho-prompt-cd")

md("""
## 4. The schema

The fields you want, and their types. Two levels, as in session 3: one record,
then an array of them, because a document holds many.

Types are `"string"`, `"number"`, `"integer"`, `"boolean"`, and `"array"`. Put
in `required` only the fields that must always be there.

**Tip:** writing schemas by hand is fiddly. Paste yours into Claude or ChatGPT
and ask it to check the JSON schema is valid.
""", "ho-schema-md")

code('''
MY_ITEM_SCHEMA = {                 # one record
    "type": "object",
    "properties": {
        "FIELD_ONE":   {"type": "string"},
        "FIELD_TWO":   {"type": "string"},
        "FIELD_THREE": {"type": "number"},
    },
    "required": ["FIELD_ONE"],
    "additionalProperties": False,
}

MY_SCHEMA = {                      # a document holds many of them
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": MY_ITEM_SCHEMA}
    },
    "required": ["records"],
    "additionalProperties": False,
}
''', "ho-schema-cd")

md("""
## 5. Run it

Each processed document goes to the model with your system prompt, your user
prompt and your schema, and the rows come back into one table with a `source`
column saying which paper each came from.

Then **read the table against the papers**. The shape is guaranteed; the
content is not. That has been the whole point of the day.
""", "ho-run-md")

code('''
if not my_papers:
    print(f"Upload PDFs into {MY_FOLDER}/ and run step 1 first.")
else:
    frames = []
    for path, folder in my_folders.items():
        name = os.path.basename(path)
        text, figures = load_pages(folder)
        print(f"{name}: {len(text)} characters, {len(figures)} figures")
        try:
            data = extract(text[:30000], schema=MY_SCHEMA, prompt=MY_PROMPT,
                           figures=figures[:8], system=MY_SYSTEM)
        except Exception as exc:
            print(f"   FAILED -- {exc}")
            continue
        rows = to_table(data, key="records")
        print(f"   {len(rows)} record(s)")
        if len(rows):
            rows.insert(0, "source", name)
            frames.append(rows)

    my_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    my_table.to_csv("my_results.csv", index=False)
    print(f"\\n{len(my_table)} rows, saved to my_results.csv")
    display(my_table)
''', "ho-run-cd")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": [], "gpuType": "T4",
                             "include_colab_link": True},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

with open(HANDS_ON, "w") as fh:
    json.dump(nb, fh, indent=2, ensure_ascii=False)
print(f"wrote {HANDS_ON}: {len(cells)} cells, "
      f"{len(definitions().splitlines())} lines of definitions carried over")
