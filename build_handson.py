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


md('''
<a href="https://colab.research.google.com/github/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction/blob/main/hands-on.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
''', "view-in-github")

md('''
# Hands-on — your own documents

Let's now apply the methods we learned in the workshop to papers and data you care about.

We will need the following information:
- **where your PDFs are**
- a **system prompt** setting the LLM role
- a **user prompt** with details on WHAT to extract
- a **schema** constraining the shape of the output.

Then run the last cell and read the table.

The first code block is every function the workshop built,
collected in one place so you do not have to run five sessions again.
''', "ho-title")

md('''
## Setup

We will start by setting up Ollama
''', "ho-setup-md")

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

md('''
## Everything the workshop built

Here we have all thhe imports, the constants and every function from
`workshop.ipynb`.
''', "ho-defs-md")

code(definitions(), "ho-defs-cd")

md('''
## 1. Your documents

Click the folder icon 📁 in the left sidebar and upload **two to five PDFs**
into `my_pdfs/`. Start small: we now want to try things out, not write a paper.

Run the cell below. It prints a sample of the text of each PDF so we can judge the OCR quality.
''', "ho-folder-md")

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

md('''
### Which ones do you not trust?

List the files whose text looked wrong. Those will get re-read page by page with the
OCR model.

Leave the list empty if they all looked good.

We could use the AI equipped with tools to decide, but human intelligence will be faster in this case with just a few PDFs that you chose.
''', "ho-rescan-md")

code(r'''
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
    print("\nwritten to:", *my_folders.values(), sep="\n  ")
''', "ho-rescan-cd")

md('''
## 2. The system prompt

Who should the model be while it reads? This is the role, not the task —
session 1's point that these are role-playing machines. Be specific about the
field and about care: a model told it is a careful taxonomist behaves
differently from one told nothing.
''', "ho-system-md")

code("""
# @title Default title text
MY_SYSTEM = (
    '''You are a careful DESCRIBE THE SPECIALITY HERE.
    You read primary literature and record only what the text actually says.
    When something is not stated, you leave it out rather than guessing.'''
)
""", "ho-system-cd")

md('''
## 3. The user prompt

What do you want out of each document? Say what to extract, and say what every
field of your schema means.

If you do not explain a field, you will let the LLM judge, and it can make more mistakes.
''', "ho-prompt-md")

code("""
MY_PROMPT = (
    '''DESCRIBE WHAT TO EXTRACT HERE.
    FIELD_ONE is ...
    FIELD_TWO is ...
    FIELD_THREE is ...
    Do not invent values that are not stated in the text.'''
)
""", "ho-prompt-cd")

md('''
## 4. The schema

Now constrain the fields you want, and their types. Use two levels, as in session 3: one for records, with possibly many records per document, and another for document.

Valid types are `"string"`, `"number"`, `"integer"`, `"boolean"`, `"object"` and `"array"`. Put
in `required` only the fields that must always be there.

Use `"additionalProperties": True` only if you want to allow the model to find properties that you did not list explicitly.

**Tip:** writing schemas by hand is fiddly. You can start the broad strokes by hand and tehn get Claude or ChatGPT
to check whether JSON schema is valid and fix it.
''', "ho-schema-md")

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

MY_SCHEMA = {                      # a document holds many records, and document-level attributes
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": MY_ITEM_SCHEMA},
        "DOCUMENT_ATTRIBUTE_ONE": {"type": "number"}
    },
    "required": ["records"],
    "additionalProperties": False,
}
''', "ho-schema-cd")

md('''
## 5. Run it

Each processed document will go to the model with your system prompt, your user
prompt and your schema, and the rows come back into one table with a `source`
column saying which paper each came from. The workhorse is the function `extract()` defined above, which includes the LLM call.

Then we will **read the table against the papers**. Because we have a schema, the shape of the output will be guaranteed. But the content may be wrong. How would you measure how much to trust?
''', "ho-run-md")

code(r'''
if not my_papers:
    print(f"Upload PDFs into {MY_FOLDER}/ and run step 1 first.")
else:
    frames = []
    for path, folder in my_folders.items():
        name = os.path.basename(path)
        text, figures = load_pages(folder)
        print(f"{name}: {len(text)} characters, {len(figures)} figures")
        try:
            data = extract(text, schema=MY_SCHEMA, prompt=MY_PROMPT,
                           figures=figures, system=MY_SYSTEM)
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
    print(f"\n{len(my_table)} rows, saved to my_results.csv")
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
