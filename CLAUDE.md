# Working in this repository

## Check for manual changes before regenerating a notebook

`workshop.ipynb` and `demo.ipynb` are edited by hand in Colab between sessions.
Anything that writes a notebook programmatically — `build_notebook.py`,
`build_handson.py`, an ad-hoc script — can overwrite that work silently, and a
notebook diff is too noisy to notice it afterwards.

Before running any of them:

1. `git fetch origin <branch>` and `git log HEAD..origin/<branch>` — there may
   be "Created using Colab" commits not pulled yet.
2. `python build_notebook.py --check` — reports whether the script and the
   notebook have drifted, and names the first cell where they differ.
3. If they have drifted, **the notebook is right and the script is stale**.
   Regenerate the script from the notebook, never the other way round.

`build_notebook.py` keeps the outputs of every cell whose source is unchanged,
so rebuilding after a deliberate edit to the script is safe. Rebuilding over
unpulled Colab work is not.

`hands-on.ipynb` is generated from `workshop.ipynb` by `build_handson.py`, so
edits made to its opening block by hand will be lost on the next regeneration.
Change the function in the workshop instead.

## The prose belongs to the author

Fix outright typos and broken links freely. Do not rewrite explanations, append
paragraphs to existing markdown cells, or reword for accuracy without asking
first — including in cells that started out written by Claude and were edited
afterwards.

## Don't commit PDFs

`.gitignore` covers `example_pdfs/*.pdf` and `my_pdfs/`. The two example papers
are already tracked; everything else stays out. An unpublished manuscript
reached this public repository once through an upload folder.
