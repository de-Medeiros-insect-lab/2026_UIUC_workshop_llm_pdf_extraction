# Extracting structured data from PDFs with open models

Workshop materials, UIUC 2026. Everything runs free on Google Colab with
open-weight models — no API keys, no credit card.

## The two notebooks

Start with the demo, then take it apart.

| | |
| --- | --- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction/blob/main/demo.ipynb) | **`demo.ipynb`** — one function, three arguments, a table out. Ten minutes, mostly model download. Run this first. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction/blob/main/workshop.ipynb) | **`workshop.ipynb`** — the day itself. Five sessions building all of it from an empty cell. |

The demo imports `extract_folder()` from `pdf_extraction.py`, which is the
finished pipeline in about two hundred lines. `workshop.ipynb` deliberately
does not import it: every piece is written out in the notebook, so you can see
and change it.

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

Requires Ollama ≥0.32 — earlier versions cannot load these models.

1. Install [Ollama](https://ollama.ai)
2. Pull the models:
   ```bash
   ollama pull qwen3.5:9b
   ollama pull deepseek-ocr
   ```
3. Start the Ollama server:
   ```bash
   ollama serve
   ```
4. In another terminal, set up the Python environment:
   ```bash
   mamba env create -f environment.yml
   mamba activate uic_workshop_2026
   ```
5. Open the notebook:
   ```bash
   jupyter notebook workshop.ipynb
   ```

## The 2025 version

This workshop previously used Anthropic's Claude:
[2025_ESA_workshop_claude_pdfs](https://github.com/de-Medeiros-insect-lab/2025_ESA_workshop_claude_pdfs).
