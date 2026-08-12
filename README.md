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
