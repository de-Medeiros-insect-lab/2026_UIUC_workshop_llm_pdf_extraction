# 2026 UIUC workshop: extracting structured data from PDFs with open models

**Date:** 2026-08-12
**Author:** Bruno de Medeiros, with Claude
**Supersedes:** the ESA 2025 workshop notebook (`pdf_data_extraction.ipynb`,
Claude Haiku 4.5 via AWS Bedrock), archived at
[`de-Medeiros-insect-lab/2025_ESA_workshop_claude_pdfs`](https://github.com/de-Medeiros-insect-lab/2025_ESA_workshop_claude_pdfs)
**Target repo:** [`de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction`](https://github.com/de-Medeiros-insect-lab/2026_UIUC_workshop_llm_pdf_extraction)

## Goal

Rewrite the 2025 workshop so that every model runs **free on Google Colab**,
with **open weights**, and no API key distribution. Audience: systematic
entomologists, not software engineers. Format: half a day (~240 min of content).

## Why the 2025 notebook needs more than a version bump

Three things changed, and each one removes or replaces a whole section.

1. **No key distribution.** 2025 ran Claude Haiku 4.5 through AWS Bedrock on
   the instructor's expiring credits, with students emailing for individual
   keys. Open models on Ollama need no keys at all, which deletes the entire
   setup ceremony and the accompanying anxiety about students burning grant
   funds.
2. **Structured output is now enforced, not negotiated.** 2025 spent four
   cells escalating a prompt until Claude reliably emitted JSON (add an
   example, strengthen the system prompt, add XML tags), then repaired the
   result with `json-repair`. Ollama's `format=` parameter takes a JSON Schema
   and constrains decoding, so non-conforming output is not merely discouraged
   but unrepresentable. Both sections are obsolete.
3. **PDFs stop being free.** Claude ingested PDFs natively; open models do not.
   Students must now confront that a born-digital paper has a text layer while
   a scan is pixels — which turns out to be the most valuable thing in the new
   notebook, for reasons below.

## The reproducibility argument (a reversal from 2025)

The 2025 notebook advised that running models locally was not worth the setup
effort. For this task in 2026 that is no longer true, and the strongest reason
is one the old notebook did not make.

The 2025 notebook is its own counter-example: a commented-out cell pins
`claude-sonnet-4-20250514`, a model that no longer exists. A systematist
publishing a methods section can archive open weights alongside their data;
they cannot archive a deprecated API endpoint. Add to that:

- **Data sovereignty** — unpublished specimen records, and localities for
  endangered taxa, that should not transit a commercial API.
- **Zero marginal cost** — iterate on prompts and schemas without watching a
  bill, which matters when developing an extraction pipeline.
- **Pin the runtime too, not just the weights.** Discovered the hard way while
  preparing this: `qwen3.5` and `gemma4` refuse to load on Ollama 0.13.2 and
  require ≥0.32.x. Archiving weights is insufficient if the runtime that can
  read them is unspecified.

## Model stack

One backend (Ollama), two models, both chosen on measured evidence.

| role | model | size | why |
| --- | --- | --- | --- |
| reasoning, extraction, tool use | `qwen3.5:9b` | 6.6 GB | vision + tools + thinking, 256K context, fits a T4 |
| page transcription | `deepseek-ocr` | 6.7 GB | ~4× faster than Qwen warm (6.4 s vs 24.7 s) at equal accuracy; token cost saturates |

**Why Qwen over Gemma 4.** Both were pulled and tested. Gemma 4 is ~2× faster
on structured extraction, but **misread the genus name off a page image** —
returning `Anchyloberus` for `Anchylorhynchus`. For a taxonomy workshop that
is disqualifying. Gemma also ignored an explicit "answer with a number"
instruction where Qwen complied, and produced a third as much reasoning text
(293 vs 935 chars), making it a thinner demo for the thinking section.

**Why a separate OCR model is justified.** Initially it looked unjustified, on
timings later found to be contaminated by model-swap thrashing. Measured
cleanly on p. 267 of Marshall 1929, scoring against 20 taxonomic terms read off
the page by eye:

| source | prompt tokens | terms | sec |
| --- | --- | --- | --- |
| PDF text layer (`get_text()`) | — | 16/20 | 0 |
| `deepseek-ocr` @ 72 dpi | 331 | 15/20 | 12.1 |
| `qwen3.5:9b` @ 72 dpi | 271 | 15/20 | 26.6 |
| **`deepseek-ocr` @ 100 dpi** | **961** | **19/20** | **11.8** |
| `qwen3.5:9b` @ 100 dpi | 475 | 19/20 | 26.5 |
| `deepseek-ocr` @ 150 dpi | **961** | 19/20 | 12.0 |
| `qwen3.5:9b` @ 150 dpi | 1043 | 20/20 | 32.4 |

Two teaching artifacts fall out of this table:

- **DeepSeek-OCR's prompt-token count saturates at 961** while Qwen's climbs
  with resolution. That is optical compression, demonstrated on the workshop's
  own document rather than asserted from a paper. It is the single clearest
  answer to "why does a purpose-built OCR model exist?"
- **Both models beat the PDF's own text layer**, recovering the æ ligatures it
  destroyed. Re-reading the image genuinely repairs corrupt text.

**Calibrated defaults, with measured reasons:**

- **100 dpi.** 72 dpi loses four terms (`Curculionid`, `Parisacalles`,
  `funicle`, `ogival`); 150 dpi buys Qwen one term for ~22% more time and buys
  DeepSeek-OCR nothing at all. Note the sweep timings above include some load
  overhead (models alternated); the warm cost model below is the fair
  head-to-head.
- **`think` is task-shaped, not a global setting.** `think=False` for
  transcription: on defaults Qwen spent 47 minutes emitting 123,055 characters
  of thinking, hit `done_reason='length'`, and returned *zero* content — given
  nothing to reason about, reasoning ruminates. But `think=True` is *required*
  for the agentic loop in Section 7, which does not function without it.
  Transcription is recall; tool choice is judgement. Match the setting to which
  one you have.

**Cost model** (warm, 100 dpi, M1 Max — validate on a T4 before the day):

| operation | cost |
| --- | --- |
| `get_page_text` (PyMuPDF) | ~0 s, free |
| `ocr_page` (DeepSeek-OCR) | 6.4 s |
| Qwen reading a page image | 24.7 s |
| model swap | 8–9 s |

Ollama keeps one model resident and evicts the other, so the agentic loop pays
~8–9 s per alternation. Mitigated by design: the model decides *which* pages
need OCR, then those are transcribed as a batch, so the loop pays two swaps
rather than eight. This is a real engineering constraint, taught as one.

## Why Muse Glimmer is not the workhorse

Meta's Muse Glimmer (30B, Apache 2.0, multimodal, tool-tuned, released
2026-08-10) is the obvious headline model and **cannot run on free Colab**: the
Ollama build is 18 GB against a T4's 16 GB VRAM and ~12 GB system RAM, so it
neither fits nor spills. It is also absent from Ollama Cloud, being positioned
as run-it-yourself.

It keeps a cameo: the instructor's 32 GB M1 Max clears its bar, so it can be
demonstrated live as "what you would run on a workstation."

## Client library

`ollama`-native for the notebook body; an OpenAI-compatible swap demonstrated
in Section 9. Tested against the same server, and the result is more
interesting than expected:

| capability | verdict |
| --- | --- |
| tool calling | equivalent; args arrive as a JSON string needing `json.loads()`, response path is deeper, `tool_call_id` required |
| structured output | **equivalent** — both `.parse(response_format=Model)` and raw `json_schema` with `strict:True` work; the SDK handles the `additionalProperties`/`$defs` massaging |
| **thinking** | **leaks.** `ollama` exposes `r.message.thinking`. The compat endpoint returns it under a bare `reasoning` key inside `model_extra` — not OpenAI's convention, not the `reasoning_content` of DeepSeek/vLLM, and invisible to the typed SDK |

This makes Section 9 concrete and honest: the abstraction holds for tools and
schemas and leaks exactly at reasoning. That predicts what will break when a
student switches providers next year, which is the transferable lesson.

## Documents

**Modern (born-digital):** `deMedeiros2013Zootaxa.pdf` — three
*Anchylorhynchus* species. Retained from 2025. `get_text()` extracts it
perfectly, which is the point: it establishes the easy case.

**Legacy (scanned):** Marshall 1929, "Four new South American Curculionidæ",
*Annals & Magazine of Natural History* ser. 10 vol. 4: 264–270. This document is
better than a plain scan, because **its text layer exists and is silently
corrupt**:

```
image:       new South American Curculionidæ.  267
text layer:  new South American Cureulionidse. 267

Elytra → Elylra      Fig. 20 → Fi.q. 20      265 → 2~5
sub-linear → suh-linear      on Two Fossil Frogs → oa Two Fossil F~'ogs
```

This is the real failure mode in legacy literature and it is nastier than a
blank page. A blank page announces itself — `if not text` routes to OCR. Dirty
OCR announces nothing: `get_text()` returns 2,573 confident characters, the
extraction runs clean, and a species arrives attributed to family
`Cureulionidse`, which matches no taxonomic authority. Nothing errors; the data
is simply wrong, at scale.

So the pipeline gets an honest job: not "is there text?" but **"is this text
trustworthy?"** And because the clean page image is available for comparison,
the improvement can be *shown*. Note that answering that question turns out to
be work for code rather than for the model — see Section 7 in § Running order.

**Prepared file:** `example_pdfs/Marshall1929_AnnMagNatHist.pdf`, 8 pages,
processed for distribution: document info and XMP normalised to the article's own
citation, and content verified clean. The Taylor & Francis cover page is retained
as provenance and as front-matter noise for the extraction lesson.

Page map of the prepared file:

| page | printed | content |
| --- | --- | --- |
| 1 | — | Taylor & Francis cover: citation, DOI, ISSN. Front-matter noise |
| 2 | 264 | **Fish figure captions** from the preceding article (`Itoplichthys langsdorfii`, `Cottus gobio`, `Cyclopterus lumpus`) under a Marshall running header, then Marshall's text begins |
| 3–7 | 265–269 | Marshall's four weevil descriptions. Clean content |
| 8 | 270 | Running header reads `Mr. H. W. Parker on Two Fossil Frogs` while the body is Marshall's weevil description |

Three distinct scope traps, all authentic:

- **Front matter** that has no specimen content but plenty of extractable text.
- **Bleed-over at both ends** — scan boundaries do not respect article
  boundaries, in either direction.
- **A running header that contradicts its own body text** (page 8). A pipeline
  that attributes content by page header will file weevil morphology under a
  frog paper. This is the most valuable trap in the document, because it defeats
  the obvious heuristic rather than an obviously naive one.

One 8-page document therefore teaches skip-the-front-matter,
distrust-the-text-layer, distrust-the-header, and mind-your-scope. Marshall 1929
on South American weevils also sits ~85 years from the 2013 *Anchylorhynchus*
paper in the same family, so the contrast between the two PDFs *is* the "Legacy
to Innovation" arc.

## Running order (~240 min)

| # | Section | min | Status |
| --- | --- | --- | --- |
| 0 | Why this changed since 2025 | 5 | new |
| 1 | Ollama on Colab | 20 | rewritten |
| 2 | Messages + system prompt / role-play — *hands-on 1* | 25 | kept, merged |
| 3 | PDFs are not text: text layer vs pixels | 30 | new |
| 4 | OCR for legacy literature | 30 | new |
| 5 | Thinking | 15 | kept, compressed |
| 6 | Structured output by construction — *hands-on 2* | 35 | rewritten |
| 7 | Tool use & the agentic loop — *capstone* | 45 | new |
| 8 | Scaling up: one-string swap to cloud | 20 | new |
| 9 | Where to go from here | 15 | updated |

### Cut from 2025, and why

- **The JSON persuasion ladder** (4 cells → 1). Obsoleted by constrained
  decoding. One cell retained showing the old approach so students recognise it
  in the wild.
- **`json-repair`** (section → 3 sentences). Valid JSON by construction. It
  survives as "you will need this when your provider lacks schema support."
- **All AWS/Bedrock setup**, plus the commented-out alternative-client cell.
- **The manual PDF upload dance** (download → right-click → sidebar → upload →
  verify filename) → one `wget`. Reclaims ~8 min of room-wide confusion.

≈50 min freed, which funds Sections 3, 4, and 7.

### Retained

- **The role-play framing** (Shanahan et al. 2023) unchanged. It matters *more*
  with 9B models, which need firmer steering than Claude did.
- **Three hands-on cells**, at Sections 2, 6, and 7.
- **Thinking**, now showing raw chain-of-thought rather than Claude's summary —
  a better teaching artifact.

### Added

- **Section 3** — the text-layer/pixels distinction, PyMuPDF, page rendering.
- **Section 4** — DeepSeek-OCR, the dirty-text-layer comparison, the dpi and
  token-saturation calibration.
- **Section 6** — schema *design* replaces prompt-wrangling: what fields, what
  types, enums vs free text, encoding "not applicable", and Pydantic
  validators that catch semantically-wrong-but-syntactically-valid output (a
  5000 mm beetle). More durable knowledge than prompt tricks.
- **Section 7** — the agentic loop, reached as the third beat of a difficulty
  ramp that spans Sections 3, 4, and 7:

  1. **Modern PDF (§3).** The text layer works. No OCR needed. Establishes the
     easy case and what "good" looks like.
  2. **Legacy PDF, OCR by hand (§4).** You call `ocr_page` yourself and watch it
     repair the corrupt text layer. Teaches the mechanism with the decision
     still in your hands.
  3. **Let the agent decide (§7).** Give it both tools and let it route per
     page. Now the decision is delegated — and it can only be delegated because
     of what beats 1 and 2 taught you to check.

  Each beat motivates the next, and nothing is delegated before it is
  understood.

  **Measured, and it hinges on one parameter.** `qwen3.5:9b` escalates correctly
  **only with `think=True`**:

  | setting | behaviour |
  | --- | --- |
  | `think=False` | calls `get_page_text`, notices the corruption, rationalises it as "minor OCR artifacts", answers from the bad text, invents a spelling. Reproduced across four prompt formulations |
  | `think=True` | `get_page_text` → reads `Cureulionidse` → `ocr_page` → answers `Curculionidae` correctly, in ~52 s |

  This makes Section 7 the payoff for the thinking material in Section 5, and it
  refines that lesson into something task-shaped rather than a blanket rule:

  - **`think=False` for transcription.** Nothing to reason about; the model
    ruminates for 47 minutes and returns no content.
  - **`think=True` for judgement.** Choosing tools and assessing whether input
    is trustworthy is precisely reasoning work, and the loop does not function
    without it.

  Knowing *which* kind of task you have is the transferable skill.

  **Also keep the deterministic gate** (`looks_corrupt()`), not as a
  replacement but as the cost comparison. The gate is instant and free; the
  reasoning agent takes ~52 s per page but generalises to decisions you did not
  anticipate. At 4,500 papers you want the gate; while exploring you want the
  agent. Hands-on: students run both on pages 2–8 and compare which pages each
  routes to OCR.

  **Still worth checking before the day:** whether a larger model escalates with
  `think=False`. Untested — `gpt-oss:20b` would not load on Ollama 0.32.9 (stale
  model file). If a cloud model manages it, that is a good Section 8 aside.
- **Section 8** — `qwen3.5:9b` → `qwen3.5:cloud`, identical code. Free tier,
  no credit card, self-issued key in Colab Secrets. This preserves the 2025
  security lesson ("never hardcode keys") without the billing anxiety, and
  makes the scaling argument visibly rather than rhetorically.
  **Caveat to verify:** Ollama Cloud's free tier rations by model "level"
  (level 1 light, up to level 4 heavy) with session limits resetting every 5
  hours and 1 concurrent model. `qwen3.5:cloud` is listed as medium usage.
  Confirm ~20 students can each complete one call inside the free tier before
  committing to this as a live exercise rather than an instructor demo.

## Facilitation notes

- Structured extraction on a full paper runs ~3 min. Hands-on sections must be
  paced so students *start* a run and then discuss while it works, rather than
  waiting in silence. Build this into the facilitator script, not the code.
- Model pulls are ~14 GB total. Each student has their own VM and Colab's
  network is fast, but Section 1 should start the pull and then teach over it.

## Open risks

1. ~~**Colab refuses free GPU runtimes at peak hours**~~ — **Resolved: pair
   up.** Colab may refuse free GPU runtimes between 9am and 6pm PT, precisely
   the workshop window, for a room of ~20 simultaneous requests. Any student
   without a GPU pairs with one who has one. No notebook code, no accounts, no
   degraded path through the sections that matter most — and it is arguably
   better pedagogy for the hands-on blocks. Implications for the notebook:
   Section 1 must detect and clearly report GPU absence (so a student knows
   immediately to pair rather than discovering it three sections later), and the
   facilitator script needs a pairing step. No CPU fallback code is written.
2. **All timings measured on an M1 Max.** Task reliability transfers (identical
   weights) but speed does not (no MLX on Colab, different quant path). One
   validation pass on a real free-tier T4 is required before the day.
3. **Two models against 16 GB VRAM.** 6.6 + 6.7 GB plus KV cache means only
   one stays resident. Batching OCR calls limits this to two swaps, but wants
   confirming on a T4.
4. ~~**Example PDF preparation**~~ — **Resolved.** The legacy PDF has been
   processed for distribution and verified. See § Documents. Note that page
   numbering in all workshop code must reference the prepared 8-page file.

## Out of scope

- Fine-tuning.
- Multi-document corpus workflows (pointed at instead: the ARE 2026 beetle
  review and its code).
- Anything requiring paid compute.
