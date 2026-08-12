# Facilitator notes

## Timing (~240 min)

| # | Section | min |
| --- | --- | --- |
| 0 | Why this changed since 2025 | 5 |
| 1 | Ollama on Colab | 20 |
| 2 | Messages + system prompt — hands-on 1 | 25 |
| 3 | PDFs are not text | 30 |
| 4 | OCR for legacy literature | 30 |
| 5 | Thinking | 15 |
| 6 | Structured output — hands-on 2 | 35 |
| 7 | Tool use & the agentic loop — capstone | 45 |
| 8 | Scaling up to a cloud model | 20 |
| 9 | Where to go from here | 15 |

## Pacing

Long-running cells are the main hazard. Structured extraction on a full paper
takes ~3 min. **Have students start the run, then talk over it** — do not let
the room sit in silence watching a spinner. The same applies to the ~14 GB model
pull in section 1: start it, then teach section 0 while it downloads.

## Before you begin

- Ask everyone to set Runtime → Change runtime type → T4 **before** running
  anything, and to raise a hand if they cannot get a GPU.
- Pair up anyone without a GPU immediately. Colab refuses free GPUs at busy
  times, and CPU is far too slow for the OCR and vision sections.

## Section 7 is a staged failure

The point of the capstone is that the model *fails* first. Do not fix it early.
With `think=False`, let students watch it read the corrupt text, notice the
corruption, talk itself out of it, and invent a spelling. Then flip the single
argument to `think=True` and watch it call `ocr_page` and get the answer right.
The lesson is that `think` is task-shaped: reasoning is wasted on transcription
and indispensable for judgement.

Then make the stage-3 argument deliberately: there is no regex that does this
job. Students will reach for one — building it is the obvious idea — so tell
them what happened when this workshop tried. The gate needed URL stripping,
hand-tuned tilde and ampersand patterns, and a threshold correction, and it was
still fitted to one scanner's damage. It failed silently on pages it had not
been tuned against, which is exactly the failure mode Section 3 warns about.

Land the two honest options: usually you already know which of your PDFs are
scans, so just OCR those; and when you genuinely have a mixed pile, ask a model
that can reason.

## Live demo opportunity

Muse Glimmer (30B) will not run on free Colab — 18 GB against a T4's 16 GB. If
you have a 32 GB machine to hand, running it locally makes a good contrast for
section 8.
