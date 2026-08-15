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

**Hardware caveat:** every timing in this document and in the notebook (the
~3 min above, the capstone estimate below, the per-model numbers in Section
4) was measured on an Apple M1 Max laptop, not a Colab T4. We have never
measured this workshop on a T4. Numbers on the day will differ — plan pacing
around "start it and talk," not around the specific seconds.

**Section 7's capstone (Hands-on 3) is the biggest timing risk in the whole
workshop.** It runs nine `think=True` tool loops back to back, alternating
between the chat model and the OCR model on almost every page. On the M1 Max
this notebook's own run took about seven and a half minutes (individual pages ranged from 12 to 97 seconds); expect longer on a shared T4. Section 7 only
has 45 minutes total — start the capstone cell early and keep teaching over
it, the same as the model pull in section 1, or it will eat the section.

## Section 2: the model is confidently wrong

In section 2, the notebook asks the model to identify insect structures. When
you run it, the model will make a confident claim about a taxonomic group — it
may say something is in Staphylinoidea (rove beetles) when your workshop
audience knows it belongs elsewhere. This is pedagogically load-bearing: the
single most persuasive argument for grounding every answer in the source PDF is
watching the model state something confidently and wrongly about the audience's
own specialty. Pause and discuss the error. The model's output may differ on
the day, so read whatever answer comes back and check it live rather than
expecting that exact error.

## Send this out in advance

Students have already been asked to install the **Ollama desktop app**.
Session 1.1 starts in that app rather than the notebook, so it is worth one
reminder — anyone without it is locked out of the opening exercise, and there
is no Colab fallback for that part.

The one thing still worth asking for ahead of time:

- A free account at <https://ollama.com> (no card needed) and an API key from
  <https://ollama.com/settings/keys>, kept somewhere they can paste from.
  Session 5 walks them through adding it to Colab Secrets; they only need to
  arrive with the key. Twenty simultaneous signups would cost you ten minutes
  of that session.

`deepseek-r1:1.5b` does **not** need pulling in advance — it is about a
gigabyte and lands in a minute, so let them pull it live in Session 1.1. That
is also a useful thing for them to watch happen once.

Everything in Sessions 1–4 runs locally and needs no key, so a student who
turns up without one can still follow the whole morning.

## Before you begin

- Ask everyone to set Runtime → Change runtime type → T4 **before** running
  anything, and to raise a hand if they cannot get a GPU.
- Pair up anyone without a GPU immediately. Colab refuses free GPUs at busy
  times. There is deliberately **no CPU code path** in this notebook — a student
  without a GPU cannot run it and must pair up.
- The free cloud tier is metered per account. Keep Session 5's cloud calls
  small — one short extraction each is plenty to make the point.

## Section 7 is a staged failure

The point of the capstone is that the model *fails* first. Do not fix it early.
With `think=False`, let students watch it read the corrupt text, notice the
corruption, talk itself out of it, and invent a spelling. Then flip the single
argument to `think=True` and watch it call `ocr_page` and get the answer right.
The lesson is that `think` is task-shaped: reasoning is wasted on transcription
and indispensable for judgement.

Then make the "no regex version of this" argument deliberately (that is the
notebook's own heading for it, in the section right after the two tool-loop
cells): there is no regex that does this job. Students will reach for one —
building it is the obvious idea — so tell
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
