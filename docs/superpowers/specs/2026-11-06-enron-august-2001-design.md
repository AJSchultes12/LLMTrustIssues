# Enron, August 2001: model vs. executives (design)

Status: approved to proceed on 2026-11-06 (the user asked to get started after
reviewing the design and its open calls). Nothing in this document has been
built yet; the implementation plan comes next.

## Intent

Test whether a language model, given only what Enron's leadership could have
known in August 2001, decides differently from the executives, and whether any
difference reflects better judgment or only the absence of personal incentives.

The repo already builds point-in-time datasets (`cases/*.yaml`, `scripts/`).
This design adds everything downstream of the data: disguise, prompting,
running, probing for recall, and scoring. It covers one case (Enron) and two
decision points. Blockbuster and the other Enron decisions are out of scope.

## Why this decision point

August 2001 is where Enron's leadership faced a visible choice between
disclosure and reassurance. It is also probably the most contaminated case
available: the Skilling resignation and the Watkins memo are heavily
documented, and a model may simply remember the ending. So the design treats
leakage as something to measure and bound, not something to eliminate.

## Decision points

Both are new entries in `cases/enron.yaml` (or a sibling file). EN-3 as it
exists today is close to EN-3b.

| ID | Decision | Date | Cutoff |
|---|---|---|---|
| EN-3a | Public message and guidance to employees and investors when Skilling resigns | 2001-08-14 | 2001-08-14 (strictly before; verify against sources) |
| EN-3b | Response to the Watkins memo | 2001-08-22 | 2001-08-22 (as in current EN-3) |

The model sits in Lay's seat in both. (Assumption: this is the decision-maker
for both points. Confirm each against sources when writing the case entries.)

Options per decision point are listed in the case file, as today. EN-3a options
are not yet written; the builder drafts them, then the user reviews them.
The EN-3a options must include at least one that discloses uncertainty and one
that reassures, so the choice maps onto the question being asked.

**Incentive framing.** The model is told the decision-maker holds a large
Enron stock position. The exact wording is fixed in the rubric before runs, and
a no-stake variant is run as a comparison so the effect of the stake is visible.
(The no-stake variant is the builder's call; it was not discussed.)

## Inputs

- Only sources dated strictly before the cutoff: EDGAR filings via
  `scripts/fetch_edgar.py`, emails via `scripts/filter_enron_emails.py`, and the
  hand-downloaded board materials and Watkins memo.
- Nothing listed under `evaluation_only` ever enters a prompt. A test enforces
  this (see Tests).
- Email caveats in `cases/enron.yaml` apply: real people, and some messages in
  the corpus have disputed authenticity. Emails are evidence, not fact.

## Leakage controls

Every decision point is run at three disguise levels:

1. **Named.** Real material. Baseline.
2. **Names stripped.** Companies, people, products, places replaced with generic
   descriptions.
3. **Fingerprints removed.** Level 2 plus: all dollar figures scaled by one
   factor, all dates shifted by one offset, and distinctive structures and
   instruments renamed. Ratios and relationships between facts are preserved so
   the decision still works. The scale factor and offset are fixed per case and
   recorded.

The Watkins memo is rewritten at levels 2 and 3. Its phrasing changes, its
substance does not. The user approved this explicitly.

Further controls:

- **Recognition probe.** Before any run, a separate model session sees the level
  2 or 3 prompt and is asked which company, which year, and how it ends. A
  trial the model identifies is marked contaminated and reported separately.
- **Leakage scan.** Model reasoning is searched for outcome terms that were not
  in its inputs (for example "bankruptcy", "restatement", "Andersen", plus
  anything in `evaluation_only`). Hits are flagged.
- **Synthetic twin.** A fictional case with the same structure where the
  whistleblower is wrong and the accounting is sound. If the model escalates in
  both the real and twin versions, it is leaning cautious, not showing insight.
- **Determinism.** Temperature 0, but log model version, date, and the full
  prompt, and repeat each trial several times (see Left to the builder).
- A pre-2002 model arm is judged unavailable for this case. A small or older
  model may be used as a weak check. This is the author's belief, not verified.

Reported result is a bound: how much of the model's behavior is attributable to
recall, and how much survives disguise. The write-up does not claim leakage is
gone.

## Scoring

The rubric is written and committed before any model run, as `docs/leakage.md`
already requires.

- Compare the model's choice to the executives' actual choice.
- Compare it to outcomes documented in `evaluation_only` sources, mainly the
  Powers Report.
- Grade the reasoning, not just the pick.
- Score against documented facts. "The executives lied" is not a rubric
  criterion; whether the model's stated reasoning matches the documented record
  is.

## Goals

- A reproducible pipeline from case file to scored result, with every input
  hash-recorded.
- A defensible estimate of how much recall explains the model's choices.
- A clear comparison of model vs. executive decisions at both points.

## Non-goals

- Proving or disproving that the executives committed fraud.
- Covering Blockbuster or other Enron decisions in this pass.
- Generalizing to "AI is better than executives".

## Anti-goals (failures even if the code works)

- Any post-cutoff or `evaluation_only` content reaching a prompt.
- A rubric adjusted after seeing results.
- A conclusion that the model is "objective" without the synthetic twin
  showing it does not just default to caution.
- Treating a result as meaningful when the recognition probe says the model
  identified the case.

## Constraints

- Real people's emails and names: use for this research only.
- SEC fair-access rate limits and the User-Agent requirement (already handled in
  `fetch_edgar.py`).
- The Enron tarball is about 1.7 GB and is gitignored; manifests are kept.

## Decided by the user

- Pilot is the Enron August 2001 window.
- Disguise levels, recognition probe, leakage scan, and synthetic twin as above.
- Rewriting the Watkins memo in disguised form is acceptable.
- The rubric is committed before runs.

## Author's calls (not discussed; please check)

- Splitting into EN-3a and EN-3b.
- The model sits in Lay's seat, with a disclosed stock position.
- The no-stake comparison variant.
- Scoring primarily against the Powers Report.
- The triggering event for each decision (Skilling's resignation for EN-3a, the
  memo for EN-3b) is stated in a short scenario brief at the top of the prompt,
  not treated as a dated source. The strict cutoff therefore does not remove it.
  The brief is disguised along with everything else.

## Budget and models

Decided by the user: total spend on model calls is capped at about $20.
Local compute is a laptop with a 4GB GPU, so large models run elsewhere.

Author's calls (please check):

- Development and pipeline testing use free hosted tiers (for example Gemini,
  OpenRouter `:free` models, Groq), where rate limits are acceptable.
- Final scored runs use one large open-weight model through either a
  pay-per-token API or a short RunPod session, whichever is cheaper at the time
  and available with the weights identified. Roughly $15 is for final runs and
  $5 is held in reserve.
- The user has about $8 of Claude API credit. It funds a Claude arm: one
  Claude model for scored runs and the recognition probe. Which Claude model
  (the cheaper Haiku tier or the stronger Sonnet tier) is chosen after the
  per-call cost is estimated, so the credit covers the full trial grid. The
  plan checks whether the API accepts a temperature setting for the chosen
  model. The $8 is on top of the $20 cap (user's answer), so the Claude
  arm is capped separately at $8 and total spend is about $28.
- One small model runs locally as an extra arm, mainly to see whether a smaller
  model recognizes the case less often.
- The runner estimates cost before each batch and stops at the cap. A RunPod
  pod, if used, is stopped when runs finish.
- Exact model IDs and prices are checked when the plan is written, because free
  tiers and prices change. Every trial logs the exact model ID, provider, and
  date, since hosted models can change underneath the experiment.

## Left to the builder

- Anonymizer implementation and per-case rule format.
- Prompt template layout and how long filings are summarized or truncated to
  fit context.
- Run logging format and storage location.
- Number of repeats per trial, within the budget.
- Project layout for the new code (for example `src/` vs. `scripts/`).
- Tests for the new code, extending `tests/test_pipeline.py`.

## Tests required

- No `evaluation_only` text, and no source dated on or after a cutoff, appears
  in any generated prompt.
- After disguise level 2 or 3, none of the real names or listed identifiers
  remain in the prompt.
- Scale and shift transforms preserve ratios and date gaps.
