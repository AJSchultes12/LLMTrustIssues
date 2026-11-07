# Hindsight leakage controls

The model has read about Blockbuster and Enron. Point-in-time inputs are
necessary but not sufficient. Planned controls, each reported as a separate
experimental arm:

1. **Named vs. anonymized.** Run every decision twice: once with real names, once
   with company, people, products, and places replaced (e.g. "a national video
   rental chain", "an online DVD subscription startup"). A large gap between the
   two suggests the model is recalling, not reasoning.
2. **Recognition probe.** After the anonymized run, ask the model to name the
   company. If it can, mark that trial as contaminated.
3. **Pre-cutoff models.** Where available, repeat with older or smaller models
   whose training data predates widespread coverage of the outcome.
4. **Synthetic controls.** Build fictional cases with the same structure but a
   different "correct" answer, to check the model is not just biased toward
   "the bold option" or "the cautious option".
5. **Determinism.** Temperature 0 improves reproducibility but is not fully
   deterministic across runs or providers. Log model version, date, and full
   prompts, and repeat each trial several times.
6. **Obfuscate** Modify Data specifically by a certain factor to ensure no simple signal 
   for the model to trace back to the case beyond the names

## Scoring

Decide the scoring rubric before running anything and commit it, so results
cannot shape the rubric. Compare the model's choice to the actual decision and to
outcomes recorded in `evaluation_only` sources, and grade the reasoning, not just
the pick.
