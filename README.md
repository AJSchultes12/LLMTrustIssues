# LLMTrustIssues

Could a temperature-0 language model have made better calls than the executives
behind some of history's biggest business failures? This repo builds a
point-in-time dataset for each decision, so a model only sees what the
decision-makers could have known.

## Cases

| Case | Type | Decision points |
|---|---|---|
| Blockbuster | Strategy failure | BB-1 Netflix $50M offer (2000) · BB-2 Kill late fees (Dec 2004) · BB-3 Keyes and Total Access (Jul 2007) |
| Enron | Governance / fraud failure | EN-1 LJM1 approval and conflict waiver (Jun 1999) · EN-2 LJM2 approval (Oct 1999) · EN-3 Lay's response to the Watkins memo (Aug 2001) |

Each case lives in `cases/<name>.yaml` with dates, cutoffs, the options to present
to the model, the actual decision, and the sources allowed for that decision.

## The cutoff rule

**Every source must be dated strictly before the decision point's `cutoff`.**
SEC filings are filtered on filing date, emails on their UTC send date.
Anything listed under `evaluation_only` (the Powers Report, later 10-Ks,
bankruptcy records) is ground truth for scoring and never goes into a prompt.

`tests/test_pipeline.py` enforces this: a cutoff later than its decision date
fails the build.

## Getting the data

```bash
pip install -r requirements.txt

# SEC requires a User-Agent with your contact info
export SEC_USER_AGENT="LLMTrustIssues research Your Name you@example.com"

# Preview what each decision point gets, then download
python scripts/fetch_edgar.py cases/blockbuster.yaml --dry-run
python scripts/fetch_edgar.py cases/blockbuster.yaml
python scripts/fetch_edgar.py cases/enron.yaml

# Enron email corpus (~1.7 GB), then cut it per decision point
curl -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
python scripts/filter_enron_emails.py enron_mail_20150507.tar.gz --cutoff 1999-06-27 --out data/raw/enron/EN-1/emails.jsonl
python scripts/filter_enron_emails.py enron_mail_20150507.tar.gz --cutoff 1999-10-11 --out data/raw/enron/EN-2/emails.jsonl
python scripts/filter_enron_emails.py enron_mail_20150507.tar.gz --cutoff 2001-08-14 --custodians lay-k skilling-j --out data/raw/enron/EN-3a/emails.jsonl
python scripts/filter_enron_emails.py enron_mail_20150507.tar.gz --cutoff 2001-08-22 --custodians lay-k skilling-j --out data/raw/enron/EN-3b/emails.jsonl

pytest -q
```

Downloads go to `data/raw/<case>/<decision>/` with a `manifest.json` recording
each file's URL and SHA-256, so others can confirm they have the same inputs.

## Known gaps to fill by hand

- **Board materials.** The June 28 1999 Enron board agenda and the Watkins memo
  are DOJ trial exhibits (links in `cases/enron.yaml`). Download them manually and
  add them to the decision folder.
- **BB-1 date.** Accounts disagree on whether the Netflix meeting was early or
  September 2000. The cutoff uses the earlier date; confirm before running.
- **BB-1 Netflix data.** Netflix was private in 2000, so its numbers come from
  founders' later recollections. They are labeled as such in the case file.
- **Contemporaneous news and analyst reports** (e.g. on late fees in 2004) are
  not included. If you add them, keep them dated before the cutoff.

## Hindsight leakage

Modern models already know how these stories end. Filtering the inputs does not
fix that. See `docs/leakage.md` for the controls this project plans to test.

## Data ethics

The Enron emails are real messages from real people, most of whom did nothing
wrong. Use them for this research only. In April 2026, forensic researchers also
raised doubts about the authenticity of some messages in the corpus; treat any
single email as evidence, not fact.
