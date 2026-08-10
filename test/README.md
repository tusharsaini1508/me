# Real evaluation data goes here

Add at least twelve rights-cleared receipt photos and a `labels.json` or
`labels.csv` file compatible with `eval.py`. Do not use the synthetic images in
`tests/fixtures/` to claim real-world accuracy metrics; those are only
deterministic regression fixtures.

Once the collection exists, run:

```bash
python eval.py --data test/
```
