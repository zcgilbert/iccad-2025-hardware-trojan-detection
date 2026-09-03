# `results/` — what the organisers measured

| Item | What it is |
|---|---|
| [`official_scorecard_cada1033.xlsx`](official_scorecard_cada1033.xlsx) | The organisers' own scorecard: per-case status, precision, recall, F1 and awarded score across all 60 hidden test cases |
| [`contest_submission/`](contest_submission/) | The 60 prediction files exactly as they were scored, plus the run log |

Nothing here was regenerated while tidying this repository. The scorecard is
the file the organisers returned, and the predictions are the ones it scored.

The 60 hidden test cases themselves were never released, so the 121.89 / 180
cannot be re-derived locally — but the arithmetic can be checked, and
[`../docs/results.md`](../docs/results.md) does exactly that.
