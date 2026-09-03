# Results — case-by-case analysis

Everything here is derived from two artefacts kept in this repository:

* [`results/official_scorecard_cada1033.xlsx`](../results/official_scorecard_cada1033.xlsx)
  — the organisers' scorecard: precision, recall, F1 and awarded score for each
  of the 60 hidden cases.
* [`results/contest_submission/`](../results/contest_submission/) — the
  predictions actually submitted.

Together these pin down the ground truth of the hidden set exactly: the
scorecard says whether each verdict was right, and the submission says what the
verdict was, so the true label of every case follows.

## Headline

| Metric | Value |
|---|---|
| Test cases | 60 (hidden) |
| Executed without crash or timeout | 60 / 60 |
| Correct Trojaned / clean verdict | 50 / 60 |
| Sum of per-gate F1 bonus | 21.89 |
| **Total** | **121.89 / 160** (76.2 %) |

### Why the denominator is 160, not 180

The scorecard reports a nominal maximum of 180 — three points across 60 cases
— but no submission could have reached it. The F1 bonus is only awarded on
designs that are genuinely Trojaned, so a clean design caps at the two verdict
points. With the hidden set's actual composition, 40 Trojaned and 20 clean
(derived below), the most any entry could score was

    40 × 3  +  20 × 2  =  160

so 160 is the denominator used throughout this document.

## Two different questions, two very different answers

The single "mean F1 = 0.365" figure conflates two tasks the contest scores
separately. Split apart, the picture is much sharper.

### Question 1 — is this design Trojaned at all?

Reconstructed confusion matrix over all 60 cases:

|  | truth: Trojaned | truth: clean |
|---|---:|---:|
| **said Trojaned** | 35 | 5 |
| **said clean** | 5 | 15 |

The hidden set is 40 Trojaned and 20 clean designs.

| | |
|---|---|
| Accuracy | **83.3 %** (50 / 60) |
| Precision | **87.5 %** (35 / 40) |
| Recall | **87.5 %** (35 / 40) |
| F1 | **0.875** |

Detection at the design level is the strong part of this system, and the errors
are symmetric — five missed Trojans and five false alarms — rather than
skewed one way.

### Question 2 — *which* gates are the Trojan?

Of the 35 Trojaned designs correctly flagged, 34 earned a gate-level bonus:

| | |
|---|---|
| Median F1 among those 34 | **0.769** |
| Best case (case 22) | precision 1.000, recall 1.000, F1 **1.000** |
| Cases scoring above 2.9 / 3 | 9 |
| Detected but zero gate overlap | 1 (case 21) |

So localisation is not uniformly poor: when the Trojan is found, the median
design has three quarters of its gates correctly identified. The headline mean
of 0.365 is low mostly because it averages in the 20 clean designs — where F1
is defined as zero — and the 10 wrong verdicts.

Best cases:

| Case | Precision | Recall | F1 | Score |
|---|---:|---:|---:|---:|
| 22 | 1.000 | 1.000 | 1.000 | 3.000 / 3 |
| 36 | 0.985 | 0.977 | 0.981 | 2.981 |
| 38 | 1.000 | 0.951 | 0.975 | 2.975 |
| 34 | 0.927 | 1.000 | 0.962 | 2.962 |
| 16 | 0.914 | 1.000 | 0.955 | 2.955 |

## Failure mode 1 — missed Trojans (cases 2, 3, 26, 30, 37)

Five Trojaned designs were reported clean. Each costs the full 3 points.

The pattern is Trojan logic that does not form a **separable component**. The
subgraph branch is the model's strongest structural signal, and it rests on the
assumption that inserted logic attaches weakly to its host. A Trojan threaded
through existing logic — sharing nets, reusing host gates — presents no
distinctive component, and the node branch alone does not catch it.

`design37` in the repository samples is a local instance of exactly this: 23
Trojan gates, none found. The visualiser renders it in
[the README](../README.md#what-the-model-actually-sees).

## Failure mode 2 — false alarms (cases 42, 43, 44, 48, 58)

Five clean designs were reported Trojaned. The minimum-total filter exists to
suppress precisely this — it discards any prediction with fewer than 10
surviving gates — and it is clearly not sufficient: these five produced clusters
large and connected enough to survive all three filters.

## Failure mode 3 — right design, wrong gates (case 21)

Case 21 was correctly flagged as Trojaned, but the flagged gate set had **zero
overlap** with the real Trojan. It banks the 2 verdict points and no bonus. A
single case, but a reminder that a correct verdict does not imply the model
found the right thing.

## The precision/recall trade in the post-filters

Seventeen cases achieved **recall 1.000** — every Trojan gate found. In nine of
them (4, 5, 6, 10, 11, 12, 13, 14, 15) precision was **below 0.20**: the whole
Trojan was caught along with several times as much host logic.

Those nine still contributed **19.8 points**. That is the scoring function
working as designed, and it is why the threshold and filters sit on the recall
side: a correct verdict is worth 2 points and per-gate F1 at most 1, so a
low-precision hit is worth far more than a cautious miss. Averaged over the
cases that scored, precision is 0.42 against recall 0.62.

## Where the missing points are

Against the achievable ceiling of 160, the shortfall is 38.11 points, and it
decomposes exactly:

| Source | Points lost |
|---|---:|
| 5 missed Trojans (3 each) | 15.00 |
| 5 false alarms (2 each) | 10.00 |
| Gate-level F1 shortfall on the 35 detected designs | 13.11 |
| **Total** | **38.11** |

The verdict-level failures are the bigger bucket — **25 of the 38 points** —
and they are also the more tractable target. Closing the gate-level gap
entirely, i.e. perfect localisation on every design already detected, would
recover only 13.11. Both verdict failure modes trace back to the same
weakness: Trojans that do not decompose into a separable component. That needs
a new structural signal, not threshold retuning.

## Reproducing these numbers

```bash
python src/predict.py --netlists <netlists> --labels <labels> \
                      --model models/trojan_gnn.pt --out build/predictions
```

`predict.py` implements the contest's scoring function directly
(`score_case()`), so running it against a labelled directory reproduces the
same arithmetic the organisers applied.
