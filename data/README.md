# Data

Three distinct things live here. They are kept apart because they mean
different things, and mixing them up would make any evaluation meaningless.

## `public_benchmark/` — the organisers' released cases

Six designs from the **official public benchmark**, together with the
**published reference answers**.

```
public_benchmark/netlists/design8.v      <-> solutions/result8.txt
```

Three are Trojaned (8, 38, 48) and three are clean (22, 24, 25). These are real
contest circuits, so they are the right sample to look at if you want to see
what the input format actually looks like at scale.

Because the answers are public, a number measured on these is not evidence of
generalisation. See the note on `src/public_case_lookup.py` in the main README.

## `holdout/` — our own held-out split

Six designs held out of training, with ground truth.

```
holdout/netlists/design28.v  <-> labels/result28.txt
```

These are what `src/predict.py` and `tools/smoke_test.py` exercise, and what the
figures in the main README were rendered from. All six are Trojaned.

## `trojan_definitions/` — the ten contest Trojans

The Trojan library as **RTL**, exactly as the organisers supplied it, plus
[`README.txt`](trojan_definitions/README.txt) describing each one's trigger and
payload. Behaviours span the five classes the contest defines: information
leakage, trigger events, control-flow manipulation, logic modification and
obfuscation.

These are the *source* of the training corpus, not part of it. They are RTL,
not gate-level netlists, so the graph builder will reject them — that is
correct, they have to go through synthesis first (`synthesis/`).

## `dataset_manifest.csv` — the full corpus, summarised

One row for each of the **2,300 generated training netlists**, with its source
pipeline, base design, Trojan type, variant, rewrite rate, label, gate count,
Trojan gate count and size.

The corpus itself (~106 MB of Verilog) is not distributed. The manifest is, so
the dataset's composition can be audited and every statistic quoted in
[`../docs/dataset.md`](../docs/dataset.md) reproduced, without the download.

```bash
# Regenerate the manifest from a raw corpus
python tools/build_dataset_manifest.py --corpus <generated_data> --out manifest.csv
```

## The label file format

Both `labels/` and `solutions/` use the contest's own format — either

```
NO_TROJAN
```

or

```
TROJANED
TROJAN_GATES
g12
g47
END_TROJAN_GATES
```

`src/dataset.py` is the single implementation of reading and writing it.

## A note on file naming

Netlists and their labels are paired by three conventions that all appear in
this project:

| Netlist | Label |
|---|---|
| `<stem>.v` | `<stem>.txt` |
| `designN.v` | `resultN.txt` |
| `<prefix>_netlist.v` | `<prefix>_results.txt` |

The second is **fixed by the contest specification** and cannot be changed,
which is why a resolver is necessary rather than merely convenient. It is
implemented once, in `src/dataset.py`, and every script uses it.
