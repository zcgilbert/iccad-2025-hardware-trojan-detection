# `tools/` — utilities that need nothing installed

Every script here runs on the Python standard library alone: no PyTorch, no
PyTorch Geometric, no install step. That is deliberate — a reader should be
able to check that this repository works before deciding to install anything.

| Script | What it does |
|---|---|
| [`smoke_test.py`](smoke_test.py) | Verifies a fresh clone. Runs in two tiers: standard-library checks always, and PyTorch checks when it is available. Start here |
| [`check_augmentation.py`](check_augmentation.py) | Proves the augmentation rewrites preserve circuit function — every rule over its complete truth table, and whole netlists by simulation |
| [`visualize_netlist.py`](visualize_netlist.py) | Draws a netlist as an SVG, colouring each gate by whether it was correctly flagged, missed, or a false alarm |
| [`build_dataset_manifest.py`](build_dataset_manifest.py) | Summarises a generated corpus into one CSV row per netlist — the source of every dataset figure quoted in the documentation |

```bash
python tools/smoke_test.py          # no arguments needed
```
