# `src/` — the detection pipeline

Three steps, in order. A netlist goes in, a `TROJANED` / `NO_TROJAN` verdict
plus a gate list comes out.

| Step | Module | What it does |
|---|---|---|
| 1 | [`build_graph.py`](build_graph.py) | Parses a netlist and turns it into a feature graph: gates become nodes, 48 features per node, plus a descriptor of the component each gate sits in |
| 2 | [`train.py`](train.py) | Trains the network on those graphs and writes a checkpoint |
| 3 | [`predict.py`](predict.py) | Runs a trained checkpoint over new netlists, applies the structural post-filters, and writes contest-format results |

Supporting modules:

| Module | What it is |
|---|---|
| [`netlist_parser.py`](netlist_parser.py) | A reader for flattened contest Verilog. Standard library only, so the tools in `../tools/` can use it without PyTorch |
| [`dataset.py`](dataset.py) | The label file format, and the one place that resolves which label file belongs to which netlist |
| [`gnn.py`](gnn.py) | The network itself, shared by training and inference so the two cannot drift apart |
| [`augment_netlists.py`](augment_netlists.py) | Rewrites gates into equivalent sub-circuits to enlarge the training set |
| [`public_case_lookup.py`](public_case_lookup.py) | A contest shortcut, off by default. Read its docstring before using it |

**Six of these ran during evaluation** — `netlist_parser`, `dataset`,
`build_graph`, `gnn`, `predict` and `public_case_lookup`, plus the checkpoint
in `../models/`. `train.py` and `augment_netlists.py` are offline: they built
the model, they did not answer questions.

Run any of them with `--help`. Usage examples are in the
[main README](../README.md#quick-start).
