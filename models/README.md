# `models/`

`trojan_gnn.pt` is the trained checkpoint that produced the contest result —
the submitted weights, byte for byte, not a retrained copy.

It is a plain PyTorch `state_dict` for `src/gnn.TrojanGNN`: 48 input features
per gate, 21 per component, three bidirectional graph-convolution layers of
width 64. `src/gnn.load_model()` loads it with `strict=True`, so a mismatch
fails loudly rather than silently producing confident nonsense.

The feature switches at the top of `src/build_graph.py` are part of this
checkpoint's contract. Change one and these weights no longer apply.
