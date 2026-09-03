#!/usr/bin/env python3
"""Check that a fresh clone works, before spending time on a real run.

Runs in two tiers:

* **Tier 1 -- standard library only.**  Netlist parsing, label round-tripping
  and the SVG visualiser.  These must pass on any Python 3.9+.
* **Tier 2 -- needs PyTorch and PyTorch Geometric.**  Graph construction and
  inference with the shipped checkpoint.  Skipped with a clear message when
  those are not installed, so tier 1 still tells you something useful.

    python tools/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

PASSED: list = []
FAILED: list = []
SKIPPED: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print("  PASS  " + name + (("  -- " + detail) if detail else ""))
    else:
        FAILED.append(name)
        print("  FAIL  " + name + (("  -- " + detail) if detail else ""))


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print("  SKIP  " + name + "  -- " + reason)


SAMPLE_NETLIST = os.path.join(ROOT, "data", "holdout", "netlists", "design28.v")
SAMPLE_LABEL = os.path.join(ROOT, "data", "holdout", "labels", "result28.txt")
CHECKPOINT = os.path.join(ROOT, "models", "trojan_gnn.pt")


def tier1_stdlib() -> None:
    print("\n[tier 1] standard library only")

    check("sample data present", os.path.isfile(SAMPLE_NETLIST), SAMPLE_NETLIST)
    check("checkpoint present", os.path.isfile(CHECKPOINT), CHECKPOINT)
    if not os.path.isfile(SAMPLE_NETLIST):
        return

    import dataset
    from netlist_parser import parse_netlist

    parsed = parse_netlist(SAMPLE_NETLIST)
    check("netlist parses", parsed.num_gates > 0,
          str(parsed.num_gates) + " gates")
    check("every gate has a known type",
          all(t in ("and", "or", "nand", "nor", "not", "buf", "xor", "xnor", "dff")
              for t in parsed.gate_type.values()))

    edges, forward, reverse = parsed.adjacency()
    check("graph has edges", len(edges) > 0, str(len(edges)) + " edges")
    check("adjacency is consistent",
          sum(len(f) for f in forward) == sum(len(r) for r in reverse) == len(edges))

    is_trojaned, gates = dataset.read_label(SAMPLE_LABEL)
    check("label file reads", is_trojaned and len(gates) > 0,
          str(len(gates)) + " Trojan gates")
    check("labels name real gates", gates <= set(parsed.gate_names))

    # Round-trip a label file through the writer and back.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "roundtrip.txt")
        dataset.write_label(path, sorted(gates))
        again_trojaned, again = dataset.read_label(path)
        check("label round-trip", again_trojaned and again == gates)

        dataset.write_label(path, [])
        clean_trojaned, clean_gates = dataset.read_label(path)
        check("clean label round-trip", not clean_trojaned and not clean_gates)

    check("naming convention: designN -> resultN",
          dataset.output_name_for("design28") == "result28.txt")
    check("naming convention: prefix_netlist -> prefix_results",
          dataset.output_name_for("aes_netlist") == "aes_results.txt")

    # Every augmentation rewrite rule, over its complete truth table.
    sys.path.insert(0, HERE)
    import check_augmentation as aug
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        rules_checked, rule_failures = aug.check_rules()
    check("augmentation rules are function-preserving", not rule_failures,
          str(rules_checked) + " rules proven exhaustively")

    # The visualiser is the other stdlib-only entry point.
    import visualize_netlist as viz

    level = viz.combinational_levels(parsed.num_gates, parsed.gate_type,
                                     forward, reverse)
    check("depth is non-negative and bounded",
          all(0 <= v < parsed.num_gates for v in level),
          "max depth " + str(max(level)))

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "smoke.svg")
        argv = sys.argv
        sys.argv = ["visualize_netlist.py", "--netlist", SAMPLE_NETLIST,
                    "--truth", SAMPLE_LABEL, "--out", out]
        try:
            viz.main()
            import xml.etree.ElementTree as ET
            ET.parse(out)
            check("visualiser emits valid SVG", os.path.getsize(out) > 1000,
                  str(os.path.getsize(out)) + " bytes")
        finally:
            sys.argv = argv


def tier2_torch() -> None:
    print("\n[tier 2] needs PyTorch + PyTorch Geometric")
    try:
        import torch                                     # noqa: F401
        import torch_geometric                           # noqa: F401
    except ImportError as error:
        skip("graph construction", str(error))
        skip("model inference", "same reason")
        print("\n  Install with:  pip install -r requirements.txt")
        return

    import torch
    import build_graph
    from gnn import load_model

    check("feature schema is 48 columns",
          len(build_graph.FEATURE_ORDER) == 48,
          str(len(build_graph.FEATURE_ORDER)) + " features")
    check("feature index agrees with order",
          all(build_graph.FEATURE_INDEX[n] == i
              for i, n in enumerate(build_graph.FEATURE_ORDER)))

    # A small simulation count keeps the smoke test quick.  Real runs must use
    # the training value (1000) -- see README.
    data = build_graph.netlist_to_graph(SAMPLE_NETLIST, SAMPLE_LABEL, sim_count=8)
    data = build_graph.simulate_on_data(data, sim_count=8)

    check("graph built", data.x.size(0) > 0,
          str(data.x.size(0)) + " nodes x " + str(data.x.size(1)) + " features")
    check("feature width matches schema",
          data.x.size(1) == len(build_graph.FEATURE_ORDER))
    check("forward and backward edges are transposes",
          torch.equal(data.edge_index_bw, data.edge_index_fw[[1, 0], :]))
    check("labels align with gates", data.y.size(0) == data.x.size(0))
    check("some gates are labelled Trojan", int((data.y == 1).sum()) > 0,
          str(int((data.y == 1).sum())) + " positives")
    check("subgraph ids index the subgraph table",
          int(data.subgraph_id.max()) < data.subgraph_feat.size(0))
    check("disabled features are zeroed",
          all(float(data.x[:, build_graph.FEATURE_INDEX[n]].abs().sum()) == 0.0
              for n in build_graph.DISABLED_FEATURES))

    if not os.path.isfile(CHECKPOINT):
        skip("model inference", "no checkpoint at " + CHECKPOINT)
        return

    model = load_model(CHECKPOINT, node_in=data.x.size(1),
                       sg_in=data.subgraph_feat.size(1))
    with torch.no_grad():
        logits = model(data)
    check("model runs", logits.shape == (data.x.size(0), 2),
          "logits " + str(tuple(logits.shape)))
    check("logits are finite", bool(torch.isfinite(logits).all()))

    import predict
    prediction = (torch.softmax(logits, dim=1)[:, 1] >= 0.5).long()
    filtered = predict.drop_isolated(prediction, data.edge_index_fw, 1)
    filtered = predict.drop_small_clusters(filtered, data.edge_index_fw, 5)
    check("post-filters only remove positives",
          int(filtered.sum()) <= int(prediction.sum()),
          str(int(prediction.sum())) + " -> " + str(int(filtered.sum())) + " gates")
    check("min-total filter clears a sparse prediction",
          int(predict.enforce_min_total(torch.tensor([1, 0, 0]), 10).sum()) == 0)


def main() -> int:
    print("smoke test for iccad-2025-hardware-trojan-detection")
    print("python " + sys.version.split()[0] + "  |  repo " + ROOT)

    tier1_stdlib()
    tier2_torch()

    print("")
    print(str(len(PASSED)) + " passed, " + str(len(FAILED)) + " failed, " +
          str(len(SKIPPED)) + " skipped")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
        return 1
    print("all good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
