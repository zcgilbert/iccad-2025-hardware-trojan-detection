#!/usr/bin/env python3
"""Summarise the generated training corpus into a single CSV manifest.

The full corpus produced by the three generation pipelines is far too large to
distribute (~106 MB of Verilog).  Rather than ship it, we ship this manifest:
one row per generated netlist, recording where it came from and how big it is,
so the dataset can be inspected and audited without downloading it.

Every row is derived by parsing the actual files -- nothing is hand-written --
so the statistics quoted in README.md and docs/dataset.md are reproducible by
re-running this script against the raw corpus.

Usage
-----
    python tools/build_dataset_manifest.py --corpus path/to/generated_data \
                                           --out data/dataset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

# A gate instantiation always begins with one of the nine contest primitives.
GATE_LINE_RE = re.compile(r"^(and|or|nand|nor|not|buf|xor|xnor|dff)\b", re.IGNORECASE)

# Which sub-directories of the corpus belong to which generation pipeline, and
# where the matching label files live.  `None` means the pipeline emits no
# label files because every design in it is Trojan-free by construction.
PIPELINES: Dict[str, Tuple[str, Optional[str]]] = {
    "design_compiler": (
        "trojaned_data/dc_pipeline_netlists",
        "trojaned_data/dc_pipeline_results",
    ),
    "genus": (
        "trojaned_data/genus_pipeline_netlists",
        "trojaned_data/genus_pipeline_results",
    ),
    "augmentation": (
        "trojaned_data/data_augmentation_pipeline_netlists",
        "trojaned_data/data_augmentation_pipeline_results",
    ),
    "trojan_free": (
        "trojan_free_data/trojan_free_netlists",
        None,
    ),
}

# Filename conventions produced by each pipeline.  They differ because each
# pipeline was written at a different stage of the project; the manifest is
# where that history gets normalised into one schema.
NAME_PATTERNS = [
    # design_compiler:  top3_trojan7_12_netlist.v
    re.compile(r"^(?P<base>top\d+)_trojan(?P<trojan>\d+)_(?P<variant>\d+)_netlist$"),
    # genus:            trojan7_12_g_netlist.v  or  trojan7_12_g_no_and_netlist.v
    re.compile(r"^trojan(?P<trojan>\d+)_(?P<variant>\d+)_g(?P<flavour>_no_and)?_netlist$"),
    # augmentation:     design24_0.2_0_netlist.v
    re.compile(r"^(?P<base>design\d+)_(?P<rate>[\d.]+)_(?P<variant>\d+)_netlist$"),
    # trojan-free:      TjFree_aes_g_0.2_0_netlist.v  or  TjFree_aes_netlist.v
    re.compile(r"^TjFree_(?P<base>.+?)(?:_(?P<rate>[\d.]+)_(?P<variant>\d+))?_netlist$"),
]


def count_gates(path: str) -> int:
    """Count primitive-gate instantiations in a flattened contest netlist."""
    total = 0
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            if GATE_LINE_RE.match(line.strip()):
                total += 1
    return total


def read_label(path: str) -> Tuple[str, int]:
    """Return (label, trojan_gate_count) for one contest-format result file.

    The contest format is either a single line NO_TROJAN, or TROJANED followed
    by the gate names between TROJAN_GATES and END_TROJAN_GATES.
    """
    with open(path, "r", errors="replace") as handle:
        lines = [ln.strip() for ln in handle if ln.strip()]
    if not lines or lines[0] != "TROJANED":
        return "NO_TROJAN", 0
    if "TROJAN_GATES" not in lines or "END_TROJAN_GATES" not in lines:
        return "TROJANED", 0
    start = lines.index("TROJAN_GATES") + 1
    end = lines.index("END_TROJAN_GATES")
    return "TROJANED", end - start


def parse_name(stem: str) -> Dict[str, str]:
    """Decompose a generated filename into its provenance fields."""
    for pattern in NAME_PATTERNS:
        match = pattern.match(stem)
        if match:
            return {k: (v or "") for k, v in match.groupdict().items()}
    return {}


def find_label_file(label_dir: str, stem: str) -> Optional[str]:
    """Locate the result file matching a netlist, honouring all three naming rules."""
    candidates = [stem + ".txt"]
    if stem.endswith("_netlist"):
        candidates.append(stem[: -len("_netlist")] + "_results.txt")
    design = re.match(r"^design(\d+)$", stem)
    if design:
        candidates.append("result" + design.group(1) + ".txt")
    for name in candidates:
        path = os.path.join(label_dir, name)
        if os.path.isfile(path):
            return path
    return None


def collect(corpus: str) -> List[Dict[str, object]]:
    """Walk every pipeline directory and build one manifest row per netlist."""
    rows: List[Dict[str, object]] = []
    for pipeline, (netlist_rel, label_rel) in PIPELINES.items():
        netlist_dir = os.path.join(corpus, netlist_rel)
        if not os.path.isdir(netlist_dir):
            print("[skip] " + pipeline + ": " + netlist_dir + " not found")
            continue
        label_dir = os.path.join(corpus, label_rel) if label_rel else None

        for filename in sorted(os.listdir(netlist_dir)):
            if not filename.endswith(".v"):
                continue
            stem = filename[:-2]
            path = os.path.join(netlist_dir, filename)
            fields = parse_name(stem)

            label, trojan_gates = "NO_TROJAN", 0
            if label_dir:
                label_path = find_label_file(label_dir, stem)
                if label_path:
                    label, trojan_gates = read_label(label_path)
                else:
                    label = "UNLABELLED"

            rows.append({
                "file": filename,
                "pipeline": pipeline,
                "base_design": fields.get("base", ""),
                "trojan_type": fields.get("trojan", ""),
                "variant": fields.get("variant", ""),
                "rewrite_rate": fields.get("rate", ""),
                "label": label,
                "gates": count_gates(path),
                "trojan_gates": trojan_gates,
                "bytes": os.path.getsize(path),
            })
    return rows


def summarise(rows: List[Dict[str, object]]) -> None:
    """Print the aggregate numbers quoted in the project documentation."""
    print("")
    print("total netlists: " + str(len(rows)))
    header = ("pipeline".ljust(18) + "files".rjust(7) + "trojaned".rjust(10) +
              "clean".rjust(8) + "unlabelled".rjust(12) +
              "gates(sum)".rjust(12) + "gates(median)".rjust(15))
    print(header)
    for pipeline in PIPELINES:
        subset = [r for r in rows if r["pipeline"] == pipeline]
        if not subset:
            continue
        gates = sorted(int(r["gates"]) for r in subset)
        median = gates[len(gates) // 2] if gates else 0
        print(pipeline.ljust(18) +
              str(len(subset)).rjust(7) +
              str(sum(1 for r in subset if r["label"] == "TROJANED")).rjust(10) +
              str(sum(1 for r in subset if r["label"] == "NO_TROJAN")).rjust(8) +
              str(sum(1 for r in subset if r["label"] == "UNLABELLED")).rjust(12) +
              str(sum(gates)).rjust(12) +
              str(median).rjust(15))

    trojaned = [r for r in rows if r["label"] == "TROJANED"]
    if trojaned:
        trojan_gate_total = sum(int(r["trojan_gates"]) for r in trojaned)
        gate_total = sum(int(r["gates"]) for r in trojaned)
        ratio = trojan_gate_total / max(1, gate_total)
        print("")
        print("Trojan gates in Trojaned designs: " + str(trojan_gate_total) +
              " of " + str(gate_total) + " gates (" + format(ratio, ".4%") + ")")
        print("  -> this is the class imbalance the focal loss in train.py exists to handle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CSV manifest of the generated training corpus.")
    parser.add_argument("--corpus", required=True,
                        help="root of the raw generated corpus (generated_data/)")
    parser.add_argument("--out", default="data/dataset_manifest.csv",
                        help="destination CSV path")
    args = parser.parse_args()

    rows = collect(args.corpus)
    if not rows:
        raise SystemExit("no netlists found -- check --corpus")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarise(rows)
    print("")
    print("wrote " + str(len(rows)) + " rows -> " + args.out)


if __name__ == "__main__":
    main()
