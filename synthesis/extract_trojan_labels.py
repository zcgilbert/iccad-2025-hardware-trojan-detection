#!/usr/bin/env python3
"""Derive contest-format Trojan labels from a synthesised netlist.

The label problem
-----------------
We know which *RTL* module is the Trojan, but training needs to know which
*gates* are Trojan after synthesis has flattened everything.  Synthesis is what
makes this recoverable: flattening a hierarchical design preserves the
hierarchy inside the instance names, so a gate that came out of the Trojan
module still carries ``trojan`` in its escaped name or drives a net that does.
Labelling therefore happens *before* ``convert_to_contest_format.py`` renames
everything to ``g0, g1, ...``.

Two labelling modes, chosen by filename
---------------------------------------
* **Whole-file** -- when the filename begins with ``trojan``, the design *is* a
  standalone synthesised Trojan module, so every gate is Trojan.  This is how
  the pure-positive part of the corpus is produced.
* **Selective** -- otherwise the design is a host circuit with a Trojan
  injected into it, and only gates whose name or connected nets carry the
  Trojan prefix are labelled.

Gate numbering
--------------
Gates are counted in encounter order and named ``g0, g1, ...`` -- the exact
same walk ``convert_to_contest_format.py`` performs, so index *k* refers to the
same instance in both outputs.  Keep the two traversals in step.

Usage
-----
    python synthesis/extract_trojan_labels.py \\
        --in build/dc/netlists --out build/dc/labels
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import importlib.util

# Reuse the shared label writer instead of re-implementing the file format.
_DATASET_PATH = Path(__file__).resolve().parent.parent / "src" / "dataset.py"
_spec = importlib.util.spec_from_file_location("dataset", _DATASET_PATH)
dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset)

# Gate instantiations as the synthesis tools emit them: the primitives arrive
# escaped (``\and``), flip-flops do not.
GATE_PATTERNS = [
    re.compile(r"^dff\b"),
    re.compile(r"^\\and\b"), re.compile(r"^\\or\b"), re.compile(r"^\\not\b"),
    re.compile(r"^\\nand\b"), re.compile(r"^\\nor\b"),
    re.compile(r"^\\xor\b"), re.compile(r"^\\xnor\b"),
    re.compile(r"^\\buf\b"),
]

_INSTANCE_NAME_RE = re.compile(r"^\S+\s+(\S+)\s*\(")
_PIN_SIGNAL_RE = re.compile(r"\.\w+\s*\(\s*([^)]+)\s*\)")

DEFAULT_TROJAN_PREFIX = "\\trojan"
# Filenames starting with this are standalone Trojan modules: label everything.
WHOLE_FILE_PREFIX = "trojan"


def is_gate_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in GATE_PATTERNS)


def instance_name(line: str) -> str:
    match = _INSTANCE_NAME_RE.match(line)
    return match.group(1) if match else ""


def touches_trojan(instantiation: str, name: str, prefix: str) -> bool:
    """True when the instance name or any connected net carries the prefix."""
    if prefix in name:
        return True
    return any(prefix in signal
               for signal in _PIN_SIGNAL_RE.findall(instantiation))


def scan_gates(path: str) -> List[Tuple[str, str]]:
    """Walk the netlist and return ``(instance_name, full_text)`` per gate.

    Instantiations may span several lines, so a gate is accumulated until its
    closing ``);`` is seen.  The order of this list defines the ``gN``
    numbering.
    """
    with open(path, errors="replace") as handle:
        lines = handle.readlines()

    gates: List[Tuple[str, str]] = []
    inside = False
    buffer = ""
    name = ""

    for line in lines:
        stripped = line.strip()

        if is_gate_line(stripped):
            inside = True
            buffer = stripped
            name = instance_name(stripped)
        elif inside:
            buffer += " " + stripped
        else:
            continue

        if stripped.endswith(");"):
            gates.append((name, buffer))
            inside = False
            buffer = ""
            name = ""

    return gates


def label_netlist(netlist_path: str, output_path: str,
                  trojan_prefix: str = DEFAULT_TROJAN_PREFIX) -> Tuple[int, int]:
    """Write one label file. Returns ``(trojan_gates, total_gates)``."""
    gates = scan_gates(netlist_path)
    stem = Path(netlist_path).stem
    whole_file = stem.startswith(WHOLE_FILE_PREFIX)

    if whole_file:
        trojan_indices = range(len(gates))
    else:
        trojan_indices = [i for i, (name, text) in enumerate(gates)
                          if touches_trojan(text, name, trojan_prefix)]

    trojan_names = ["g" + str(i) for i in trojan_indices]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset.write_label(output_path, trojan_names)
    return len(trojan_names), len(gates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive contest-format Trojan labels from synthesised netlists.")
    parser.add_argument("--in", dest="input", required=True,
                        help="input .v file, or a directory of them")
    parser.add_argument("--out", dest="output", required=True,
                        help="output .txt file, or the destination directory")
    parser.add_argument("--pattern", default="*_flat.v",
                        help="glob applied in directory mode (default: %(default)s)")
    parser.add_argument("--prefix", default=DEFAULT_TROJAN_PREFIX,
                        help="identifier prefix marking Trojan logic "
                             "(default: %(default)s)")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        trojan, total = label_netlist(args.input, args.output, args.prefix)
        print("labelled " + str(trojan) + " of " + str(total) + " gates -> " +
              args.output)
        return

    paths = sorted(Path(args.input).glob(args.pattern))
    if not paths:
        raise SystemExit("no files matched " + args.pattern + " in " + args.input)

    failed = 0
    for path in paths:
        stem = path.stem
        if stem.endswith("_flat"):
            stem = stem[: -len("_flat")]
        output_path = os.path.join(args.output, stem + "_results.txt")
        try:
            trojan, total = label_netlist(str(path), output_path, args.prefix)
            verdict = ("TROJANED, " + str(trojan) + "/" + str(total) + " gates"
                       if trojan else "NO_TROJAN")
            print("[ok]  " + path.name + " -> " + verdict)
        except Exception as error:                       # noqa: BLE001
            print("[err] " + path.name + ": " + str(error))
            failed += 1

    print("")
    print("labelled " + str(len(paths) - failed) + " of " + str(len(paths)) +
          " netlists -> " + args.output)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
