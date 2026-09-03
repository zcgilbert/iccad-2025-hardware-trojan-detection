"""Generate logically-equivalent netlist variants for data augmentation.

The problem this solves
-----------------------
The contest's Trojan library is small: ten reference Trojans, injected into a
handful of host designs.  A network trained on that alone learns the *syntax*
of those particular gate arrangements -- "this exact NAND-XOR shape is a
Trojan" -- rather than the structural properties that generalise.  The hidden
test set uses designs nobody has seen.

So each gate is rewritten into an equivalent sub-circuit with some
probability: a NAND becomes an AND feeding a NOT, an XOR becomes four NANDs,
and so on.  The circuit computes exactly the same function, its structure
changes substantially, and the Trojan labels are carried across to whichever
new gates replace a labelled one.  The result is many structurally distinct
presentations of the same logic, which forces the model to key on topology
rather than on a memorised gate pattern.

Correctness
-----------
* Every rewrite rule below is a Boolean identity; the design's function is
  preserved by construction.
* A rewritten Trojan gate contributes *all* of its replacement gates to the
  label set, so no Trojan logic loses its label.
* Flip-flops are never rewritten -- there is no combinational identity for
  them, and touching sequential elements would change the state encoding.

Usage
-----
    python src/augment_netlists.py --netlists data/holdout/netlists \
                                   --labels   data/holdout/labels \
                                   --out      build/augmented \
                                   --rate 0.2 --variants 3
"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import dataset

PRIMITIVE_GATES = {"and", "or", "nand", "nor", "xor", "xnor", "not", "buf", "dff"}
# Flip-flops have no combinational equivalent, so they are never rewritten.
REWRITABLE_GATES = PRIMITIVE_GATES - {"dff"}

# Each rule is an explicit netlist template: a list of
# ``(gate_type, output, (input, ...))`` steps.  Symbolic names are resolved
# when the rule is applied:
#
#   "a", "b"  the original gate's inputs        ("b" only for 2-input gates)
#   "y"       the original gate's output net    (always the last step)
#   "t0", …   fresh internal nets
#
# Writing the wiring out in full rather than inferring it from gate order is
# deliberate.  An earlier version chained gates positionally -- each gate
# consuming the next unused input, then the previous gate's output -- which
# happens to be correct for two- and three-gate rules and silently wrong for
# longer ones.  ``tools/check_augmentation.py`` verifies every rule against the
# gate it replaces by exhaustive truth table.
Step = Tuple[str, str, Tuple[str, ...]]

REWRITE_RULES: Dict[str, List[List[Step]]] = {
    # NAND(a,b) = NOT(AND(a,b)) = OR(NOT a, NOT b)
    "nand": [
        [("and", "t0", ("a", "b")), ("not", "y", ("t0",))],
        [("not", "t0", ("a",)), ("not", "t1", ("b",)), ("or", "y", ("t0", "t1"))],
    ],
    # NOR(a,b) = NOT(OR(a,b)) = AND(NOT a, NOT b)
    "nor": [
        [("or", "t0", ("a", "b")), ("not", "y", ("t0",))],
        [("not", "t0", ("a",)), ("not", "t1", ("b",)), ("and", "y", ("t0", "t1"))],
    ],
    # XNOR(a,b) = NOT(XOR(a,b)) = ab + a'b'
    "xnor": [
        [("xor", "t0", ("a", "b")), ("not", "y", ("t0",))],
        [("and", "t0", ("a", "b")),
         ("not", "t1", ("a",)), ("not", "t2", ("b",)),
         ("and", "t3", ("t1", "t2")),
         ("or", "y", ("t0", "t3"))],
    ],
    # XOR(a,b) = a'b + ab' = the classic four-NAND construction
    "xor": [
        [("not", "t0", ("a",)), ("and", "t1", ("t0", "b")),
         ("not", "t2", ("b",)), ("and", "t3", ("a", "t2")),
         ("or", "y", ("t1", "t3"))],
        [("nand", "t0", ("a", "b")),
         ("nand", "t1", ("a", "t0")), ("nand", "t2", ("b", "t0")),
         ("nand", "y", ("t1", "t2"))],
    ],
    # AND(a,b) = NOT(NAND(a,b)) = NOR(NOT a, NOT b)
    "and": [
        [("nand", "t0", ("a", "b")), ("not", "y", ("t0",))],
        [("not", "t0", ("a",)), ("not", "t1", ("b",)), ("nor", "y", ("t0", "t1"))],
    ],
    # OR(a,b) = NOT(NOR(a,b)) = NAND(NOT a, NOT b)
    "or": [
        [("nor", "t0", ("a", "b")), ("not", "y", ("t0",))],
        [("not", "t0", ("a",)), ("not", "t1", ("b",)), ("nand", "y", ("t0", "t1"))],
    ],
    # Single-input gates.  These rules are correct but never fire: `not` and
    # `buf` instances are passed through untouched (see rewrite_netlist), so
    # inverter chains stay recognisable in the augmented output.
    "not": [
        [("nand", "y", ("a", "a"))],
        [("nor", "y", ("a", "a"))],
    ],
    "buf": [
        [("not", "t0", ("a",)), ("not", "y", ("t0",))],
    ],
}
UNARY_GATES = {"not", "buf"}

# A gate instantiation, either  nand g7(y, a, b);  or  not g8(y, a);
_BINARY_RE = re.compile(r"^(\w+)\s+(g\d+)\(([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\);")
_UNARY_RE = re.compile(r"^(\w+)\s+(g\d+)\(([^,]+)\s*,\s*([^)]+)\);")
_GATE_HEAD_RE = re.compile(r"^(\w+)\s+\w+\s*\(")
_DECLARATION_RE = re.compile(r"^(input|output|inout|wire|reg)\b")
_BODY_START_RE = re.compile(
    r"^(and|or|not|nand|nor|xor|xnor|buf|assign|always|dff)\b")


def extract_header(text: str) -> str:
    """Return the module declaration and its port/wire declarations.

    The body is regenerated from scratch, so only the interface is reused.
    """
    lines: List[str] = []
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        if not started:
            if re.match(r"^module\s", stripped):
                started = True
                lines.append(line)
            continue
        if _BODY_START_RE.match(stripped):
            break
        if _DECLARATION_RE.match(stripped) or stripped:
            lines.append(line if line.startswith("    ") else "    " + line)
    lines.append("endmodule")
    return "\n".join(lines)


def count_gates(lines: Sequence[str]) -> Tuple[int, int]:
    """Return ``(total_gates, rewritable_gates)`` for a netlist."""
    total = rewritable = 0
    for line in lines:
        match = _GATE_HEAD_RE.match(line.strip())
        if match and match.group(1) in PRIMITIVE_GATES:
            total += 1
            if match.group(1) in REWRITABLE_GATES:
                rewritable += 1
    return total, rewritable


def _next_free_ids(text: str) -> Tuple[int, int]:
    """Find unused ``n<k>`` net and ``g<k>`` gate ids to allocate from."""
    nets = [int(s[1:]) for s in re.findall(r"\bn\d+\b", text)]
    gates = [int(s[1:]) for s in re.findall(r"\bg\d+\b", text)]
    return max(nets, default=0) + 1, max(gates, default=0) + 1


def rewrite_netlist(source_lines: List[str], trojan_gates: Set[str],
                    rate: float, rng: random.Random
                    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Rewrite a fraction of the gates into equivalent sub-circuits.

    Returns ``(body_lines, new_wires, trojan_gate_mapping)`` where the mapping
    records, for each rewritten Trojan gate, the replacement gates that inherit
    its label.
    """
    joined = "".join(source_lines)
    next_net, next_gate = _next_free_ids(joined)

    body: List[str] = []
    new_wires: List[str] = []
    trojan_mapping: Dict[str, List[str]] = {}

    for line in source_lines:
        stripped = line.strip()

        # Flip-flops and single-input gates written in the compact style are
        # copied through untouched.
        if stripped.startswith(("dff ", "not ", "buf ")):
            body.append(stripped)
            continue

        binary = _BINARY_RE.match(stripped)
        unary = None if binary else _UNARY_RE.match(stripped)
        if not binary and not unary:
            continue

        if binary:
            gtype, gate_name, output, in_a, in_b = binary.groups()
            inputs = [in_a, in_b]
            passthrough = gtype + " " + gate_name + "(" + output + " ," + in_a + " ," + in_b + ");"
        else:
            gtype, gate_name, output, in_a = unary.groups()
            inputs = [in_a]
            passthrough = gtype + " " + gate_name + "(" + output + " ," + in_a + ");"

        if gtype not in REWRITABLE_GATES or rng.random() > rate:
            body.append(passthrough)
            continue

        rule = rng.choice(REWRITE_RULES[gtype])

        # Resolve the template's symbolic names against this instance: the
        # original inputs and output stay as they are, so the surrounding
        # netlist is untouched, and each "tN" becomes a fresh internal net.
        net_of: Dict[str, str] = {"y": output, "a": inputs[0]}
        if len(inputs) > 1:
            net_of["b"] = inputs[1]
        for _, out_symbol, _ in rule:
            if out_symbol != "y":
                net_of[out_symbol] = "n" + str(next_net)
                new_wires.append(net_of[out_symbol])
                next_net += 1

        replacement_names: List[str] = []
        for gate_type, out_symbol, in_symbols in rule:
            name = "g" + str(next_gate)
            pins = [net_of[out_symbol]] + [net_of[s] for s in in_symbols]
            body.append(gate_type + " " + name + "(" + " ,".join(pins) + ");")
            replacement_names.append(name)
            next_gate += 1

        if gate_name in trojan_gates:
            trojan_mapping[gate_name] = replacement_names

    return body, new_wires, trojan_mapping


def assemble(header: str, body: List[str], new_wires: List[str]) -> str:
    """Splice the rewritten body and the extra wire declarations into a module."""
    lines = [line + "\n" for line in header.splitlines()]

    if new_wires:
        for i, line in enumerate(lines):
            if re.match(r"^\s*wire\b", line):
                trimmed = line.strip().rstrip(";")
                lines[i] = ("    " + trimmed + ", " +
                            ", ".join(sorted(set(new_wires))) + ";\n")
                break
        else:
            # No wire declaration to extend, so add one before the body.
            insert_at = next(i for i, line in enumerate(lines)
                             if "endmodule" in line)
            lines.insert(insert_at,
                         "    wire " + ", ".join(sorted(set(new_wires))) + ";\n")

    insert_at = next(i for i, line in enumerate(lines) if "endmodule" in line)
    indented = ["    " + gate + "\n" for gate in body]
    return "".join(lines[:insert_at] + indented + lines[insert_at:])


def remap_labels(trojan_gates: Set[str],
                 mapping: Dict[str, List[str]]) -> List[str]:
    """Carry Trojan labels onto the gates that replaced them."""
    updated: Set[str] = set()
    for gate in trojan_gates:
        updated.update(mapping.get(gate, [gate]))
    return sorted(updated)


def augment(netlist_dir: str, label_dir: Optional[str], out_dir: str,
            rate: float, variants: int, seed: int) -> int:
    """Generate ``variants`` rewritten copies of every netlist. Returns the count."""
    netlist_out = os.path.join(out_dir, "netlists")
    label_out = os.path.join(out_dir, "labels")
    os.makedirs(netlist_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    rng = random.Random(seed)
    written = 0

    for stem, path in dataset.iter_netlists(netlist_dir):
        with open(path, errors="replace") as handle:
            source_lines = handle.readlines()

        total, rewritable = count_gates(source_lines)
        header = extract_header("".join(source_lines))
        _, trojan_gates = dataset.read_label(dataset.find_label(label_dir, stem))

        print(stem + ": " + str(total) + " gates, " + str(rewritable) +
              " rewritable, " + str(len(trojan_gates)) + " Trojan")

        for variant in range(variants):
            body, new_wires, mapping = rewrite_netlist(
                source_lines, trojan_gates, rate, rng)
            name = stem + "_rate" + str(rate) + "_v" + str(variant)

            with open(os.path.join(netlist_out, name + ".v"), "w") as handle:
                handle.write(assemble(header, body, new_wires))

            # Labels are only meaningful where the source had them.
            if trojan_gates:
                dataset.write_label(os.path.join(label_out, name + ".txt"),
                                    remap_labels(trojan_gates, mapping))

            print("  -> " + name + ".v (+" + str(len(new_wires)) + " nets)")
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate equivalent netlist variants for augmentation.")
    parser.add_argument("--netlists", required=True)
    parser.add_argument("--labels", default=None,
                        help="label directory; variants inherit remapped labels")
    parser.add_argument("--out", required=True,
                        help="output root; netlists/ and labels/ are created here")
    parser.add_argument("--rate", type=float, default=0.2,
                        help="probability of rewriting each gate (default: %(default)s)")
    parser.add_argument("--variants", type=int, default=1,
                        help="variants per design (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed, so a run is reproducible (default: %(default)s)")
    args = parser.parse_args()

    if not 0.0 <= args.rate <= 1.0:
        raise SystemExit("--rate must be between 0 and 1")

    written = augment(args.netlists, args.labels, args.out,
                      args.rate, args.variants, args.seed)
    print("")
    print("wrote " + str(written) + " variant(s) to " + args.out)


if __name__ == "__main__":
    main()
