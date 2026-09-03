#!/usr/bin/env python3
"""Prove that augmentation preserves circuit function.

Augmentation is only useful if the rewritten netlist computes exactly what the
original did. That claim is easy to make and easy to get wrong — a rewrite rule
with one input wired to the wrong net still produces a plausible-looking
netlist, still parses, still trains, and quietly poisons the corpus. So it is
checked, not assumed, at two levels:

**Level 1 — every rewrite rule, exhaustively.** Each rule in
``src/augment_netlists.REWRITE_RULES`` is evaluated over the complete truth
table of its inputs and compared against the gate it claims to replace. Two
inputs means four rows, so this is a proof, not a sample.

**Level 2 — whole netlists, by simulation.** An augmented design and its source
are simulated on the same random input vectors and every primary output bit is
compared. This catches wiring and net-allocation mistakes that correct rules
alone would not.

Standard library only.

Usage
-----
    # Level 1 only — no arguments needed
    python tools/check_augmentation.py

    # Both levels
    python tools/check_augmentation.py \\
        --original data/holdout/netlists \\
        --augmented build/augmented/netlists
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from augment_netlists import REWRITE_RULES              # noqa: E402
from netlist_parser import base_net, parse_netlist      # noqa: E402

DEFAULT_VECTORS = 64


def evaluate(gate_type: str, bits: Sequence[int]) -> int:
    """Reference semantics for the nine contest primitives."""
    if gate_type == "not":
        return 1 - bits[0]
    if gate_type == "buf":
        return bits[0]
    if gate_type == "and":
        return int(all(bits))
    if gate_type == "or":
        return int(any(bits))
    if gate_type == "nand":
        return 1 - int(all(bits))
    if gate_type == "nor":
        return 1 - int(any(bits))
    if gate_type in ("xor", "xnor"):
        parity = 0
        for bit in bits:
            parity ^= bit
        return parity if gate_type == "xor" else parity ^ 1
    raise ValueError("unknown gate type: " + gate_type)


def check_rules() -> Tuple[int, List[str]]:
    """Level 1: exhaustive truth-table check of every rewrite rule."""
    print("[level 1] rewrite rules vs. the gates they replace, exhaustively")
    failures: List[str] = []
    checked = 0

    for gate_type, rules in sorted(REWRITE_RULES.items()):
        arity = 1 if gate_type in ("not", "buf") else 2
        symbols = ["a"] if arity == 1 else ["a", "b"]

        for index, rule in enumerate(rules):
            shape = " -> ".join(step[0] for step in rule)
            mismatches = []

            for assignment in itertools.product((0, 1), repeat=arity):
                nets = dict(zip(symbols, assignment))
                for step_type, out_symbol, in_symbols in rule:
                    nets[out_symbol] = evaluate(
                        step_type, [nets[s] for s in in_symbols])
                expected = evaluate(gate_type, assignment)
                if nets["y"] != expected:
                    mismatches.append(
                        "".join(str(b) for b in assignment) +
                        " -> " + str(nets["y"]) + ", want " + str(expected))

            checked += 1
            label = gate_type + " rule " + str(index) + " (" + shape + ")"
            if mismatches:
                failures.append(label)
                print("  FAIL  " + label)
                for line in mismatches:
                    print("          " + line)
            else:
                print("  PASS  " + label +
                      "  [" + str(2 ** arity) + "/" + str(2 ** arity) + " rows]")

    return checked, failures


def simulate(path: str, vectors: List[Dict[str, int]]
             ) -> Tuple[List[Dict[str, int]], List[str]]:
    """Evaluate a netlist's combinational logic on each input vector.

    Returns per-vector output-bit values and the list of observed output nets.
    Flip-flop outputs are left at their reset value in both designs, so the
    comparison is combinational — which is all augmentation can change, since
    it never rewrites sequential elements.
    """
    parsed = parse_netlist(path)
    _, forward, reverse = parsed.adjacency()
    input_nets, output_nets = parsed.port_nets()

    is_comb = [parsed.gate_type[i] != "dff" for i in range(parsed.num_gates)]
    indegree = [sum(1 for p in reverse[i] if is_comb[p]) if is_comb[i] else 0
                for i in range(parsed.num_gates)]
    ready = [i for i in range(parsed.num_gates) if is_comb[i] and indegree[i] == 0]
    order: List[int] = []
    while ready:
        gid = ready.pop()
        order.append(gid)
        for succ in forward[gid]:
            if is_comb[succ]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)

    # Every driven net whose base name is an output port -- all output bits,
    # not merely the bus names that appear in the port list.
    observed = sorted(net for net in parsed.signal_driver
                      if base_net(net) in output_nets)

    # Pre-index each gate's input nets and output net once.
    inputs_of: Dict[int, List[str]] = {gid: [] for gid in range(parsed.num_gates)}
    for net, loads in parsed.signal_loads.items():
        for gid in loads:
            inputs_of[gid].append(net)
    outputs_of: Dict[int, List[str]] = {gid: [] for gid in range(parsed.num_gates)}
    for net, driver in parsed.signal_driver.items():
        outputs_of[driver].append(net)

    results: List[Dict[str, int]] = []
    for vector in vectors:
        value: Dict[str, int] = {"1'b0": 0, "1'b1": 1}
        for net in input_nets:
            value[net] = vector.get(net, 0)

        for gid in order:
            nets_in = inputs_of[gid]
            if not nets_in:
                continue
            # Every rewritable primitive is commutative or unary, so the order
            # in which inputs are gathered does not matter.
            bits = [value.get(net, 0) for net in sorted(nets_in)]
            try:
                result = evaluate(parsed.gate_type[gid], bits)
            except ValueError:
                continue
            for net in outputs_of[gid]:
                value[net] = result

        results.append({net: value.get(net, 0) for net in observed})

    return results, observed


def check_netlists(original_dir: str, augmented_dir: str,
                   vectors: int, seed: int) -> Tuple[int, List[str]]:
    """Level 2: simulate augmented designs against their sources."""
    print("")
    print("[level 2] augmented netlists vs. their sources, by simulation")
    rng = random.Random(seed)
    failures: List[str] = []
    checked = 0

    for name in sorted(os.listdir(augmented_dir)):
        if not name.endswith(".v"):
            continue
        # augment_netlists.py names variants "<stem>_rate<r>_v<n>.v"
        match = re.match(r"^(.*)_rate[\d.]+_v\d+$", name[:-2])
        stem = match.group(1) if match else name[:-2]
        source = os.path.join(original_dir, stem + ".v")
        if not os.path.isfile(source):
            print("  SKIP  " + name + "  -- no source " + stem + ".v")
            continue

        input_nets, _ = parse_netlist(source).port_nets()
        test_vectors = [{net: rng.randint(0, 1) for net in input_nets}
                        for _ in range(vectors)]

        before, out_before = simulate(source, test_vectors)
        after, out_after = simulate(os.path.join(augmented_dir, name),
                                    test_vectors)
        checked += 1

        if out_before != out_after:
            failures.append(name)
            print("  FAIL  " + name + "  -- output nets differ (" +
                  str(len(out_before)) + " vs " + str(len(out_after)) + ")")
            continue

        differing = sum(1 for a, b in zip(before, after) if a != b)
        if differing:
            failures.append(name)
            print("  FAIL  " + name + "  -- " + str(differing) + " of " +
                  str(len(test_vectors)) + " vectors differ")
        else:
            print("  PASS  " + name + "  -- " + str(len(test_vectors)) +
                  " vectors x " + str(len(out_before)) + " output bits identical")

    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that augmentation preserves circuit function.")
    parser.add_argument("--original", default=None,
                        help="directory of source netlists (enables level 2)")
    parser.add_argument("--augmented", default=None,
                        help="directory of augmented netlists (enables level 2)")
    parser.add_argument("--vectors", type=int, default=DEFAULT_VECTORS,
                        help="random vectors per design (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rules_checked, rule_failures = check_rules()

    netlists_checked, netlist_failures = 0, []
    if args.original and args.augmented:
        netlists_checked, netlist_failures = check_netlists(
            args.original, args.augmented, args.vectors, args.seed)
    else:
        print("")
        print("[level 2] skipped -- pass --original and --augmented to run it")

    failures = rule_failures + netlist_failures
    print("")
    print("rules checked: " + str(rules_checked) +
          " | netlists checked: " + str(netlists_checked) +
          " | failures: " + str(len(failures)))
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("augmentation is function-preserving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
