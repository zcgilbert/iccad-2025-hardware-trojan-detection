"""A minimal reader for flattened contest netlists.

Deliberately dependency-free: only the Python standard library.  Graph building
needs PyTorch, but reading a netlist does not, and keeping the two apart means
the inspection tools in ``tools/`` run anywhere -- no CUDA, no PyG, no install.

Scope
-----
The contest guarantees one flat module built exclusively from named primitive
instances, so this handles the whole accepted grammar and nothing more.  It is
not a general Verilog front end and does not pretend to be: anything outside
that grammar is ignored rather than half-understood.

Accepted forms::

    nand g7(y, a, b);                                   # output first
    not  g8(y, a);
    dff  g9(.RN(r), .SN(s), .CK(clk), .D(d), .Q(q));    # named pins
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

GATE_TYPES = ["and", "or", "nand", "nor", "not", "buf", "xor", "xnor", "dff"]
COMBINATIONAL_TYPES = set(GATE_TYPES) - {"dff"}

# Pins that carry a value *into* a flip-flop.  Q is the output.
DFF_INPUT_PINS = ("D", "CK", "RN", "SN", "RST", "SET")

_GATE_HEAD_RE = re.compile(
    r"^(?:and|or|nand|nor|not|buf|xor|xnor|dff)\b", re.IGNORECASE)
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE_RE = re.compile(r"//.*")
_SPLIT_RE = re.compile(r"[\s(),;]+")


def base_net(signal: str) -> str:
    """Strip a bit-select: ``n7[3]`` -> ``n7``."""
    bracket = signal.find("[")
    return signal[:bracket] if bracket >= 0 else signal


class ParsedNetlist:
    """The raw facts read straight out of the Verilog, before any analysis."""

    def __init__(self) -> None:
        self.gate_names: List[str] = []
        self.gate_type: Dict[int, str] = {}
        self.signal_driver: Dict[str, int] = {}
        self.signal_loads: Dict[str, List[int]] = defaultdict(list)
        self.signal_pins: Dict[str, Set[str]] = defaultdict(set)
        self.dff_d_signal: Dict[int, str] = {}
        self.inputs: Set[str] = set()
        self.outputs: Set[str] = set()

    @property
    def num_gates(self) -> int:
        return len(self.gate_names)

    def adjacency(self):
        """Return ``(edges, forward, reverse)`` over gate indices.

        An edge runs from the gate driving a net to every gate loading it.
        """
        n = self.num_gates
        forward: List[List[int]] = [[] for _ in range(n)]
        reverse: List[List[int]] = [[] for _ in range(n)]
        edges: List[tuple] = []

        for signal, driver in self.signal_driver.items():
            for load in self.signal_loads.get(signal, []):
                edges.append((driver, load))
                forward[driver].append(load)
                reverse[load].append(driver)

        return edges, forward, reverse

    def port_nets(self):
        """Return ``(input_nets, output_nets)`` with bus bounds removed.

        Port declarations tokenise to include the bus bounds as digits, which
        are dropped here.
        """
        return ({t for t in self.inputs if not t.isdigit()},
                {t for t in self.outputs if not t.isdigit()})


def parse_netlist(path: str) -> ParsedNetlist:
    """Read a flattened contest netlist."""
    with open(path, errors="replace") as handle:
        text = handle.read()

    text = _COMMENT_BLOCK_RE.sub("", text)
    text = _COMMENT_LINE_RE.sub("", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    parsed = ParsedNetlist()
    gate_lines: List[str] = []

    for line in lines:
        if line.startswith("input"):
            parsed.inputs.update(re.findall(r"\w+", line)[1:])
        elif line.startswith("output"):
            parsed.outputs.update(re.findall(r"\w+", line)[1:])
        elif _GATE_HEAD_RE.match(line):
            gate_lines.append(line)

    for line in gate_lines:
        tokens = [t for t in _SPLIT_RE.split(line) if t]
        if len(tokens) < 3:
            continue

        gtype = tokens[0].lower()
        gate_name = tokens[1]
        gid = parsed.num_gates

        if gtype == "dff":
            pins: Dict[str, str] = {}
            for i in range(2, len(tokens) - 1, 2):
                pin = tokens[i].lstrip(".").upper()
                signal = tokens[i + 1]
                pins[pin] = signal
                parsed.signal_pins[signal].add(pin)
            output_signal = pins.get("Q")
            input_signals = [v for k, v in pins.items() if k in DFF_INPUT_PINS]
            if "D" in pins:
                parsed.dff_d_signal[gid] = pins["D"]
        else:
            output_signal = tokens[2]
            input_signals = tokens[3:]

        parsed.gate_names.append(gate_name)
        parsed.gate_type[gid] = gtype

        if output_signal:
            parsed.signal_driver[output_signal] = gid
        for signal in input_signals:
            parsed.signal_loads[signal].append(gid)

    return parsed
