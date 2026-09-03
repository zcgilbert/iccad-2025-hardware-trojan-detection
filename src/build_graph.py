"""Turn a flattened gate-level Verilog netlist into a labelled PyG graph.

This is step one of the pipeline: everything downstream (``train.py``,
``predict.py``) consumes the :class:`torch_geometric.data.Data` objects this
module produces.

What a graph looks like
-----------------------
* **Nodes** are gate instances -- the nine contest primitives
  ``and/or/nand/nor/not/buf/xor/xnor`` plus ``dff``.  Nets are *not* nodes.
* **Edges** run driver -> load.  Every edge is materialised twice, as
  ``edge_index_fw`` and its transpose ``edge_index_bw``, because the model
  convolves over both directions (see ``gnn.BiGCNLayer``).
* **Node features** are the 48 columns of :data:`FEATURE_ORDER` -- gate type,
  flip-flop pin roles, I/O flags, structural counts, six families of graph
  distance, a static signal probability, and five simulation tallies.
* **Labels** ``y`` mark the gates named in the design's contest label file.
* **Subgraph annotations** describe the weakly-connected component each gate
  belongs to once the graph is cut at flip-flop boundaries -- see
  :func:`attach_subgraphs` for why that matters.

Feature design rationale
------------------------
A Trojan has no single give-away gate type; what distinguishes it is *where it
sits*.  Payload logic tends to be far from any primary output, shallowly
connected to the rest of the design, and rarely toggling.  The distance,
neighbourhood and simulation-tally families exist to make exactly those
properties visible to the network.

Feature and normalisation selection are controlled by the module-level toggles
below.  Disabled columns are kept in place and zeroed rather than dropped, so
the feature-vector width -- and therefore checkpoint compatibility -- stays
constant no matter which toggles are set.

Usage
-----
    python src/build_graph.py --netlists data/holdout/netlists \
                              --labels   data/holdout/labels \
                              --out      build/graphs
"""

from __future__ import annotations

import os
import random
import re
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data

import dataset
from netlist_parser import ParsedNetlist, base_net, parse_netlist

# Excel export is a debugging aid only; the pipeline runs fine without pandas.
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# The order of this list IS the column order of `data.x`, and the trained
# checkpoint depends on it.  Append-only: never reorder or remove an entry.
FEATURE_ORDER: List[str] = [
    # -- gate type, one-hot ------------------------------------------------
    "is_and", "is_or", "is_nand", "is_nor", "is_not", "is_buf",
    "is_xor", "is_xnor", "is_dff",
    # -- which flip-flop pin this gate drives (consumer-side marker) --------
    "is_ck", "is_d", "is_q", "is_rst", "is_set",
    # -- primary input / output ------------------------------------------
    "is_PI", "is_PO",
    # -- local structure --------------------------------------------------
    "is_sink_gate", "fanin_2level", "fanout_2level", "fanout_dff_count",
    "fanin_same_type", "level",
    # -- distances, combinational paths only; -1 means unreachable ---------
    "min_from_PI", "max_from_PI", "min_to_PO", "max_to_PO",
    "min_from_DFF", "max_from_DFF", "min_to_DFF", "max_to_DFF",
    "min_from_COMB", "max_from_COMB", "min_to_COMB", "max_to_COMB",
    "min_from_GND", "max_from_GND", "min_from_VDD", "max_from_VDD",
    # -- four-hop neighbourhood size and gate-type diversity ---------------
    "in_4lvl_cnt", "out_4lvl_cnt", "in_type_cnt", "out_type_cnt",
    # -- static probability that the gate output is 1 ----------------------
    "p_is_1",
    # -- random-simulation tallies (filled in by simulate_on_data) ---------
    "num_1to0", "num_0to1", "longest_1", "longest_0", "total_1",
]
FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_ORDER)}

GATE_TYPES = ["and", "or", "nand", "nor", "not", "buf", "xor", "xnor", "dff"]
# The one-hot block sits at the front of FEATURE_ORDER, so a gate type's index
# in this list is also its column index.
GATE_TYPE_INDEX = {name: i for i, name in enumerate(GATE_TYPES)}

# Flip-flop pin name -> the feature it sets on the gate driving that pin.
PIN_TO_FEATURE = {
    "CK": "is_ck", "D": "is_d", "Q": "is_q",
    "RN": "is_rst", "SN": "is_set",
    "RST": "is_rst", "SET": "is_set",
}
DFF_INPUT_PINS = ("D", "CK", "RN", "SN", "RST", "SET")

DISTANCE_FEATURES = [
    "min_from_PI", "max_from_PI", "min_to_PO", "max_to_PO",
    "min_from_DFF", "max_from_DFF", "min_to_DFF", "max_to_DFF",
    "min_from_COMB", "max_from_COMB", "min_to_COMB", "max_to_COMB",
    "min_from_GND", "max_from_GND", "min_from_VDD", "max_from_VDD",
]
SIM_FEATURES = ["num_1to0", "num_0to1", "longest_1", "longest_0", "total_1"]


# ---------------------------------------------------------------------------
# Feature and normalisation toggles
# ---------------------------------------------------------------------------
# These are the ablation switches used during development.  They are kept at
# the values the shipped checkpoint was trained with -- changing them changes
# the meaning of `data.x` and invalidates `models/trojan_gnn.pt`.
#
# ENABLED_FEATURES is None  -> everything except DISABLED_FEATURES is on.
# ENABLED_FEATURES is a set -> only those are on.
# Either way, an off column is zeroed, not removed.
ENABLED_FEATURES: Optional[Set[str]] = None

# The 13 columns switched off for the submitted run.  They are computed and
# then zeroed, so the vector width -- and checkpoint compatibility -- is
# unaffected.  This is the configuration models/trojan_gnn.pt was trained
# with; turning any of them back on changes what data.x means and the shipped
# weights no longer apply.
DISABLED_FEATURES: Set[str] = {
    "fanin_same_type",
    # distances relative to the register boundary
    "min_from_DFF", "max_from_DFF", "min_to_DFF", "max_to_DFF",
    # distances relative to combinational sources / sinks
    "min_from_COMB", "max_from_COMB", "min_to_COMB",
    # distances from a tied constant
    "min_from_GND", "max_from_GND", "min_from_VDD", "max_from_VDD",
    "p_is_1",
}

# ENABLED_NORM is None  -> normalise REASONABLE_TO_NORMALIZE minus DISABLED_NORM.
# ENABLED_NORM is a set -> normalise exactly those.
# NEVER_NORMALIZE always wins.
ENABLED_NORM: Optional[Set[str]] = None

# Already binary or already in [0, 1]; scaling them would only destroy meaning.
NEVER_NORMALIZE: Set[str] = {
    "is_and", "is_or", "is_nand", "is_nor", "is_not", "is_buf",
    "is_xor", "is_xnor", "is_dff",
    "is_ck", "is_d", "is_q", "is_rst", "is_set",
    "is_PI", "is_PO", "is_sink_gate",
    "fanin_same_type", "p_is_1",
}

_SCALABLE = {
    "level", *DISTANCE_FEATURES,
    "fanin_2level", "fanout_2level", "fanout_dff_count",
    "in_4lvl_cnt", "out_4lvl_cnt", "in_type_cnt", "out_type_cnt",
    *SIM_FEATURES,
}
REASONABLE_TO_NORMALIZE: Set[str] = set(_SCALABLE)

# Scaling is switched off for every scalable column, so the shipped model was
# trained on raw magnitudes rather than per-design normalised ones.
DISABLED_NORM: Set[str] = set(_SCALABLE)

# Random vectors applied per design when tallying switching activity.
SIM_COUNT_DEFAULT = 1000

# Sentinel for "no path exists"; collapsed to -1 before it reaches a feature.
INF = 10 ** 9


# ---------------------------------------------------------------------------
# Toggle helpers
# ---------------------------------------------------------------------------
def _enabled_feature_names() -> Set[str]:
    if ENABLED_FEATURES is None:
        return set(FEATURE_ORDER) - set(DISABLED_FEATURES)
    return set(ENABLED_FEATURES)


def _normalise_on(name: str) -> bool:
    """Whether feature ``name`` should be scaled."""
    if name not in _enabled_feature_names():
        return False
    if name in NEVER_NORMALIZE:
        return False
    if ENABLED_NORM is None:
        return name in REASONABLE_TO_NORMALIZE and name not in DISABLED_NORM
    return name in ENABLED_NORM


def _feature_mask() -> torch.Tensor:
    """1.0 for every enabled column, 0.0 for the rest."""
    enabled = _enabled_feature_names()
    return torch.tensor([1.0 if n in enabled else 0.0 for n in FEATURE_ORDER],
                        dtype=torch.float)


def _apply_mask(x: torch.Tensor) -> torch.Tensor:
    """Zero out every disabled column, keeping the tensor width unchanged."""
    if x.numel() == 0:
        return x
    return x * _feature_mask().to(x.device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Scaling primitives
# ---------------------------------------------------------------------------
def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    low, high = min(values), max(values)
    if high <= low:
        return [0.0] * len(values)
    span = high - low
    return [(v - low) / span for v in values]


def _scale_distance(values: List[int]) -> List[float]:
    """Scale distances to [0, 1] while keeping -1 as a distinct 'unreachable'."""
    reachable = [v for v in values if v >= 0]
    peak = max(reachable) if reachable else 0
    if peak <= 0:
        return [-1.0 if v < 0 else 0.0 for v in values]
    return [-1.0 if v < 0 else v / peak for v in values]


def _maybe_minmax(name: str, values: List[float]) -> List[float]:
    return _minmax(values) if _normalise_on(name) else list(values)


def _maybe_scale_distance(name: str, values: List[int]) -> List[float]:
    if _normalise_on(name):
        return _scale_distance(values)
    return [float(v) for v in values]


def _maybe_scale_level(name: str, levels: List[int]) -> List[float]:
    if not _normalise_on(name):
        return [float(v) for v in levels]
    peak = max(levels) if levels else 0
    return [(v / peak) if peak > 0 else 0.0 for v in levels]


def _maybe_scale_type_count(name: str, values: List[float]) -> List[float]:
    # There are nine gate types, so nine is the natural denominator.
    return [v / 9.0 for v in values] if _normalise_on(name) else list(values)


def _maybe_scale_sim(name: str, values: List[float], denominator: int) -> List[float]:
    if not _normalise_on(name):
        return [float(v) for v in values]
    denom = max(denominator, 1)
    return [float(v) / denom for v in values]


# ---------------------------------------------------------------------------
# Node record
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """One gate instance and every derived quantity computed about it."""

    gid: int
    name: str
    gtype: str
    outs: List[int] = field(default_factory=list)
    ins: List[int] = field(default_factory=list)

    level: int = 0
    p1: float = 0.5

    # Distances start at INF (min) / -1 (max) and are finalised to -1 when the
    # target turns out to be unreachable.
    min_from_PI: int = INF
    max_from_PI: int = -1
    min_to_PO: int = INF
    max_to_PO: int = -1

    min_from_DFF: int = INF
    max_from_DFF: int = -1
    min_to_DFF: int = INF
    max_to_DFF: int = -1

    min_from_COMB: int = INF
    max_from_COMB: int = -1
    min_to_COMB: int = INF
    max_to_COMB: int = -1

    min_from_GND: int = INF
    max_from_GND: int = -1
    min_from_VDD: int = INF
    max_from_VDD: int = -1

    in_4lvl: Set[int] = field(default_factory=set)
    out_4lvl: Set[int] = field(default_factory=set)
    in_types: Set[str] = field(default_factory=set)
    out_types: Set[str] = field(default_factory=set)

    def finalise_unreachable(self) -> None:
        """Replace the INF sentinel with -1 so the value can enter a feature."""
        for attr in ("min_from_PI", "min_to_PO",
                     "min_from_DFF", "min_to_DFF",
                     "min_from_COMB", "min_to_COMB",
                     "min_from_GND", "min_from_VDD"):
            if getattr(self, attr) >= INF:
                setattr(self, attr, -1)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def _build_edges(parsed: ParsedNetlist) -> Tuple[List[Tuple[int, int]],
                                                 List[List[int]],
                                                 List[List[int]]]:
    """Connect every driver to every load of the nets it drives."""
    n = parsed.num_gates
    forward: List[List[int]] = [[] for _ in range(n)]
    reverse: List[List[int]] = [[] for _ in range(n)]
    edges: List[Tuple[int, int]] = []

    for signal, driver in parsed.signal_driver.items():
        for load in parsed.signal_loads.get(signal, []):
            edges.append((driver, load))
            forward[driver].append(load)
            reverse[load].append(driver)

    return edges, forward, reverse


def _combinational_levels(nodes: List[Node], is_comb: List[bool],
                          forward: List[List[int]], reverse: List[List[int]]
                          ) -> List[int]:
    """Assign each combinational gate its longest-path depth, and return the
    topological order used by every later dynamic-programming pass.

    Flip-flops are excluded: they break combinational paths, so depth is only
    meaningful between register boundaries.  Gates caught in a combinational
    loop never reach in-degree zero and are simply absent from the order.
    """
    n = len(nodes)
    indegree = [0] * n
    for gid in range(n):
        if not is_comb[gid]:
            continue
        indegree[gid] = sum(1 for pred in reverse[gid] if is_comb[pred])

    queue = deque(gid for gid in range(n) if is_comb[gid] and indegree[gid] == 0)
    for gid in queue:
        nodes[gid].level = 0

    order: List[int] = []
    while queue:
        gid = queue.popleft()
        order.append(gid)
        for succ in forward[gid]:
            if not is_comb[succ]:
                continue
            indegree[succ] -= 1
            nodes[succ].level = max(nodes[succ].level, nodes[gid].level + 1)
            if indegree[succ] == 0:
                queue.append(succ)
    return order


def _gate_output_probability(gtype: str, input_probs: List[float]) -> float:
    """Probability the gate outputs 1, assuming independent inputs.

    The independence assumption is wrong in the presence of reconvergent
    fanout, but it is cheap and monotone, which is all the feature needs: a
    signal that is almost always 0 is a classic Trojan-trigger signature.
    """
    if not input_probs:
        return 0.5
    if gtype == "not":
        return 1.0 - input_probs[0]
    if gtype == "buf":
        return input_probs[0]
    if gtype == "and":
        product = 1.0
        for p in input_probs:
            product *= p
        return product
    if gtype == "or":
        product = 1.0
        for p in input_probs:
            product *= 1.0 - p
        return 1.0 - product
    if gtype == "xor":
        if len(input_probs) == 1:
            return input_probs[0]
        p = input_probs[0]
        for q in input_probs[1:]:
            p = p * (1.0 - q) + (1.0 - p) * q
        return p
    if gtype == "nand":
        return 1.0 - _gate_output_probability("and", input_probs)
    if gtype == "nor":
        return 1.0 - _gate_output_probability("or", input_probs)
    if gtype == "xnor":
        return 1.0 - _gate_output_probability("xor", input_probs)
    return 0.5


def _propagate_probabilities(nodes: List[Node], parsed: ParsedNetlist,
                             is_comb: List[bool], order: List[int],
                             dff_d_pred: List[int]) -> None:
    """Push signal probabilities forward, then settle the flip-flops."""
    for gid in order:
        if not is_comb[gid]:
            continue
        input_probs = [nodes[pred].p1 for pred in nodes[gid].ins]
        nodes[gid].p1 = _gate_output_probability(parsed.gate_type[gid], input_probs)

    for gid in range(len(nodes)):
        if parsed.gate_type[gid] != "dff":
            continue
        if dff_d_pred[gid] != -1:
            # Q takes on whatever D sees.
            nodes[gid].p1 = nodes[dff_d_pred[gid]].p1
        elif nodes[gid].ins:
            nodes[gid].p1 = sum(nodes[p].p1 for p in nodes[gid].ins) / len(nodes[gid].ins)
        else:
            nodes[gid].p1 = 0.5


class _CombGraph:
    """The combinational-only view: the same graph with flip-flops removed.

    Every distance feature is measured on this view, because a path that runs
    through a flip-flop is not a combinational path.
    """

    def __init__(self, n: int, is_comb: List[bool], forward: List[List[int]]):
        self.n = n
        self.is_comb = is_comb
        self.forward: List[List[int]] = [[] for _ in range(n)]
        self.reverse: List[List[int]] = [[] for _ in range(n)]
        self.indegree = [0] * n
        self.outdegree = [0] * n

        for u in range(n):
            if not is_comb[u]:
                continue
            for v in forward[u]:
                if not is_comb[v]:
                    continue
                self.forward[u].append(v)
                self.reverse[v].append(u)
                self.outdegree[u] += 1
                self.indegree[v] += 1

    def bfs_from(self, seeds: Set[int]) -> List[int]:
        """Shortest distance forward from any seed."""
        return self._bfs(seeds, self.forward)

    def bfs_to(self, targets: Set[int]) -> List[int]:
        """Shortest distance backward to any target."""
        return self._bfs(targets, self.reverse)

    def _bfs(self, seeds: Set[int], adjacency: List[List[int]]) -> List[int]:
        distance = [INF] * self.n
        queue = deque()
        for seed in seeds:
            if seed < 0 or not self.is_comb[seed]:
                continue
            distance[seed] = 0
            queue.append(seed)
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if distance[v] > distance[u] + 1:
                    distance[v] = distance[u] + 1
                    queue.append(v)
        return distance

    def longest_from(self, seeds: Set[int], order: List[int],
                     seed_value: int) -> List[int]:
        """Longest distance forward from any seed, by DP over ``order``."""
        best = [-1] * self.n
        for seed in seeds:
            best[seed] = max(best[seed], seed_value)
        for u in order:
            if best[u] < 0:
                continue
            for v in self.forward[u]:
                best[v] = max(best[v], best[u] + 1)
        return best

    def longest_to(self, targets: Set[int], order: List[int],
                   seed_value: int) -> List[int]:
        """Longest distance backward to any target, by DP over reversed order."""
        best = [-1] * self.n
        for target in targets:
            best[target] = max(best[target], seed_value)
        for u in reversed(order):
            if best[u] < 0:
                continue
            for p in self.reverse[u]:
                best[p] = max(best[p], best[u] + 1)
        return best


def _boundary_sets(parsed: ParsedNetlist, comb: _CombGraph,
                   forward: List[List[int]], reverse: List[List[int]],
                   gate_pi: List[bool], gate_po: List[bool]
                   ) -> Dict[str, Set[int]]:
    """Collect the seed sets each distance family is measured from."""
    n = parsed.num_gates
    is_comb = comb.is_comb

    dff_successors = {v for u in range(n) if parsed.gate_type[u] == "dff"
                      for v in forward[u] if is_comb[v]}
    dff_predecessors = {p for u in range(n) if parsed.gate_type[u] == "dff"
                        for p in reverse[u] if is_comb[p]}

    gnd_loads: Set[int] = set()
    vdd_loads: Set[int] = set()
    for signal, loads in parsed.signal_loads.items():
        if signal == "1'b0":
            gnd_loads.update(u for u in loads if is_comb[u])
        elif signal == "1'b1":
            vdd_loads.update(u for u in loads if is_comb[u])

    return {
        "pi": {gid for gid in range(n) if is_comb[gid] and gate_pi[gid]},
        "po": {gid for gid in range(n) if is_comb[gid] and gate_po[gid]},
        "dff_succ": dff_successors,
        "dff_pred": dff_predecessors,
        "comb_src": {gid for gid in range(n)
                     if is_comb[gid] and comb.indegree[gid] == 0},
        "comb_sink": {gid for gid in range(n)
                      if is_comb[gid] and comb.outdegree[gid] == 0},
        "gnd": gnd_loads,
        "vdd": vdd_loads,
    }


def _assign_distances(nodes: List[Node], parsed: ParsedNetlist,
                      comb: _CombGraph, seeds: Dict[str, Set[int]],
                      order: List[int]) -> None:
    """Compute all six distance families and write them onto the nodes."""
    n = parsed.num_gates

    min_from_pi = comb.bfs_from(seeds["pi"])
    min_to_po = comb.bfs_to(seeds["po"])
    max_from_pi = comb.longest_from(seeds["pi"], order, 0)
    max_to_po = comb.longest_to(seeds["po"], order, 0)

    # Distance 1, not 0: these gates are the flip-flop's neighbours, so they
    # sit one hop away from the register boundary itself.
    min_from_dff = comb.bfs_from(seeds["dff_succ"])
    min_to_dff = comb.bfs_to(seeds["dff_pred"])
    max_from_dff = comb.longest_from(seeds["dff_succ"], order, 1)
    max_to_dff = comb.longest_to(seeds["dff_pred"], order, 1)

    min_from_comb = comb.bfs_from(seeds["comb_src"])
    min_to_comb = comb.bfs_to(seeds["comb_sink"])
    max_from_comb = comb.longest_from(seeds["comb_src"], order, 0)
    max_to_comb = comb.longest_to(seeds["comb_sink"], order, 0)

    min_from_gnd = comb.bfs_from(seeds["gnd"]) if seeds["gnd"] else [INF] * n
    min_from_vdd = comb.bfs_from(seeds["vdd"]) if seeds["vdd"] else [INF] * n
    max_from_gnd = comb.longest_from(seeds["gnd"], order, 0)
    max_from_vdd = comb.longest_from(seeds["vdd"], order, 0)

    for gid in range(n):
        node = nodes[gid]

        node.min_from_PI = min_from_pi[gid]
        node.max_from_PI = max_from_pi[gid]
        node.min_to_PO = min_to_po[gid]
        node.max_to_PO = max_to_po[gid]

        if parsed.gate_type[gid] == "dff":
            # A flip-flop is its own register boundary: distance zero, by
            # definition, rather than "unreachable".
            node.min_from_DFF = node.min_to_DFF = 0
            node.max_from_DFF = node.max_to_DFF = 0
        else:
            node.min_from_DFF = min_from_dff[gid]
            node.min_to_DFF = min_to_dff[gid]
            node.max_from_DFF = max_from_dff[gid]
            node.max_to_DFF = max_to_dff[gid]

        node.min_from_COMB = min_from_comb[gid]
        node.min_to_COMB = min_to_comb[gid]
        node.max_from_COMB = max_from_comb[gid]
        node.max_to_COMB = max_to_comb[gid]

        node.min_from_GND = min_from_gnd[gid]
        node.min_from_VDD = min_from_vdd[gid]
        node.max_from_GND = max_from_gnd[gid]
        node.max_from_VDD = max_from_vdd[gid]

        node.finalise_unreachable()


def _neighbourhood_features(nodes: List[Node], parsed: ParsedNetlist,
                            forward: List[List[int]],
                            reverse: List[List[int]]) -> None:
    """Record the four-hop neighbourhood and the gate types around each gate.

    The walk refuses to enter or leave a flip-flop, keeping the neighbourhood
    combinational for the same reason the distances are.
    """
    gate_type = parsed.gate_type

    def walk(start: int, hops: int, adjacency: List[List[int]]) -> Set[int]:
        seen = {start}
        frontier = {start}
        for _ in range(hops):
            nxt: Set[int] = set()
            for u in frontier:
                if gate_type[u] == "dff":
                    continue
                for v in adjacency[u]:
                    if v not in seen and gate_type[v] != "dff":
                        seen.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        seen.discard(start)
        return seen

    for gid, node in enumerate(nodes):
        node.out_4lvl = walk(gid, 4, forward)
        node.in_4lvl = walk(gid, 4, reverse)
        node.out_types = {gate_type[v] for v in node.outs}
        node.in_types = {gate_type[v] for v in node.ins}


def _gate_io_flags(parsed: ParsedNetlist) -> Tuple[List[bool], List[bool]]:
    """Mark gates that read a primary input / drive a primary output.

    Bus bounds arrive as digit tokens from the port declarations and are
    dropped here.
    """
    n = parsed.num_gates
    input_nets = {t for t in parsed.inputs if not t.isdigit()}
    output_nets = {t for t in parsed.outputs if not t.isdigit()}

    reads_pi = [False] * n
    drives_po = [False] * n

    for signal, loads in parsed.signal_loads.items():
        if base_net(signal) in input_nets:
            for gid in loads:
                reads_pi[gid] = True

    for signal, driver in parsed.signal_driver.items():
        if base_net(signal) in output_nets:
            drives_po[driver] = True

    return reads_pi, drives_po


def _gate_dff_pin_roles(parsed: ParsedNetlist) -> List[Set[str]]:
    """For each gate, the flip-flop pins its output feeds.

    Driving a clock line is very different from driving a data line, and a
    Trojan that gates a clock or forces a reset shows up here.
    """
    roles: List[Set[str]] = [set() for _ in range(parsed.num_gates)]
    for signal, loads in parsed.signal_loads.items():
        pins = parsed.signal_pins.get(signal)
        if not pins:
            continue
        features = {PIN_TO_FEATURE[p] for p in pins if p in PIN_TO_FEATURE}
        if not features:
            continue
        for gid in loads:
            roles[gid] |= features
    return roles


def _build_feature_matrix(nodes: List[Node], parsed: ParsedNetlist,
                          reads_pi: List[bool], drives_po: List[bool],
                          pin_roles: List[Set[str]]) -> torch.Tensor:
    """Assemble the raw (unscaled) feature matrix, one row per gate."""
    idx = FEATURE_INDEX
    rows: List[List[float]] = []

    for gid, node in enumerate(nodes):
        row = [0.0] * len(FEATURE_ORDER)

        if node.gtype in GATE_TYPE_INDEX:
            row[GATE_TYPE_INDEX[node.gtype]] = 1.0

        for feature in pin_roles[gid]:
            row[idx[feature]] = 1.0

        row[idx["is_PI"]] = 1.0 if reads_pi[gid] else 0.0
        row[idx["is_PO"]] = 1.0 if drives_po[gid] else 0.0
        row[idx["is_sink_gate"]] = 1.0 if not node.outs else 0.0

        # Two-hop fanin / fanout, counted as sets so reconvergence is not
        # double-counted.
        fanin = set(node.ins)
        for pred in list(fanin):
            fanin |= set(nodes[pred].ins)
        fanout = set(node.outs)
        for succ in list(fanout):
            fanout |= set(nodes[succ].outs)

        row[idx["fanin_2level"]] = float(len(fanin))
        row[idx["fanout_2level"]] = float(len(fanout))
        row[idx["fanout_dff_count"]] = float(
            sum(1 for g in fanout if parsed.gate_type.get(g) == "dff"))

        driver_types = [parsed.gate_type.get(p) for p in node.ins]
        same = sum(1 for t in driver_types if t == node.gtype)
        row[idx["fanin_same_type"]] = (same / len(driver_types)) if driver_types else 0.0

        row[idx["level"]] = float(node.level)

        for name in DISTANCE_FEATURES:
            row[idx[name]] = float(getattr(node, name))

        row[idx["in_4lvl_cnt"]] = float(len(node.in_4lvl))
        row[idx["out_4lvl_cnt"]] = float(len(node.out_4lvl))
        row[idx["in_type_cnt"]] = float(len(node.in_types))
        row[idx["out_type_cnt"]] = float(len(node.out_types))

        row[idx["p_is_1"]] = float(node.p1)

        # Simulation columns stay zero here; simulate_on_data fills them.
        rows.append(row)

    return torch.tensor(rows, dtype=torch.float)


def _normalise(raw: torch.Tensor) -> torch.Tensor:
    """Apply the configured scaling, column family by column family."""
    idx = FEATURE_INDEX
    out = raw.clone()

    levels = [int(v) for v in raw[:, idx["level"]].tolist()]
    out[:, idx["level"]] = torch.tensor(
        _maybe_scale_level("level", levels), dtype=torch.float)

    for name in DISTANCE_FEATURES:
        values = [int(v) for v in raw[:, idx[name]].tolist()]
        out[:, idx[name]] = torch.tensor(
            _maybe_scale_distance(name, values), dtype=torch.float)

    for name in ("fanin_2level", "fanout_2level", "fanout_dff_count",
                 "in_4lvl_cnt", "out_4lvl_cnt"):
        out[:, idx[name]] = torch.tensor(
            _maybe_minmax(name, raw[:, idx[name]].tolist()), dtype=torch.float)

    for name in ("in_type_cnt", "out_type_cnt"):
        out[:, idx[name]] = torch.tensor(
            _maybe_scale_type_count(name, raw[:, idx[name]].tolist()),
            dtype=torch.float)

    return out


def netlist_to_graph(netlist_path: str, label_path: Optional[str] = None,
                     sim_count: int = SIM_COUNT_DEFAULT) -> Data:
    """Build the labelled graph for one netlist.

    ``label_path`` is optional: without it every gate is labelled 0, which is
    what inference on unseen designs needs.
    """
    if os.path.isdir(netlist_path):
        raise ValueError("expected a Verilog file, got a directory: " + netlist_path)

    parsed = parse_netlist(netlist_path)
    n = parsed.num_gates
    if n == 0:
        # The original code crashed with an opaque IndexError here.  A netlist
        # with no primitive instances is almost always an RTL source file that
        # has not been through synthesis yet, so say so.
        raise ValueError(
            "no primitive gate instances found in " + netlist_path +
            " -- expected a synthesised, flattened gate-level netlist")

    edges, forward, reverse = _build_edges(parsed)

    nodes = [Node(gid=i, name=parsed.gate_names[i], gtype=parsed.gate_type[i],
                  outs=list(forward[i]), ins=list(reverse[i]))
             for i in range(n)]

    # Which flip-flop drives each flip-flop's D pin (-1 when unknown).
    dff_d_pred = [-1] * n
    for gid, signal in parsed.dff_d_signal.items():
        driver = parsed.signal_driver.get(signal)
        if driver is not None:
            dff_d_pred[gid] = driver

    is_comb = [parsed.gate_type[i] != "dff" for i in range(n)]
    order = _combinational_levels(nodes, is_comb, forward, reverse)
    _propagate_probabilities(nodes, parsed, is_comb, order, dff_d_pred)

    comb = _CombGraph(n, is_comb, forward)
    reads_pi, drives_po = _gate_io_flags(parsed)
    seeds = _boundary_sets(parsed, comb, forward, reverse, reads_pi, drives_po)
    _assign_distances(nodes, parsed, comb, seeds, order)
    _neighbourhood_features(nodes, parsed, forward, reverse)

    pin_roles = _gate_dff_pin_roles(parsed)
    raw = _build_feature_matrix(nodes, parsed, reads_pi, drives_po, pin_roles)
    x = _apply_mask(_normalise(raw))

    _, trojan_gates = dataset.read_label(label_path)
    y = torch.tensor([1 if name in trojan_gates else 0
                      for name in parsed.gate_names], dtype=torch.long)

    if edges:
        edge_index_fw = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index_bw = edge_index_fw[[1, 0], :]
        edge_pairs = torch.tensor(edges, dtype=torch.long)
    else:
        edge_index_fw = torch.empty((2, 0), dtype=torch.long)
        edge_index_bw = torch.empty((2, 0), dtype=torch.long)
        edge_pairs = torch.empty((0, 2), dtype=torch.long)

    data = Data(
        x=x,
        edge_index_fw=edge_index_fw,
        edge_index_bw=edge_index_bw,
        y=y,
        gate_names=parsed.gate_names,
        gate_types=[parsed.gate_type[i] for i in range(n)],
        edges=edge_pairs,
        inputs=sorted(parsed.inputs),
        outputs=sorted(parsed.outputs),
        levels=torch.tensor([nd.level for nd in nodes], dtype=torch.long),
        probs=torch.tensor([nd.p1 for nd in nodes], dtype=torch.float),
        sim_count=sim_count,
        dff_d_pred=torch.tensor(dff_d_pred, dtype=torch.long),
        feature_mask=_feature_mask(),
        raw_feature_matrix=raw,      # unscaled copy, for the Excel export
    )

    data = attach_subgraphs(data, cut_at_dff=True)
    data = attach_wcc_labels(data, pos_ratio_threshold=0.01)
    return data


# Kept under the original name so older notebooks and scripts still import it.
parse_verilog_to_bidirectional_graph = netlist_to_graph


# ---------------------------------------------------------------------------
# Subgraph (weakly-connected component) annotations
# ---------------------------------------------------------------------------
def attach_subgraphs(data: Data, cut_at_dff: bool = True) -> Data:
    """Split the design at flip-flop boundaries and describe each piece.

    Why this is the second branch of the model: an inserted Trojan is usually
    *structurally* separable.  Its payload forms a small component with few
    connections back to the host logic, which is invisible to a purely local
    gate-level view but obvious once the design is decomposed.  Each gate is
    then classified alongside a descriptor of the component it lives in:
    size, internal edge count, density, boundary-edge count and ratio, a
    gate-type histogram, and the component means of a few node features.
    """
    num_nodes = data.x.size(0)

    if not hasattr(data, "edge_index_fw") or data.edge_index_fw.numel() == 0:
        data.subgraph_id = torch.zeros(num_nodes, dtype=torch.long)
        data.subgraph_feat = torch.zeros(1, 8, dtype=torch.float)
        data.subgraph_sizes = torch.tensor([num_nodes], dtype=torch.long)
        return data

    edge_fw = data.edge_index_fw
    edge_bw = getattr(data, "edge_index_bw", edge_fw[[1, 0], :])
    undirected = torch.cat([edge_fw, edge_bw], dim=1)

    if cut_at_dff and hasattr(data, "gate_types"):
        types = list(data.gate_types)
        keep = torch.tensor(
            [not (types[u] == "dff" or types[v] == "dff")
             for u, v in undirected.t().tolist()], dtype=torch.bool)
        undirected = undirected[:, keep]

    if undirected.numel() > 0:
        low = torch.min(undirected, dim=0).values
        high = torch.max(undirected, dim=0).values
        undirected = torch.unique(torch.stack([low, high], dim=0), dim=1)

    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in undirected.t().tolist():
        adjacency[u].append(v)
        adjacency[v].append(u)

    component = [-1] * num_nodes
    num_components = 0
    for start in range(num_nodes):
        if component[start] != -1:
            continue
        component[start] = num_components
        stack = [start]
        for node in stack:
            for neighbour in adjacency[node]:
                if component[neighbour] == -1:
                    component[neighbour] = num_components
                    stack.append(neighbour)
        num_components += 1

    component_t = torch.tensor(component, dtype=torch.long)
    sizes = torch.bincount(component_t, minlength=num_components)
    num_components = sizes.size(0)

    internal = torch.zeros(num_components, dtype=torch.long)
    boundary = torch.zeros(num_components, dtype=torch.long)
    for u, v in undirected.t().tolist():
        cu, cv = component[u], component[v]
        if cu == cv:
            internal[cu] += 1
        else:
            boundary[cu] += 1
            boundary[cv] += 1

    type_hist = torch.zeros(num_components, len(GATE_TYPES), dtype=torch.float)
    for gid, gtype in enumerate(data.gate_types):
        col = GATE_TYPE_INDEX.get(gtype, -1)
        if col >= 0:
            type_hist[component_t[gid], col] += 1.0
    type_hist = type_hist / sizes.float().unsqueeze(1).clamp_min(1.0)

    # Component means of the node features most indicative of "off to the side".
    pick = [name for name in ("p_is_1", "fanin_2level", "fanout_2level",
                              "in_4lvl_cnt", "out_4lvl_cnt",
                              "min_to_PO", "min_from_PI")
            if name in FEATURE_INDEX]
    if pick:
        totals = torch.zeros(num_components, len(pick), dtype=torch.float)
        columns = torch.stack([data.x[:, FEATURE_INDEX[n]] for n in pick], dim=1)
        totals.index_add_(0, component_t, columns)
        means = totals / sizes.float().unsqueeze(1).clamp_min(1.0)
    else:
        means = torch.zeros(num_components, 0, dtype=torch.float)

    node_count = sizes.float().unsqueeze(1)
    edge_count = internal.float().unsqueeze(1)
    boundary_count = boundary.float().unsqueeze(1)
    density = (2.0 * edge_count) / (node_count * (node_count - 1.0) + 1e-6)
    boundary_ratio = boundary_count / (boundary_count + edge_count + 1e-6)

    data.subgraph_id = component_t
    data.subgraph_feat = torch.cat(
        [node_count, edge_count, density, boundary_count, boundary_ratio,
         type_hist, means], dim=1).float()
    data.subgraph_sizes = sizes
    return data


def attach_wcc_labels(data: Data, pos_ratio_threshold: float = 0.01) -> Data:
    """Label every gate in a component that contains enough Trojan gates.

    An auxiliary, coarser target: it says "this whole component is
    Trojan-related", which is easier to learn than the exact gate set.
    """
    if not hasattr(data, "subgraph_id") or not hasattr(data, "y"):
        return data

    component = data.subgraph_id
    num_components = int(component.max().item()) + 1
    sizes = torch.bincount(component, minlength=num_components).clamp_min(1)

    positives = torch.zeros(num_components, dtype=torch.long)
    positives.index_add_(0, component, (data.y == 1).long())

    ratio = positives.float() / sizes.float()
    data.y_wcc = (ratio[component] >= pos_ratio_threshold).long()
    return data


# ---------------------------------------------------------------------------
# Random simulation
# ---------------------------------------------------------------------------
def simulate_on_data(data: Data, sim_count: int = SIM_COUNT_DEFAULT) -> Data:
    """Fill the five switching-activity columns by random simulation.

    Rare switching is one of the strongest Trojan signals there is: trigger
    logic is designed to stay quiet until a condition that essentially never
    occurs.  Random vectors are driven into the primary inputs, the
    combinational logic settles in topological order, flip-flops update, and
    the per-gate transition counts and run lengths are tallied.

    The RNG is seeded with 0, so the same netlist always yields the same
    features -- training and inference must not disagree here.
    """
    num_nodes = data.x.size(0)
    levels = getattr(data, "levels", None)
    if levels is None:
        return data

    forward: List[List[int]] = [[] for _ in range(num_nodes)]
    reverse: List[List[int]] = [[] for _ in range(num_nodes)]
    edges = getattr(data, "edges", None)
    if edges is not None and edges.numel():
        for u, v in edges.tolist():
            forward[u].append(v)
            reverse[v].append(u)

    gate_types = list(data.gate_types)
    dff_d_pred = getattr(data, "dff_d_pred", None)
    dff_d_pred = dff_d_pred.tolist() if dff_d_pred is not None else None

    value = [0] * num_nodes
    previous: List[Optional[int]] = [None] * num_nodes
    num_1to0 = [0] * num_nodes
    num_0to1 = [0] * num_nodes
    longest_1 = [0] * num_nodes
    longest_0 = [0] * num_nodes
    run_length = [0] * num_nodes
    total_1 = [0] * num_nodes

    # Evaluate shallow gates before deep ones so each gate sees settled inputs.
    order = [gid for gid, _ in sorted(enumerate(levels.tolist()),
                                      key=lambda pair: pair[1])]

    rng = random.Random(0)
    for _ in range(sim_count):
        # Drive the primary inputs: gates with no predecessor that are not
        # flip-flops are the design's combinational sources.
        for gid in range(num_nodes):
            if not reverse[gid] and gate_types[gid] != "dff":
                value[gid] = rng.randint(0, 1)

        for gid in order:
            gtype = gate_types[gid]
            preds = reverse[gid]
            if not preds:
                continue
            if gtype in ("and", "or", "xor", "nand", "nor", "xnor"):
                acc = value[preds[0]]
                for pred in preds[1:]:
                    if gtype in ("and", "nand"):
                        acc &= value[pred]
                    elif gtype in ("or", "nor"):
                        acc |= value[pred]
                    else:
                        acc ^= value[pred]
                if gtype in ("nand", "nor", "xnor"):
                    acc ^= 1
                value[gid] = acc
            elif gtype == "not":
                value[gid] = 0 if value[preds[0]] == 1 else 1
            elif gtype == "buf":
                value[gid] = value[preds[0]]
            # 'dff' is handled after the combinational sweep.

        # Flip-flops commit at the end of the cycle: Q takes the value D saw.
        for gid in range(num_nodes):
            if gate_types[gid] != "dff":
                continue
            if dff_d_pred and dff_d_pred[gid] != -1:
                value[gid] = value[dff_d_pred[gid]]
            elif reverse[gid]:
                value[gid] = value[reverse[gid][0]]

        for gid in range(num_nodes):
            current = value[gid]
            if previous[gid] is not None:
                if previous[gid] == 1 and current == 0:
                    num_1to0[gid] += 1
                if previous[gid] == 0 and current == 1:
                    num_0to1[gid] += 1
                run_length[gid] = run_length[gid] + 1 if current == previous[gid] else 1
            else:
                run_length[gid] = 1

            if current == 1:
                total_1[gid] += 1
                longest_1[gid] = max(longest_1[gid], run_length[gid])
            else:
                longest_0[gid] = max(longest_0[gid], run_length[gid])
            previous[gid] = current

    raw_tallies = {
        "num_1to0": num_1to0, "num_0to1": num_0to1,
        "longest_1": longest_1, "longest_0": longest_0, "total_1": total_1,
    }
    for name, values in raw_tallies.items():
        setattr(data, "raw_" + name, torch.tensor(values, dtype=torch.float))

    # Transition counts are bounded by the number of steps between vectors;
    # run lengths and the one-count are bounded by the vector count itself.
    steps = max(sim_count - 1, 1)
    scaled = {
        "num_1to0": _maybe_scale_sim("num_1to0", num_1to0, steps),
        "num_0to1": _maybe_scale_sim("num_0to1", num_0to1, steps),
        "longest_1": _maybe_scale_sim("longest_1", longest_1, sim_count),
        "longest_0": _maybe_scale_sim("longest_0", longest_0, sim_count),
        "total_1": _maybe_scale_sim("total_1", total_1, sim_count),
    }
    for name, values in scaled.items():
        column = FEATURE_INDEX[name]
        data.x[:, column] = torch.tensor(values, dtype=torch.float)
        if hasattr(data, "raw_feature_matrix"):
            data.raw_feature_matrix[:, column] = getattr(data, "raw_" + name).float()

    data.x = _apply_mask(data.x)
    data.feature_mask = _feature_mask()
    data.sim_count = sim_count
    return data


# ---------------------------------------------------------------------------
# Batch conversion and Excel export
# ---------------------------------------------------------------------------
def export_graph_to_excel(data: Data, excel_path: str) -> None:
    """Dump one graph to a spreadsheet -- a debugging aid, not part of training.

    Being able to eyeball raw feature values next to the label is how most of
    the feature bugs in this project were found.
    """
    if not _HAS_PANDAS:
        raise RuntimeError("pandas is required for Excel export")
    if not hasattr(data, "raw_feature_matrix"):
        raise RuntimeError("raw_feature_matrix missing; rebuild the graph")

    gate_names = list(data.gate_names)
    scaled = pd.DataFrame(data.x.numpy(), columns=FEATURE_ORDER)
    raw_matrix = data.raw_feature_matrix.numpy()

    frames = [pd.DataFrame({"gate_name": gate_names, "label": data.y.tolist()}),
              pd.DataFrame({"raw__" + n: raw_matrix[:, FEATURE_INDEX[n]]
                            for n in FEATURE_ORDER})]
    enabled = _enabled_feature_names()
    normalised = {"norm__" + n: scaled[n].values
                  for n in FEATURE_ORDER if n in enabled and _normalise_on(n)}
    if normalised:
        frames.append(pd.DataFrame(normalised))
    gates_sheet = pd.concat(frames, axis=1)

    if hasattr(data, "edges") and data.edges.numel() > 0:
        pairs = data.edges.numpy()
        src, dst = pairs[:, 0].tolist(), pairs[:, 1].tolist()
        edges_sheet = pd.DataFrame({
            "src_id": src, "src_name": [gate_names[i] for i in src],
            "dst_id": dst, "dst_name": [gate_names[i] for i in dst],
        })
    else:
        edges_sheet = pd.DataFrame(
            columns=["src_id", "src_name", "dst_id", "dst_name"])

    io_sheet = pd.DataFrame({
        "inputs": ["\n".join(getattr(data, "inputs", []))],
        "outputs": ["\n".join(getattr(data, "outputs", []))],
    })

    with pd.ExcelWriter(excel_path) as writer:
        gates_sheet.to_excel(writer, sheet_name="Gates", index=False)
        edges_sheet.to_excel(writer, sheet_name="Edges", index=False)
        io_sheet.to_excel(writer, sheet_name="IO", index=False)


def build_graphs(netlist_dir: str, label_dir: Optional[str], out_dir: str,
                 excel_dir: Optional[str] = None,
                 sim_count: int = SIM_COUNT_DEFAULT,
                 simulate: bool = True) -> Tuple[int, int]:
    """Convert every netlist in a directory; return ``(converted, failed)``.

    One bad netlist never aborts the batch -- the corpus is large enough that
    losing a run to a single parse failure would be expensive.
    """
    os.makedirs(out_dir, exist_ok=True)
    if excel_dir:
        os.makedirs(excel_dir, exist_ok=True)

    converted = failed = 0
    for stem, netlist_path in dataset.iter_netlists(netlist_dir):
        label_path = dataset.find_label(label_dir, stem)
        try:
            data = netlist_to_graph(netlist_path, label_path, sim_count=sim_count)
            if simulate:
                data = simulate_on_data(data, sim_count=sim_count)
            torch.save(data, os.path.join(out_dir, stem + ".pt"))
            if excel_dir and _HAS_PANDAS:
                export_graph_to_excel(data, os.path.join(excel_dir, stem + ".xlsx"))

            trojan_gates = int((data.y == 1).sum().item())
            matched = os.path.basename(label_path) if label_path else "no label file"
            print("[ok]  " + stem + " -> " + str(data.x.size(0)) + " gates, " +
                  str(trojan_gates) + " Trojan (" + matched + ")")
            converted += 1
        except Exception as error:                       # noqa: BLE001
            print("[err] " + stem + ": " + str(error))
            traceback.print_exc()
            failed += 1

    return converted, failed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert gate-level netlists into PyTorch Geometric graphs.")
    parser.add_argument("--netlists", required=True,
                        help="directory of .v netlists")
    parser.add_argument("--labels", default=None,
                        help="directory of contest label files (optional)")
    parser.add_argument("--out", required=True,
                        help="directory to write .pt graphs into")
    parser.add_argument("--excel", default=None,
                        help="also dump per-design spreadsheets here (needs pandas)")
    parser.add_argument("--sims", type=int, default=SIM_COUNT_DEFAULT,
                        help="random vectors per design (default: %(default)s)")
    parser.add_argument("--no-sim", action="store_true",
                        help="skip simulation; leaves the tally columns at zero")
    args = parser.parse_args()

    converted, failed = build_graphs(
        netlist_dir=args.netlists,
        label_dir=args.labels,
        out_dir=args.out,
        excel_dir=args.excel if _HAS_PANDAS else None,
        sim_count=args.sims,
        simulate=not args.no_sim,
    )
    print("")
    print("converted " + str(converted) + " graph(s) into " + args.out +
          (", " + str(failed) + " failed" if failed else ""))


if __name__ == "__main__":
    main()
