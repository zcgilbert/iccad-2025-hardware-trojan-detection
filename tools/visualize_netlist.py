#!/usr/bin/env python3
"""Draw a netlist as an SVG, colouring gates by detection outcome.

A confusion matrix says *how many* gates were missed.  This says *which ones*,
and what they were attached to -- which is the question you actually want
answered when a design scores badly.  Gates are laid out left to right by
combinational depth and coloured:

    green   true positive   -- Trojan gate, correctly flagged
    red     false negative  -- Trojan gate, missed
    orange  false positive  -- clean gate, wrongly flagged
    grey    true negative   -- clean gate, correctly ignored

Give it only ground truth and it simply shows where the Trojan lives.  Give it
only a prediction and it shows what the model flagged.

Real designs run to thousands of gates, which is unreadable at any zoom, so by
default the drawing is cropped to the neighbourhood of the interesting gates
(``--context`` hops around anything flagged or labelled).  ``--full`` overrides
that.

Dependency-free: standard library only, no PyTorch.  Open the output in any
browser.

Usage
-----
    python tools/visualize_netlist.py \\
        --netlist data/holdout/netlists/design28.v \\
        --truth   data/holdout/labels/result28.txt \\
        --predicted build/predictions/result28.txt \\
        --out design28.svg
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import dataset                    # noqa: E402
from netlist_parser import parse_netlist   # noqa: E402

# Outcome -> (fill, stroke, legend text)
COLOURS = {
    "tp": ("#2e9e4f", "#1c6531", "correctly flagged Trojan gate"),
    "fn": ("#d64545", "#8f2020", "missed Trojan gate"),
    "fp": ("#e08b2e", "#96590f", "false alarm"),
    "tn": ("#c9ced6", "#9aa1ac", "clean gate"),
}
BACKGROUND = "#ffffff"
EDGE_COLOUR = "#b7bec9"
EDGE_HOT_COLOUR = "#7b8494"
TEXT_COLOUR = "#2b3038"

LAYER_GAP = 110      # horizontal spacing between depth layers
ROW_GAP = 26         # vertical spacing within a layer
RADIUS = 7
MARGIN = 40
MAX_DRAWN_GATES = 1200
BARYCENTRE_SWEEPS = 4


def combinational_levels(num_gates: int, gate_type: Dict[int, str],
                         forward: List[List[int]],
                         reverse: List[List[int]]) -> List[int]:
    """Longest-path depth per gate, not crossing flip-flops.

    Gates in a combinational loop never reach in-degree zero; they keep depth 0
    rather than being dropped, so nothing vanishes from the picture.
    """
    is_comb = [gate_type[i] != "dff" for i in range(num_gates)]
    level = [0] * num_gates
    indegree = [sum(1 for p in reverse[i] if is_comb[p]) if is_comb[i] else 0
                for i in range(num_gates)]

    queue = deque(i for i in range(num_gates) if is_comb[i] and indegree[i] == 0)
    while queue:
        u = queue.popleft()
        for v in forward[u]:
            if not is_comb[v]:
                continue
            level[v] = max(level[v], level[u] + 1)
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return level


def neighbourhood(seeds: Set[int], forward: List[List[int]],
                  reverse: List[List[int]], hops: int) -> Set[int]:
    """Every gate within ``hops`` undirected steps of a seed."""
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt: Set[int] = set()
        for u in frontier:
            for v in forward[u] + reverse[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen


def order_rows(visible: List[int], level: List[int],
               reverse: List[List[int]]) -> Dict[int, Tuple[int, int]]:
    """Assign each visible gate a (layer, row), reducing edge crossings.

    A few barycentre sweeps -- put each gate near the average row of its
    drivers -- is enough to make a layered drawing readable without pulling in
    a graph-layout library.
    """
    layers: Dict[int, List[int]] = {}
    for gid in visible:
        layers.setdefault(level[gid], []).append(gid)
    for gids in layers.values():
        gids.sort()

    row_of = {gid: i for gids in layers.values() for i, gid in enumerate(gids)}
    visible_set = set(visible)

    for _ in range(BARYCENTRE_SWEEPS):
        for depth in sorted(layers):
            gids = layers[depth]

            def barycentre(gid: int) -> float:
                drivers = [p for p in reverse[gid] if p in visible_set]
                if not drivers:
                    return float(row_of[gid])
                return sum(row_of[p] for p in drivers) / len(drivers)

            gids.sort(key=barycentre)
            for i, gid in enumerate(gids):
                row_of[gid] = i

    return {gid: (level[gid], row_of[gid]) for gid in visible}


def classify(gid: int, name: str, truth: Optional[Set[str]],
             predicted: Optional[Set[str]]) -> str:
    """Outcome bucket for one gate."""
    is_true = truth is not None and name in truth
    is_pred = predicted is not None and name in predicted
    if truth is None:
        return "tp" if is_pred else "tn"
    if predicted is None:
        return "fn" if is_true else "tn"
    if is_true and is_pred:
        return "tp"
    if is_true:
        return "fn"
    if is_pred:
        return "fp"
    return "tn"


def render_svg(title: str, positions: Dict[int, Tuple[int, int]],
               gate_names: List[str], gate_type: Dict[int, str],
               outcome: Dict[int, str], forward: List[List[int]],
               counts: Dict[str, int], cropped: bool,
               total_gates: int) -> str:
    """Build the SVG document."""
    layers = sorted({layer for layer, _ in positions.values()})
    layer_x = {layer: MARGIN + i * LAYER_GAP for i, layer in enumerate(layers)}
    max_row = max((row for _, row in positions.values()), default=0)

    width = MARGIN * 2 + max(1, len(layers) - 1) * LAYER_GAP
    height = MARGIN * 2 + max_row * ROW_GAP + 110      # room for the legend

    def xy(gid: int) -> Tuple[float, float]:
        layer, row = positions[gid]
        return layer_x[layer], MARGIN + 60 + row * ROW_GAP

    parts: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 ' + str(width) + ' ' + str(height) + '" '
        'width="' + str(width) + '" height="' + str(height) + '" '
        'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif">',
        '<rect width="100%" height="100%" fill="' + BACKGROUND + '"/>',
        '<text x="' + str(MARGIN) + '" y="28" font-size="17" font-weight="600" '
        'fill="' + TEXT_COLOUR + '">' + html.escape(title) + '</text>',
    ]

    tally = ", ".join(str(counts[k]) + " " + k.upper()
                      for k in ("tp", "fn", "fp", "tn") if counts.get(k))
    # Cropping is seeded from every non-TN gate, so TP/FN/FP counts are always
    # complete for the whole design; only the TN count is a subset.
    scope = (" of " + str(total_gates) + " (TP/FN/FP complete, TN sampled)"
             if cropped else " (whole design)")
    subtitle = str(len(positions)) + " gates drawn" + scope + " - " + tally
    parts.append('<text x="' + str(MARGIN) + '" y="48" font-size="12" '
                 'fill="#5b626d">' + html.escape(subtitle) + '</text>')

    # Edges first, so nodes draw on top.
    parts.append('<g stroke-width="1" fill="none">')
    for source in positions:
        x1, y1 = xy(source)
        for target in forward[source]:
            if target not in positions:
                continue
            x2, y2 = xy(target)
            hot = outcome[source] in ("tp", "fn") or outcome[target] in ("tp", "fn")
            colour = EDGE_HOT_COLOUR if hot else EDGE_COLOUR
            # Cubic curve so parallel edges stay visually distinguishable.
            control = (x1 + x2) / 2
            parts.append(
                '<path d="M' + format(x1, ".1f") + ',' + format(y1, ".1f") +
                ' C' + format(control, ".1f") + ',' + format(y1, ".1f") +
                ' ' + format(control, ".1f") + ',' + format(y2, ".1f") +
                ' ' + format(x2, ".1f") + ',' + format(y2, ".1f") + '" ' +
                'stroke="' + colour + '" opacity="' +
                ("0.75" if hot else "0.4") + '"/>')
    parts.append('</g>')

    for gid in positions:
        x, y = xy(gid)
        fill, stroke, _ = COLOURS[outcome[gid]]
        radius = RADIUS + (2 if outcome[gid] in ("tp", "fn") else 0)
        label = (gate_names[gid] + "  (" + gate_type[gid] + ")  " +
                 outcome[gid].upper())
        parts.append(
            '<circle cx="' + format(x, ".1f") + '" cy="' + format(y, ".1f") +
            '" r="' + str(radius) + '" fill="' + fill + '" stroke="' + stroke +
            '" stroke-width="1.5"><title>' + html.escape(label) +
            '</title></circle>')
        # Flip-flops get a marker, since they are the sequential boundaries.
        if gate_type[gid] == "dff":
            parts.append(
                '<rect x="' + format(x - 3, ".1f") + '" y="' +
                format(y - 3, ".1f") + '" width="6" height="6" '
                'fill="#ffffff" opacity="0.85"/>')

    legend_y = height - 34
    x_cursor = MARGIN
    for key in ("tp", "fn", "fp", "tn"):
        fill, stroke, description = COLOURS[key]
        text = description + " (" + str(counts.get(key, 0)) + ")"
        parts.append(
            '<circle cx="' + str(x_cursor + 7) + '" cy="' + str(legend_y) +
            '" r="6" fill="' + fill + '" stroke="' + stroke + '" stroke-width="1.5"/>')
        parts.append(
            '<text x="' + str(x_cursor + 20) + '" y="' + str(legend_y + 4) +
            '" font-size="12" fill="#5b626d">' + html.escape(text) + '</text>')
        x_cursor += 26 + int(len(text) * 6.4)

    parts.append('<text x="' + str(MARGIN) + '" y="' + str(height - 12) +
                 '" font-size="11" fill="#8a919c">'
                 'left to right = combinational depth; squares mark flip-flops; '
                 'open this file directly to hover for gate names</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a netlist to SVG, colouring gates by detection outcome.")
    parser.add_argument("--netlist", required=True, help="the .v design to draw")
    parser.add_argument("--truth", default=None,
                        help="ground-truth label file")
    parser.add_argument("--predicted", default=None,
                        help="prediction file from predict.py")
    parser.add_argument("--out", default=None,
                        help="output .svg (default: alongside the netlist)")
    parser.add_argument("--context", type=int, default=3,
                        help="hops of surrounding logic to keep (default: %(default)s)")
    parser.add_argument("--full", action="store_true",
                        help="draw the whole design instead of cropping")
    parser.add_argument("--max-gates", type=int, default=MAX_DRAWN_GATES,
                        help="refuse to draw more than this (default: %(default)s)")
    args = parser.parse_args()

    parsed = parse_netlist(args.netlist)
    if parsed.num_gates == 0:
        raise SystemExit("no gates found in " + args.netlist +
                         " -- is it a synthesised gate-level netlist?")

    _, forward, reverse = parsed.adjacency()
    level = combinational_levels(parsed.num_gates, parsed.gate_type,
                                 forward, reverse)

    _, truth = dataset.read_label(args.truth) if args.truth else (False, None)
    _, predicted = dataset.read_label(args.predicted) if args.predicted else (False, None)
    truth = truth if args.truth else None
    predicted = predicted if args.predicted else None

    outcome = {gid: classify(gid, name, truth, predicted)
               for gid, name in enumerate(parsed.gate_names)}

    interesting = {gid for gid, kind in outcome.items() if kind != "tn"}
    if args.full or not interesting:
        visible = list(range(parsed.num_gates))
        cropped = False
    else:
        visible = sorted(neighbourhood(interesting, forward, reverse, args.context))
        cropped = len(visible) < parsed.num_gates

    if len(visible) > args.max_gates:
        raise SystemExit(
            str(len(visible)) + " gates is too many to draw legibly. "
            "Lower --context, or raise --max-gates if you really want it.")

    counts: Dict[str, int] = {}
    for gid in visible:
        counts[outcome[gid]] = counts.get(outcome[gid], 0) + 1

    positions = order_rows(visible, level, reverse)
    title = os.path.basename(args.netlist)
    svg = render_svg(title, positions, parsed.gate_names, parsed.gate_type,
                     outcome, forward, counts, cropped, parsed.num_gates)

    out_path = args.out or os.path.splitext(args.netlist)[0] + ".svg"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(svg)

    print("drew " + str(len(visible)) + " of " + str(parsed.num_gates) +
          " gates -> " + out_path)
    for key in ("tp", "fn", "fp", "tn"):
        if counts.get(key):
            print("  " + key.upper() + ": " + str(counts[key]) +
                  "  " + COLOURS[key][2])


if __name__ == "__main__":
    main()
