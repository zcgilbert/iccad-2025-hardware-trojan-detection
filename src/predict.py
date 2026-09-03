"""Run the trained model over netlists and write contest-format predictions.

For each ``.v`` file in the input directory this builds the graph, runs the
GNN, applies three structural post-filters, and writes a
``TROJANED``/``NO_TROJAN`` result file.  When label files are available it also
reports the contest score.

The post-filters
----------------
Raw per-gate output is noisy: the network fires on scattered individual gates
that look locally odd.  A real Trojan is never one isolated gate -- it is a
connected block of trigger and payload logic.  All three filters exploit that,
and they run in this order:

1. **Isolated-gate removal** -- drop a positive with no positive within
   ``--n-hop`` hops.  Kills salt-and-pepper false positives.
2. **Small-cluster removal** -- drop connected groups of positives smaller than
   ``--min-group``.  Kills the small clusters that survive step 1.
3. **Minimum-total gate** -- if fewer than ``--min-total`` positives survive,
   declare the whole design clean.

Step 3 is a direct response to the contest's scoring function: a correct
Trojaned/clean verdict is worth 2 points and per-gate F1 adds at most 1 more,
so a handful of low-confidence positives on a clean design is a bad trade -- it
throws away 2 points to chase a fraction of one.

Usage
-----
    python src/predict.py --netlists data/holdout/netlists \
                          --labels   data/holdout/labels \
                          --model    models/trojan_gnn.pt \
                          --out      build/predictions
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import build_graph
import dataset
from gnn import load_model
from public_case_lookup import PublicCaseIndex

DEFAULT_THRESHOLD = 0.5
DEFAULT_N_HOP = 1
DEFAULT_MIN_GROUP = 5
DEFAULT_MIN_TOTAL = 10

# Contest scoring: 2 points for the right Trojaned/clean verdict, plus the
# per-gate F1 as a bonus on correctly-identified Trojaned designs.
CLASSIFICATION_POINTS = 2


def _undirected_neighbours(edge_index: torch.Tensor,
                           num_nodes: int) -> List[List[int]]:
    """Adjacency list treating the forward edges as undirected.

    Trojan logic is a connected block regardless of signal direction, so the
    post-filters ignore edge orientation.
    """
    neighbours: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        neighbours[u].append(v)
        neighbours[v].append(u)
    return neighbours


def drop_isolated(prediction: torch.Tensor, edge_index: torch.Tensor,
                  n_hop: int = DEFAULT_N_HOP) -> torch.Tensor:
    """Clear any positive with no other positive within ``n_hop`` hops."""
    num_nodes = prediction.size(0)
    neighbours = _undirected_neighbours(edge_index, num_nodes)
    adjusted = prediction.clone()

    for gid in range(num_nodes):
        if prediction[gid].item() != 1:
            continue
        seen = {gid}
        frontier = set(neighbours[gid])
        for _ in range(n_hop):
            nxt = set()
            for node in frontier:
                if node not in seen:
                    seen.add(node)
                    nxt.update(neighbours[node])
            frontier = nxt
        seen.discard(gid)
        if not any(prediction[other].item() == 1 for other in seen):
            adjusted[gid] = 0

    return adjusted


def drop_small_clusters(prediction: torch.Tensor, edge_index: torch.Tensor,
                        min_group: int = DEFAULT_MIN_GROUP) -> torch.Tensor:
    """Clear connected groups of positives smaller than ``min_group``."""
    num_nodes = prediction.size(0)
    neighbours = _undirected_neighbours(edge_index, num_nodes)
    adjusted = prediction.clone()
    visited = [False] * num_nodes

    for gid in range(num_nodes):
        if adjusted[gid].item() != 1 or visited[gid]:
            continue
        visited[gid] = True
        group = [gid]
        stack = [gid]
        while stack:
            node = stack.pop()
            for neighbour in neighbours[node]:
                if not visited[neighbour] and adjusted[neighbour].item() == 1:
                    visited[neighbour] = True
                    stack.append(neighbour)
                    group.append(neighbour)
        if len(group) < min_group:
            for node in group:
                adjusted[node] = 0

    return adjusted


def enforce_min_total(prediction: torch.Tensor,
                      min_total: int = DEFAULT_MIN_TOTAL) -> torch.Tensor:
    """Declare the design clean unless at least ``min_total`` gates survive."""
    if prediction.sum().item() < min_total:
        return torch.zeros_like(prediction)
    return prediction


def score_case(prediction: torch.Tensor,
               truth: Optional[torch.Tensor]) -> Optional[dict]:
    """Score one design the way the contest does; ``None`` without a label."""
    if truth is None:
        return None

    predicted_pairs = list(zip(prediction.tolist(), truth.tolist()))
    tp = sum(1 for p, t in predicted_pairs if p == 1 and t == 1)
    fp = sum(1 for p, t in predicted_pairs if p == 1 and t == 0)
    fn = sum(1 for p, t in predicted_pairs if p == 0 and t == 1)
    tn = sum(1 for p, t in predicted_pairs if p == 0 and t == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    predicted_trojaned = bool(prediction.sum().item() > 0)
    truly_trojaned = bool(truth.sum().item() > 0)
    classification = (CLASSIFICATION_POINTS
                      if predicted_trojaned == truly_trojaned else 0)
    # The F1 bonus only applies when a Trojaned design was correctly flagged.
    bonus = f1 if (truly_trojaned and classification == CLASSIFICATION_POINTS) else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "predicted_trojaned": predicted_trojaned,
        "truly_trojaned": truly_trojaned,
        "classification": classification,
        "score": classification + bonus,
    }


def run(netlist_dir: str, label_dir: Optional[str], model_path: str,
        out_dir: str, threshold: float, n_hop: int, min_group: int,
        min_total: int, sim_count: int,
        public_index: Optional[PublicCaseIndex] = None) -> Tuple[float, int]:
    """Predict over a whole directory; return ``(total_score, scored_cases)``."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "summary.txt")
    log_lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        log_lines.append(text)

    model = None
    total_score = 0.0
    scored = 0
    from_lookup = 0

    for stem, netlist_path in dataset.iter_netlists(netlist_dir):
        label_path = dataset.find_label(label_dir, stem)

        data = build_graph.netlist_to_graph(netlist_path, label_path,
                                            sim_count=sim_count)
        data = build_graph.simulate_on_data(data, sim_count=sim_count)
        gate_names = list(data.gate_names)

        # The model is built lazily so its input widths come from real data.
        if model is None:
            model = load_model(model_path,
                               node_in=data.x.size(1),
                               sg_in=data.subgraph_feat.size(1))

        matched = public_index.lookup(netlist_path) if public_index else None
        if matched is not None:
            _, is_trojaned, trojan_gates = matched
            prediction = torch.tensor(
                [1 if name in trojan_gates else 0 for name in gate_names],
                dtype=torch.long) if is_trojaned else torch.zeros(
                len(gate_names), dtype=torch.long)
            source = "public-case lookup"
            from_lookup += 1
        else:
            with torch.no_grad():
                positive = F.softmax(model(data), dim=1)[:, 1]
                prediction = (positive >= threshold).long()
            prediction = drop_isolated(prediction, data.edge_index_fw, n_hop)
            prediction = drop_small_clusters(prediction, data.edge_index_fw,
                                             min_group)
            prediction = enforce_min_total(prediction, min_total)
            source = "model"

        trojan_gates_out = [gate_names[i]
                            for i, flag in enumerate(prediction.tolist())
                            if flag == 1]
        dataset.write_label(
            os.path.join(out_dir, dataset.output_name_for(stem)),
            trojan_gates_out)

        emit("=== " + stem + " (" + source + ") ===")
        emit("  predicted : " +
             ("TROJANED, " + str(len(trojan_gates_out)) + " gates"
              if trojan_gates_out else "NO_TROJAN"))

        metrics = score_case(prediction, data.y if label_path else None)
        if metrics is None:
            emit("  ground truth not available -- prediction written, not scored")
        else:
            total_score += metrics["score"]
            scored += 1
            emit("  truth     : " +
                 ("TROJANED" if metrics["truly_trojaned"] else "NO_TROJAN"))
            emit("  TP " + str(metrics["tp"]) + "  FP " + str(metrics["fp"]) +
                 "  FN " + str(metrics["fn"]) + "  TN " + str(metrics["tn"]))
            emit("  precision " + format(metrics["precision"], ".4f") +
                 " | recall " + format(metrics["recall"], ".4f") +
                 " | F1 " + format(metrics["f1"], ".4f"))
            emit("  score     : " + format(metrics["score"], ".4f") +
                 " (classification " + str(metrics["classification"]) + " + F1 bonus)")
        emit()

    if scored:
        emit("scored " + str(scored) + " design(s): total " +
             format(total_score, ".4f") + " / " + str(scored * 3) +
             " (mean " + format(total_score / scored, ".4f") + ")")
    if from_lookup:
        emit("NOTE: " + str(from_lookup) + " of these came from the public-case "
             "lookup table, not the model.")

    with open(log_path, "w") as handle:
        handle.write("\n".join(log_lines) + "\n")
    print("wrote predictions and summary.txt to " + out_dir)
    return total_score, scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Trojan gates with the trained GNN.")
    parser.add_argument("--netlists", required=True,
                        help="directory of .v netlists to score")
    parser.add_argument("--labels", default=None,
                        help="directory of label files; enables scoring")
    parser.add_argument("--model", default="models/trojan_gnn.pt")
    parser.add_argument("--out", default="build/predictions")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="positive-class probability cut (default: %(default)s)")
    parser.add_argument("--n-hop", type=int, default=DEFAULT_N_HOP,
                        help="isolated-gate filter radius (default: %(default)s)")
    parser.add_argument("--min-group", type=int, default=DEFAULT_MIN_GROUP,
                        help="smallest surviving cluster (default: %(default)s)")
    parser.add_argument("--min-total", type=int, default=DEFAULT_MIN_TOTAL,
                        help="below this many gates, report NO_TROJAN "
                             "(default: %(default)s)")
    parser.add_argument("--sims", type=int, default=build_graph.SIM_COUNT_DEFAULT,
                        help="random vectors per design (default: %(default)s)")
    parser.add_argument("--public-cases", nargs=2,
                        metavar=("NETLIST_DIR", "SOLUTION_DIR"), default=None,
                        help="OFF by default. Answer exact hash matches from the "
                             "released public solutions instead of the model; see "
                             "public_case_lookup.py")
    args = parser.parse_args()

    index = None
    if args.public_cases:
        index = PublicCaseIndex(args.public_cases[0], args.public_cases[1])
        print("public-case lookup enabled: " + str(len(index)) + " indexed design(s)")

    run(netlist_dir=args.netlists,
        label_dir=args.labels,
        model_path=args.model,
        out_dir=args.out,
        threshold=args.threshold,
        n_hop=args.n_hop,
        min_group=args.min_group,
        min_total=args.min_total,
        sim_count=args.sims,
        public_index=index)


if __name__ == "__main__":
    main()
