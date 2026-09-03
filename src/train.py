"""Train the Trojan-detection GNN on pre-built graphs.

Reads the ``.pt`` graphs produced by ``build_graph.py``, trains
``gnn.TrojanGNN`` on them, and writes the best checkpoint out.

Two decisions worth knowing about
---------------------------------
**Batch size is 1, deliberately.**  ``data.subgraph_id`` indexes into that
design's own ``subgraph_feat`` table.  PyG's default collation concatenates
node features but would not re-base those component indices, so a larger batch
would silently pair gates with the wrong component embeddings.  One graph per
step keeps the indexing correct.

**Model selection is by F1, not loss.**  The contest scores per-gate F1 on
Trojaned designs, and with a ~9:1 class imbalance the lowest-loss epoch is
routinely not the best-F1 epoch.  Loss is only the tie-break.

Reported metrics are on the training set.  Held-out evaluation is
``predict.py``'s job -- see ``data/holdout/``.

Usage
-----
    python src/train.py --graphs build/graphs --out models/trojan_gnn.pt
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

import build_graph
from gnn import FocalLoss, build_model

# Defaults reproduce the run that produced the shipped checkpoint.
DEFAULT_EPOCHS = 1000
DEFAULT_LR = 8e-5
DEFAULT_PATIENCE = 100
DEFAULT_THRESHOLD = 0.5

# Below this positive rate the dataset counts as heavily imbalanced and the
# focal-loss alpha is raised to weight the rare class harder.
IMBALANCE_TRIGGER = 0.2
ALPHA_IMBALANCED = 0.8
ALPHA_BALANCED = 0.5

# Loss must improve by more than this to count as progress for early stopping.
LOSS_EPSILON = 1e-4


def load_graphs(graph_dir: str) -> List:
    """Load every ``.pt`` graph, repairing subgraph annotations if missing.

    Graphs built by an older revision of ``build_graph.py`` may predate the
    subgraph branch; rather than force a full rebuild of the corpus, those are
    annotated on the fly.
    """
    if not os.path.isdir(graph_dir):
        raise SystemExit("no such directory: " + graph_dir)

    graphs = []
    for name in sorted(os.listdir(graph_dir)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(graph_dir, name)
        try:
            data = torch.load(path, weights_only=False)

            if data.x is None or data.x.size(0) == 0:
                print("[skip] " + name + ": empty graph")
                continue
            for attribute in ("edge_index_fw", "edge_index_bw", "y"):
                if not hasattr(data, attribute):
                    raise AttributeError("missing " + attribute)

            if not (hasattr(data, "subgraph_feat") and hasattr(data, "subgraph_id")):
                data = build_graph.attach_subgraphs(data)

            graphs.append(data)
        except Exception as error:                       # noqa: BLE001
            print("[skip] " + name + ": " + str(error))

    return graphs


def evaluate(model, loader, threshold: float, device) -> dict:
    """Score the current model over a loader, returning the usual metrics."""
    model.eval()
    probabilities: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            positive = F.softmax(model(data), dim=1)[:, 1]
            probabilities.extend(positive.cpu().tolist())
            labels.extend(data.y.cpu().tolist())

    predictions = [1 if p > threshold else 0 for p in probabilities]
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    total = tp + fp + fn + tn

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": f1_score(labels, predictions, zero_division=1),
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def train(graphs: List, epochs: int, lr: float, patience: int,
          threshold: float, dropout: float, device: Optional[str] = None):
    """Run the training loop and return the best model seen."""
    if not graphs:
        raise SystemExit("no graphs to train on -- run build_graph.py first")

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(node_in=graphs[0].x.size(1),
                        sg_in=graphs[0].subgraph_feat.size(1),
                        dropout=dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # See the module docstring for why this is 1.
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    all_labels = torch.cat([g.y for g in graphs])
    positives = int((all_labels == 1).sum())
    negatives = int((all_labels == 0).sum())
    positive_rate = positives / max(1, positives + negatives)
    alpha = (ALPHA_IMBALANCED
             if positives and negatives and positive_rate < IMBALANCE_TRIGGER
             else ALPHA_BALANCED)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    print("graphs: " + str(len(graphs)) +
          " | gates: " + str(positives + negatives) +
          " | Trojan gates: " + str(positives) +
          " (" + format(positive_rate, ".2%") + ")" +
          " | focal alpha: " + str(alpha))
    print("device: " + str(device))
    print("")

    best_f1 = -float("inf")
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for data in loader:
            data = data.to(device)
            loss = criterion(model(data), data.y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        metrics = evaluate(model, loader, threshold, device)
        print("epoch " + str(epoch).rjust(4) +
              " | loss " + format(epoch_loss, "8.4f") +
              " | F1 " + format(metrics["f1"], ".4f") +
              " | P " + format(metrics["precision"], ".4f") +
              " | R " + format(metrics["recall"], ".4f") +
              " | TP " + str(metrics["tp"]) + " FP " + str(metrics["fp"]) +
              " FN " + str(metrics["fn"]))

        # Both decisions are made against the PREVIOUS best, so they must be
        # evaluated before best_f1 / best_loss are moved.
        f1_improved = metrics["f1"] > best_f1
        f1_tied = abs(metrics["f1"] - best_f1) <= 1e-12

        # Keep the highest-F1 model; break ties on the lower loss.
        is_better = f1_improved or (f1_tied and epoch_loss < best_loss - 1e-12)
        # Early stopping demands a *meaningful* loss drop on a tie, so tiny
        # numerical wobble cannot keep a stalled run alive forever.
        is_progress = f1_improved or (f1_tied and
                                      epoch_loss < best_loss - LOSS_EPSILON)

        if is_better:
            best_f1 = metrics["f1"]
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            print("        checkpoint: F1=" + format(best_f1, ".4f") +
                  " loss=" + format(best_loss, ".4f"))

        stale_epochs = 0 if is_progress else stale_epochs + 1
        if stale_epochs >= patience:
            print("early stop at epoch " + str(epoch) + ": no improvement in " +
                  str(patience) + " epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Trojan-detection GNN on pre-built graphs.")
    parser.add_argument("--graphs", required=True,
                        help="directory of .pt graphs from build_graph.py")
    parser.add_argument("--out", default="models/trojan_gnn.pt",
                        help="where to write the trained checkpoint")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default=None,
                        help="cpu / cuda (default: cuda when available)")
    args = parser.parse_args()

    graphs = load_graphs(args.graphs)
    if not graphs:
        raise SystemExit("no usable graphs in " + args.graphs)
    print("loaded " + str(len(graphs)) + " graphs | node features " +
          str(graphs[0].x.size(1)) + " | subgraph features " +
          str(graphs[0].subgraph_feat.size(1)))

    model, best_f1 = train(graphs, args.epochs, args.lr, args.patience,
                           args.threshold, args.dropout, args.device)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("")
    print("best training F1 " + format(best_f1, ".4f") +
          " -- saved to " + args.out)


if __name__ == "__main__":
    main()
