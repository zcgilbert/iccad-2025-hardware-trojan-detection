"""The Trojan-detection network, shared by training and inference.

Originally this architecture was written out twice -- once in ``train.py`` and
once in ``test.py`` -- and the two copies had already started to drift.  Since
inference has to instantiate *exactly* the architecture the weights were
trained with, that duplication was a live correctness hazard, so the definition
now lives here and both entry points import it.

Architecture
------------
Two branches feed one classifier head, per gate:

* **Node branch** -- three ``BiGCNLayer`` blocks.  A gate-level netlist is a
  directed graph, and the useful signal flows *both* ways: what drives a gate
  and what the gate drives.  Each layer therefore runs one ``GCNConv`` over the
  forward edges and one over the reversed edges, then fuses the two.  The
  outputs of all three layers are concatenated (not just the last one), so the
  head can see 1-, 2- and 3-hop views of the neighbourhood at once.

* **Subgraph branch** -- an MLP over per-component descriptors.  Cutting the
  graph at flip-flop boundaries splits it into weakly-connected components; a
  Trojan payload typically forms its own small, weakly-attached component, so
  "which component am I in, and what does it look like" is a strong feature
  that no amount of local message passing recovers on its own.

Attribute names (``node_enc``, ``sg_enc``, ``head``, ``fw_conv``, ...) are load-
bearing: they are the keys inside ``models/trojan_gnn.pt``.  Renaming one
silently breaks weight loading, so leave them alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Defaults that the shipped checkpoint was trained with.  Inference reads the
# input dimensions from the data, but these three must match the checkpoint.
HIDDEN_CHANNELS = 64
SUBGRAPH_HIDDEN = 64
NUM_LAYERS = 3


class BiGCNLayer(nn.Module):
    """One bidirectional graph-convolution block.

    Convolves over the forward edge set and the reversed edge set separately,
    concatenates the two views, and projects them back down to
    ``out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fw_conv = GCNConv(in_channels, out_channels)
        self.bw_conv = GCNConv(in_channels, out_channels)
        self.linear = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index_fw, edge_index_bw):
        h_forward = self.fw_conv(x, edge_index_fw)
        h_backward = self.bw_conv(x, edge_index_bw)
        return F.relu(self.linear(torch.cat([h_forward, h_backward], dim=1)))


class NodeEncoder(nn.Module):
    """Stack of :class:`BiGCNLayer` blocks whose outputs are concatenated.

    Concatenating every layer's output rather than returning only the last one
    keeps the shallow (local) views available to the classifier: a Trojan
    trigger is often recognisable from its immediate neighbourhood, and deep
    message passing tends to smear that away.
    """

    def __init__(self, in_channels: int, hidden_channels: int = HIDDEN_CHANNELS,
                 num_layers: int = NUM_LAYERS, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BiGCNLayer(in_channels if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_channels * num_layers

    def forward(self, x, edge_index_fw, edge_index_bw):
        outputs = []
        h = x
        for layer in self.layers:
            h = self.dropout(layer(h, edge_index_fw, edge_index_bw))
            outputs.append(h)
        return torch.cat(outputs, dim=1)          # [num_nodes, hidden * layers]


class SubgraphEncoder(nn.Module):
    """Two-layer MLP over the per-component descriptor vector."""

    def __init__(self, in_dim: int, out_dim: int = SUBGRAPH_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, subgraph_feat):
        return self.net(subgraph_feat)            # [num_components, out_dim]


class TrojanGNN(nn.Module):
    """Per-gate binary classifier: Trojan gate or not.

    Each gate is represented by its node embedding concatenated with the
    embedding of the component it belongs to, and the pair is classified
    jointly.
    """

    def __init__(self, node_in: int, sg_in: int,
                 node_hidden: int = HIDDEN_CHANNELS,
                 sg_hidden: int = SUBGRAPH_HIDDEN,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = 0.0,
                 out_classes: int = 2) -> None:
        super().__init__()
        self.node_enc = NodeEncoder(node_in, node_hidden, num_layers, dropout)
        self.sg_enc = SubgraphEncoder(sg_in, sg_hidden)

        fused_dim = self.node_enc.out_dim + self.sg_enc.out_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, out_classes),
        )

    def forward(self, data):
        # NOTE: this indexing assumes one graph per batch.  `subgraph_id` holds
        # per-node component indices that are only valid within a single
        # design, so training uses batch_size=1 (see train.py).
        h_node = self.node_enc(data.x, data.edge_index_fw, data.edge_index_bw)
        z_component = self.sg_enc(data.subgraph_feat)[data.subgraph_id]
        return self.head(torch.cat([h_node, z_component], dim=1))


class FocalLoss(nn.Module):
    """Cross-entropy that down-weights easy examples.

    Trojan gates are a small minority of every design (the shipped corpus
    averages ~8% in host-plus-Trojan netlists), so plain cross-entropy is
    minimised by predicting "clean" everywhere.  Focal loss scales each term by
    ``(1 - p_t) ** gamma``, which collapses the contribution of the many easy
    negatives and lets the rare positives drive the gradient.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)                      # probability of the true class
        return (self.alpha * (1.0 - p_t) ** self.gamma * ce).mean()


def build_model(node_in: int, sg_in: int, dropout: float = 0.0) -> TrojanGNN:
    """Instantiate the architecture the shipped checkpoint expects."""
    return TrojanGNN(
        node_in=node_in,
        sg_in=sg_in,
        node_hidden=HIDDEN_CHANNELS,
        sg_hidden=SUBGRAPH_HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=dropout,
        out_classes=2,
    )


def load_model(checkpoint_path: str, node_in: int, sg_in: int,
               device=None) -> TrojanGNN:
    """Build the model and load trained weights, failing loudly on a mismatch.

    ``strict=True`` is deliberate: a silently partial load would produce a
    model that runs and returns confident nonsense.
    """
    model = build_model(node_in, sg_in)
    state = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    return model
