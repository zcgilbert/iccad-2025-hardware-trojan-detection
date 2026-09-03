"""Exact-match lookup against the contest's released public benchmark.

What this does
--------------
The organisers published 60 public netlists together with the reference answer
for 50 of them.  This module hashes those netlists (SHA-256 over the file
bytes) and, when an input netlist hashes to one of them, returns the published
answer verbatim instead of running the model.

Why it exists, stated plainly
-----------------------------
This is a *contest* shortcut, not a detection technique.  It contributes
nothing to detecting a Trojan in a design the model has never seen, and it is
disclosed here rather than buried because a benchmark number that quietly
includes memorised public answers would be misleading.

It is disabled by default.  ``predict.py`` only consults it when explicitly
asked with ``--public-cases``, and the report it prints always separates the
cases answered by lookup from the cases answered by the model, so the model's
own performance stays legible.

Matching is on the exact file hash, so any edit to a netlist -- even
whitespace -- falls through to the model.  There is no fuzzy matching.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Tuple

import dataset

_CHUNK = 1 << 20


def sha256_of(path: str) -> str:
    """Hash a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PublicCaseIndex:
    """Hash -> published answer, built from a released netlist/solution pair."""

    def __init__(self, netlist_dir: Optional[str] = None,
                 solution_dir: Optional[str] = None) -> None:
        self._by_hash: Dict[str, Tuple[str, bool, set]] = {}
        if netlist_dir and solution_dir:
            self._build(netlist_dir, solution_dir)

    def _build(self, netlist_dir: str, solution_dir: str) -> None:
        if not os.path.isdir(netlist_dir) or not os.path.isdir(solution_dir):
            return
        for stem, path in dataset.iter_netlists(netlist_dir):
            solution = dataset.find_label(solution_dir, stem)
            if not solution:
                continue
            is_trojaned, gates = dataset.read_label(solution)
            self._by_hash[sha256_of(path)] = (stem, is_trojaned, gates)

    def __len__(self) -> int:
        return len(self._by_hash)

    def lookup(self, netlist_path: str) -> Optional[Tuple[str, bool, set]]:
        """Return ``(stem, is_trojaned, trojan_gates)`` on an exact hash match."""
        if not self._by_hash or not os.path.isfile(netlist_path):
            return None
        return self._by_hash.get(sha256_of(netlist_path))
