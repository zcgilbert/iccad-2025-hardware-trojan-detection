"""Dataset conventions shared by every script in this project.

Why this module exists
----------------------
A netlist (``*.v``) and its label file (``*.txt``) are stored side by side, but
three different naming conventions ended up in circulation during the contest:

1. ``<stem>.txt``                  -- same stem as the netlist (our own corpus)
2. ``designN.v`` -> ``resultN.txt`` -- the OFFICIAL contest convention
3. ``<prefix>_netlist.v`` -> ``<prefix>_results.txt`` -- emitted by the
   synthesis pipelines in ``synthesis/``

Rule 2 is fixed by the contest specification and cannot be changed, so a
resolver is genuinely necessary.  What was *not* necessary was having four
copies of it: the original submission re-implemented this lookup inside
``build_graph.py``, ``test.py`` and ``augment.py`` independently, and the three
copies had drifted apart.  It now lives here, once.

The label file format is also defined here.  It is the contest's own format::

    NO_TROJAN

or::

    TROJANED
    TROJAN_GATES
    g12
    g47
    END_TROJAN_GATES
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Set, Tuple

# ``designN`` is the only stem shape that gets the ``resultN`` treatment.
_DESIGN_STEM_RE = re.compile(r"^design(\d+)$")

_NETLIST_SUFFIX = "_netlist"

# Contest label-file keywords.
LABEL_TROJANED = "TROJANED"
LABEL_CLEAN = "NO_TROJAN"
_GATES_BEGIN = "TROJAN_GATES"
_GATES_END = "END_TROJAN_GATES"


def label_candidates(stem: str) -> List[str]:
    """Return every label filename that could belong to the netlist ``stem``.

    Ordered most-specific first, so an exact ``<stem>.txt`` always wins over a
    convention-derived guess.
    """
    names = [stem + ".txt"]

    design = _DESIGN_STEM_RE.match(stem)
    if design:
        names.append("result" + design.group(1) + ".txt")

    if stem.endswith(_NETLIST_SUFFIX):
        names.append(stem[: -len(_NETLIST_SUFFIX)] + "_results.txt")

    return names


def find_label(label_dir: Optional[str], stem: str) -> Optional[str]:
    """Locate the label file for one netlist, or ``None`` if there is none.

    Labels are optional throughout the project: inference runs happily without
    them, and only reports a score when they happen to be present.
    """
    if not label_dir or not os.path.isdir(label_dir):
        return None
    for name in label_candidates(stem):
        path = os.path.join(label_dir, name)
        if os.path.isfile(path):
            return path
    return None


def read_label(path: Optional[str]) -> Tuple[bool, Set[str]]:
    """Parse a contest label file into ``(is_trojaned, trojan_gate_names)``.

    A missing, unreadable or malformed file is treated as "no Trojan", which is
    the safe default: it never invents positive labels that the model would
    then be trained or scored against.
    """
    if not path or not os.path.isfile(path):
        return False, set()

    try:
        with open(path, "r", errors="replace") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError:
        return False, set()

    if not lines or lines[0] != LABEL_TROJANED:
        return False, set()
    if _GATES_BEGIN not in lines or _GATES_END not in lines:
        # Declared TROJANED but the gate list is missing or truncated.
        return True, set()

    begin = lines.index(_GATES_BEGIN) + 1
    end = lines.index(_GATES_END)
    return True, {name for name in lines[begin:end] if name}


def write_label(path: str, trojan_gates: List[str]) -> None:
    """Write one prediction in the exact format the contest scorer expects."""
    with open(path, "w") as handle:
        if not trojan_gates:
            handle.write(LABEL_CLEAN + "\n")
            return
        handle.write(LABEL_TROJANED + "\n")
        handle.write(_GATES_BEGIN + "\n")
        for gate in trojan_gates:
            handle.write(gate + "\n")
        handle.write(_GATES_END + "\n")


def output_name_for(stem: str) -> str:
    """Name the prediction file for a netlist stem.

    Mirrors :func:`label_candidates` so that a prediction lands where the
    scorer -- and a later run of this same code -- expects to find it.
    """
    design = _DESIGN_STEM_RE.match(stem)
    if design:
        return "result" + design.group(1) + ".txt"
    if stem.endswith(_NETLIST_SUFFIX):
        return stem[: -len(_NETLIST_SUFFIX)] + "_results.txt"
    return stem + ".txt"


def iter_netlists(netlist_dir: str) -> List[Tuple[str, str]]:
    """List ``(stem, path)`` for every ``.v`` file in a directory, sorted.

    Sorting keeps every run of every script in a stable, comparable order --
    which matters when diffing two prediction runs against each other.
    """
    if not os.path.isdir(netlist_dir):
        raise SystemExit("not a directory: " + netlist_dir)
    out = []
    for name in sorted(os.listdir(netlist_dir)):
        if name.endswith(".v"):
            out.append((name[:-2], os.path.join(netlist_dir, name)))
    return out
