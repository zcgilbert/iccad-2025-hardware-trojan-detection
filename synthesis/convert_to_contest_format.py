#!/usr/bin/env python3
"""Rewrite a synthesised netlist into the contest's canonical netlist format.

Both Design Compiler and Genus emit valid Verilog, but not the shape the
contest specifies.  This normaliser bridges the gap:

* **Escaped identifiers.**  Synthesis preserves hierarchy in names such as
  ``\\core/alu/g27``.  Backslashes and slashes are replaced with underscores so
  every identifier is a plain word.
* **Anonymous ports.**  Cells come out with named connections
  (``.A(x), .Y(y)``).  Primitive gates are converted to the contest's
  positional form, output first: ``nand g7(y, a, b);``.
* **Flip-flop pin order.**  ``dff`` keeps named pins but in the fixed order
  ``.RN .SN .CK .D .Q`` so downstream parsing is positional in practice.
* **Signal renaming.**  Every net is renamed to ``n0, n1, n2, ...`` with the
  module ports first, in declaration order, so the interface stays stable.
* **Gate renaming.**  Instances become ``g0, g1, g2, ...`` in encounter order.
  This numbering is what the label files refer to, so
  ``extract_trojan_labels.py`` must walk gates in the same order -- it does.
* **Continuous assignments are dropped**, since the contest format is
  structural only.

The two synthesis flows differ in exactly one respect that matters here: the
suffix on the output filename, which is how the corpus keeps DC and Genus
results apart.  Everything else is shared, so ``--flow`` only selects that
suffix.

Usage
-----
    # whole directory
    python synthesis/convert_to_contest_format.py --flow dc \\
        --in build/dc/netlists --out build/dc/contest_netlists

    # single file
    python synthesis/convert_to_contest_format.py --in raw.v --out clean.v
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

PRIMITIVE_GATES = {"and", "or", "xor", "xnor", "nand", "nor", "not", "buf", "dff"}

# Filename suffix per flow.  The corpus relies on these to tell the two
# synthesis tools' outputs apart.
FLOW_SUFFIX = {"dc": "_netlist", "genus": "_g_netlist"}

# Backslash- or slash-containing identifiers, captured whole so the whole run
# can be flattened to underscores in one pass.
_ESCAPED_ID_RE = re.compile(
    r"[\\/]?(?P<id>[A-Za-z_][A-Za-z0-9_\\/]*)(?=[^A-Za-z0-9_]|$)")
_DECLARATION_RE = re.compile(r"^\s*(input|output|wire)")
_MODULE_RE = re.compile(r"^\s*module\s+\w+")
_MODULE_PORTS_RE = re.compile(r"^\s*module\s+\w+\s*(\(.*\);)")
_TOP_PORTS_RE = re.compile(r"module\s+top\s*\((.*?)\);")
_GATE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+\\?[A-Za-z0-9_\[\]/]*\s*\((.*)\);\s*$")
_PIN_RE = re.compile(r"\.(\w+)\(([^()]*)\)")
_BIT_SELECT_RE = re.compile(r"\[[^\]]+\]")


def _flatten_identifier(match: re.Match) -> str:
    return match.group(0).replace("\\", "_").replace("/", "_")


def _join_wrapped_lines(raw_lines: List[str]) -> List[str]:
    """Fold instantiations that span several lines into one line each."""
    joined: List[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        if "(" in line and not line.endswith(");"):
            buffer = line
            depth = buffer.count("(") - buffer.count(")")
            i += 1
            while i < len(raw_lines) and depth > 0:
                nxt = raw_lines[i].strip()
                buffer += " " + nxt
                depth += nxt.count("(") - nxt.count(")")
                i += 1
            joined.append(buffer + "\n")
        else:
            joined.append(line + "\n")
            i += 1
    return joined


def _split_module(lines: List[str]) -> Tuple[str, List[str], List[str]]:
    """Separate the module header, the declarations, and the instance body."""
    module_line = ""
    declarations: List[str] = []
    body: List[str] = []

    module_seen = False
    pending_declaration = ""
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped.startswith("//"):
            i += 1
            continue

        # Continuous assignments carry no structural information.
        if stripped.startswith("assign"):
            while not stripped.endswith(";") and i + 1 < len(lines):
                i += 1
                stripped = lines[i].strip()
            i += 1
            continue

        if not module_seen and _MODULE_RE.match(line):
            module_seen = True
            header = stripped
            while not header.endswith(");") and i + 1 < len(lines):
                i += 1
                header += lines[i].strip()
            ports = _MODULE_PORTS_RE.match(header)
            # The module is always renamed to `top`, as the contest requires.
            module_line = "module top" + ports.group(1) + "\n" if ports else ""
            i += 1
            continue

        if _DECLARATION_RE.match(stripped) or pending_declaration:
            pending_declaration = (pending_declaration + " " + stripped
                                   if pending_declaration else stripped)
            if stripped.endswith(";"):
                declarations.append(pending_declaration)
                pending_declaration = ""
            i += 1
            continue

        body.append(line)
        i += 1

    if not module_line:
        # Malformed or unrecognised header: emit a valid stub rather than
        # crashing, so a batch run reports one bad file instead of dying.
        module_line = "module top();\n"

    return module_line, declarations, body


def _build_rename_map(module_line: str,
                      declarations: List[str]) -> Dict[str, str]:
    """Map every net to ``n<k>``, module ports first and in order."""
    ports: List[str] = []
    match = _TOP_PORTS_RE.search(module_line)
    if match:
        ports = [p.strip() for p in match.group(1).split(",") if p.strip()]

    nets: List[str] = []
    for declaration in declarations:
        # Drop bus bounds; a bus is renamed as a whole.
        cleaned = _BIT_SELECT_RE.sub("", declaration)
        for token in re.split(r"[\s,]", cleaned)[1:]:
            name = re.sub(r"^\\([\w/]+)$", r"\1", token.strip(" ,;"))
            if name and name not in nets:
                nets.append(name)

    ordered = list(ports)
    for name in nets:
        if name not in ordered:
            ordered.append(name)

    return {name: "n" + str(i) for i, name in enumerate(ordered)}


def _apply_renames(text: str, rename: Dict[str, str]) -> str:
    for original, replacement in rename.items():
        text = re.sub(r"\b" + re.escape(original) + r"\b", replacement, text)
    return text


def _dff_arguments(argument_text: str) -> str:
    """Reorder flip-flop pins into the contest's canonical .RN .SN .CK .D .Q."""
    argument_text = re.sub(r"\.(\w+)\s+\(", r".\1(", argument_text)
    pins = {m.group(1): m.group(2).strip()
            for part in argument_text.split(",")
            for m in [_PIN_RE.match(part.strip())] if m}
    return ", ".join("." + pin + "(" + pins[pin] + ")"
                     for pin in ("RN", "SN", "CK", "D", "Q") if pin in pins)


def _primitive_arguments(argument_text: str) -> str:
    """Convert named pins to positional form with the output first.

    Library cells name their output ``Y`` (non-inverting) or ``ZN``
    (inverting); everything else is an input.
    """
    argument_text = re.sub(r"\.(\w+)\s+\(", r".\1(", argument_text)
    pins = {m.group(1): m.group(2).strip()
            for part in argument_text.split(",")
            for m in [_PIN_RE.match(part.strip())] if m}
    output = pins.get("Y") or pins.get("ZN")
    inputs = [v for k, v in pins.items() if k not in ("Y", "ZN")]
    return ", ".join([output] + inputs) if output else ", ".join(inputs)


def convert(input_path: str, output_path: str) -> int:
    """Normalise one netlist. Returns the number of gates written."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, errors="replace") as handle:
        raw_lines = handle.readlines()

    raw_lines = [_ESCAPED_ID_RE.sub(_flatten_identifier, line)
                 for line in raw_lines]
    lines = _join_wrapped_lines(raw_lines)
    module_line, declarations, body = _split_module(lines)
    rename = _build_rename_map(module_line, declarations)

    # Declarations, renamed and grouped: inputs, then outputs, then wires.
    inputs, outputs, wires = [], [], []
    for declaration in declarations:
        renamed = _apply_renames(declaration, rename).strip()
        target = (inputs if renamed.startswith("input")
                  else outputs if renamed.startswith("output")
                  else wires if renamed.startswith("wire") else None)
        if target is not None:
            target.append("    " + renamed + "\n")

    gate_count = 0
    new_body: List[str] = []
    for line in body:
        line = _apply_renames(line.strip(), rename)

        if line == "endmodule":
            new_body.append("endmodule\n")
            continue

        match = _GATE_RE.match(line)
        if not match:
            new_body.append("    " + line.lstrip() + "\n")
            continue

        gate_type, arguments = match.groups()
        gate_type = gate_type.lstrip("_")
        if gate_type == "dff":
            arguments = _dff_arguments(arguments)
        elif gate_type in PRIMITIVE_GATES:
            arguments = _primitive_arguments(arguments)
        new_body.append("    " + gate_type + " g" + str(gate_count) +
                        "(" + arguments + ");\n")
        gate_count += 1

    module_line = _TOP_PORTS_RE.sub(
        lambda m: "module top(" + ", ".join(
            rename.get(p.strip(), p.strip()) for p in m.group(1).split(",")) + ");",
        module_line)

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.writelines([module_line] + inputs + outputs + wires + new_body)

    return gate_count


def convert_directory(input_dir: str, output_dir: str, suffix: str,
                      overwrite: bool) -> Tuple[int, int, int]:
    """Convert every ``.v`` under ``input_dir``. Returns (done, skipped, failed)."""
    if not os.path.isdir(input_dir):
        raise SystemExit("not a directory: " + input_dir)

    done = skipped = failed = 0
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.lower().endswith(".v"):
                continue

            stem = os.path.splitext(name)[0]
            # Synthesis writes <design>_flat.v; drop that before re-suffixing.
            if stem.endswith("_flat"):
                stem = stem[: -len("_flat")]

            relative = os.path.relpath(root, input_dir)
            target_dir = os.path.join(output_dir, relative) if relative != "." \
                else output_dir
            output_path = os.path.join(target_dir, stem + suffix + ".v")

            if os.path.exists(output_path) and not overwrite:
                print("[skip] exists: " + output_path)
                skipped += 1
                continue

            try:
                gates = convert(os.path.join(root, name), output_path)
                print("[ok]   " + name + " -> " + os.path.basename(output_path) +
                      " (" + str(gates) + " gates)")
                done += 1
            except Exception as error:                   # noqa: BLE001
                print("[err]  " + name + ": " + str(error))
                failed += 1

    return done, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise synthesised netlists into the contest format.")
    parser.add_argument("--flow", choices=sorted(FLOW_SUFFIX), default="dc",
                        help="synthesis flow the input came from; selects the "
                             "output filename suffix (default: %(default)s)")
    parser.add_argument("--in", dest="input", required=True,
                        help="input .v file, or a directory to walk")
    parser.add_argument("--out", dest="output", required=True,
                        help="output .v file, or the destination directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert files that already exist")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        gates = convert(args.input, args.output)
        print("wrote " + args.output + " (" + str(gates) + " gates)")
        return

    done, skipped, failed = convert_directory(
        args.input, args.output, FLOW_SUFFIX[args.flow], args.overwrite)
    print("")
    print("converted " + str(done) + ", skipped " + str(skipped) +
          ", failed " + str(failed) + " -> " + args.output)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
