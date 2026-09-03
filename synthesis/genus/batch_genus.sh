#!/bin/bash
# ---------------------------------------------------------------------------
# batch_genus.sh - Synthesise every RTL design in a directory with Cadence Genus.
#
# Requires genus on PATH (Cadence licence needed; not reproducible without one
# -- see synthesis/README.md).
#
# Override any of these; the defaults assume you run from this directory:
#   GENUS_RTL_DIR   input RTL             (default: ./rtl)
#   GENUS_OUT_DIR   output netlists       (default: ./netlists)
#   GENUS_LIB       contest .lib library  (default: ./contest_cells.lib)
#
# Already-synthesised designs are skipped, so an interrupted run resumes.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GENUS_RTL_DIR="${GENUS_RTL_DIR:-$HERE/rtl}"
export GENUS_OUT_DIR="${GENUS_OUT_DIR:-$HERE/netlists}"
export GENUS_LIB="${GENUS_LIB:-$HERE/contest_cells.lib}"
SCRIPT_TCL="$HERE/run_genus.tcl"

if ! command -v genus >/dev/null 2>&1; then
    echo "error: genus not found on PATH" >&2
    exit 1
fi
if [ ! -d "$GENUS_RTL_DIR" ]; then
    echo "error: no RTL directory at $GENUS_RTL_DIR" >&2
    exit 1
fi

mkdir -p "$GENUS_OUT_DIR"

for rtl_file in "$GENUS_RTL_DIR"/*.v; do
    [ -e "$rtl_file" ] || { echo "no .v files in $GENUS_RTL_DIR"; exit 1; }

    filename="$(basename "$rtl_file")"
    name_no_ext="${filename%.v}"
    output_file="$GENUS_OUT_DIR/${name_no_ext}_flat.v"

    if [ -f "$output_file" ]; then
        echo "[skip] $filename - ${name_no_ext}_flat.v already exists"
        continue
    fi

    echo "=== genus: $filename ==="
    genus -no_gui -overwrite -batch \
          -execute "set argv {${filename} ${name_no_ext}}" \
          -files "$SCRIPT_TCL"
done

echo "done -> $GENUS_OUT_DIR"
