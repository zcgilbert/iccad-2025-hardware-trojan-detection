#!/bin/bash
# ---------------------------------------------------------------------------
# batch_dc.sh - Synthesise every RTL design in a directory with Design Compiler.
#
# Requires dc_shell on PATH (Synopsys licence needed; not reproducible without
# one -- see synthesis/README.md).
#
# Override any of these; the defaults assume you run from this directory:
#   DC_RTL_DIR   input RTL            (default: ./rtl)
#   DC_OUT_DIR   output netlists      (default: ./netlists)
#   DC_LIB       contest .db library  (default: ./contest_cells.db)
#
# Already-synthesised designs are skipped, so an interrupted run resumes.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DC_RTL_DIR="${DC_RTL_DIR:-$HERE/rtl}"
export DC_OUT_DIR="${DC_OUT_DIR:-$HERE/netlists}"
export DC_LIB="${DC_LIB:-$HERE/contest_cells.db}"
SCRIPT_TCL="$HERE/run_dc.tcl"

if ! command -v dc_shell >/dev/null 2>&1; then
    echo "error: dc_shell not found on PATH" >&2
    exit 1
fi
if [ ! -d "$DC_RTL_DIR" ]; then
    echo "error: no RTL directory at $DC_RTL_DIR" >&2
    exit 1
fi

mkdir -p "$DC_OUT_DIR"

for rtl_file in "$DC_RTL_DIR"/*.v; do
    [ -e "$rtl_file" ] || { echo "no .v files in $DC_RTL_DIR"; exit 1; }

    filename="$(basename "$rtl_file")"
    name_no_ext="${filename%.v}"
    output_file="$DC_OUT_DIR/${name_no_ext}_flat.v"

    if [ -f "$output_file" ]; then
        echo "[skip] $filename - ${name_no_ext}_flat.v already exists"
        continue
    fi

    echo "=== dc_shell: $filename ==="
    dc_shell -no_gui \
             -x "set filename ${filename}; set name_no_ext ${name_no_ext}" \
             -f "$SCRIPT_TCL"
done

echo "done -> $DC_OUT_DIR"
