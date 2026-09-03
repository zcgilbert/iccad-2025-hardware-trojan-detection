# ---------------------------------------------------------------------------
# run_genus.tcl - Synthesise one RTL design into a flat gate-level netlist.
#
# Driven by batch_genus.sh, which passes two positional arguments:
#   argv 0   the RTL file to read, relative to GENUS_RTL_DIR
#   argv 1   that filename without its .v extension
#
# Paths come from the environment so the flow is not tied to one machine:
#   GENUS_RTL_DIR   input RTL            (default: ./rtl)
#   GENUS_OUT_DIR   output netlists      (default: ./netlists)
#   GENUS_LIB       contest .lib library (default: ./contest_cells.lib)
#
# This flow exists alongside the Design Compiler one on purpose: two vendors
# map the same RTL to visibly different gate structures, and training on both
# stops the model keying on one tool's idioms.
# ---------------------------------------------------------------------------

proc env_or {name fallback} {
    global env
    if {[info exists env($name)] && $env($name) ne ""} {
        return $env($name)
    }
    return $fallback
}

set filename    [lindex $argv 0]
set name_no_ext [lindex $argv 1]

set RTL_DIR [env_or GENUS_RTL_DIR "./rtl"]
set OUT_DIR [env_or GENUS_OUT_DIR "./netlists"]
set TECHLIB [env_or GENUS_LIB     "./contest_cells.lib"]

file mkdir $OUT_DIR

set rtl_file "$RTL_DIR/$filename"
set OUT_FILE "$OUT_DIR/${name_no_ext}_flat.v"
puts "=== Synthesising $rtl_file ==="

# --- Library and search path -----------------------------------------------
set_db init_hdl_search_path $RTL_DIR
set_db target_library $TECHLIB
set_db link_library   $TECHLIB

read_hdl $rtl_file
elaborate

set DESIGN_LIST [get_db designs]
set TOP_DESIGN [lindex $DESIGN_LIST 0]
puts "Top module: $TOP_DESIGN"
current_design $TOP_DESIGN

regsub {^design:} $TOP_DESIGN "" CLEAN_NAME
puts "Clean top name: $CLEAN_NAME"

# --- Settings that make the result usable as training data ------------------
# Low effort keeps the Trojan structure recognisable (see run_dc.tcl).
set_db syn_opt_effort low

# Merging equivalent registers would silently delete Trojan state elements
# that happen to duplicate host logic, corrupting the labels.
set_db optimize_merge_flops   false
set_db optimize_merge_latches false

# --- Synthesis --------------------------------------------------------------
syn_generic
syn_map
syn_opt
ungroup -all -flatten
report_gate

write_hdl > $OUT_FILE
puts "Wrote: $OUT_FILE"

# Genus leaves a formal-verification scratch directory behind.
if {[file exists fv]} {
    file delete -force fv
}

exit
