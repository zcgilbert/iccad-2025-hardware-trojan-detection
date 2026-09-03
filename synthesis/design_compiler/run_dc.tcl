# ---------------------------------------------------------------------------
# run_dc.tcl - Synthesise one RTL design into a flat gate-level netlist.
#
# Driven by batch_dc.sh, which supplies these two variables:
#   filename      the RTL file to read, relative to RTL_DIR
#   name_no_ext   that filename without its .v extension
#
# Paths come from the environment so the flow is not tied to one machine:
#   DC_RTL_DIR    input RTL           (default: ./rtl)
#   DC_OUT_DIR    output netlists     (default: ./netlists)
#   DC_LIB        contest .db library (default: ./contest_cells.db)
# ---------------------------------------------------------------------------

proc env_or {name fallback} {
    global env
    if {[info exists env($name)] && $env($name) ne ""} {
        return $env($name)
    }
    return $fallback
}

set RTL_DIR [env_or DC_RTL_DIR "./rtl"]
set OUT_DIR [env_or DC_OUT_DIR "./netlists"]
set TECHLIB [env_or DC_LIB     "./contest_cells.db"]

file mkdir $OUT_DIR

set rtl_file "$RTL_DIR/$filename"
set OUT_FILE "$OUT_DIR/${name_no_ext}_flat.v"
puts "=== Synthesising $rtl_file ==="

# --- Library and search path -----------------------------------------------
set search_path   [list . $RTL_DIR]
set target_library [list $TECHLIB]
set link_library   [list "*" $TECHLIB]

read_verilog $rtl_file
current_design
set TOP_DESIGN [current_design]
current_design $TOP_DESIGN

# current_design reports as "design:<name>"; keep just the name.
regsub {^design:} $TOP_DESIGN "" CLEAN_NAME
puts "Top module: $CLEAN_NAME"

# --- Settings that make the result usable as training data ------------------
# Keep hierarchy inside instance names: this is what lets
# extract_trojan_labels.py tell Trojan gates from host gates after flattening.
set verilogout_hierarchical_instance_names true

# Do NOT let sequential optimisation fold constants into the flip-flops.
# Trojan trigger logic is deliberately near-constant, and constant propagation
# would delete the very structure we are trying to teach the model to find.
set compile_seqmap_propagate_constants   false
set compile_seqmap_propagate_high_effort false

# --- Synthesis --------------------------------------------------------------
uniquify
# Low effort on purpose: heavy optimisation restructures the Trojan beyond
# recognition, and this corpus needs many diverse designs, not fast ones.
compile -map_effort low
ungroup -all -flatten

report_area
report_timing
report_cell

write -format verilog -output $OUT_FILE
puts "Wrote: $OUT_FILE"

exit
