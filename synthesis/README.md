# Synthesis pipelines — RTL to contest-format netlists

How the training corpus was manufactured. Four stages:

```
RTL (Trojan modules + host designs)
  │
  │  run_dc.tcl / run_genus.tcl        <- needs a commercial licence
  v
flattened netlist  (<design>_flat.v)
  │
  ├─ extract_trojan_labels.py   ->  <design>_results.txt   (labels, extracted
  │                                                         BEFORE renaming)
  └─ convert_to_contest_format.py -> <design>_netlist.v    (anonymised)
```

## Licence requirement — read this first

`design_compiler/` and `genus/` drive **Synopsys Design Compiler** and
**Cadence Genus**. Both need a commercial EDA licence and will not run without
one. They are included because they are how the dataset was built, not because
they are reproducible on a personal machine.

The two Python stages need nothing beyond the standard library and run
anywhere.

## Contents

| Path | What it is |
|---|---|
| `design_compiler/run_dc.tcl` | Per-design DC synthesis script |
| `design_compiler/batch_dc.sh` | Runs DC over a directory of RTL |
| `design_compiler/contest_cells.lib` / `.db` | Cell library, DC flavour |
| `genus/run_genus.tcl` | Per-design Genus synthesis script |
| `genus/batch_genus.sh` | Runs Genus over a directory of RTL |
| `genus/contest_cells.lib` | Cell library, Genus flavour |
| `extract_trojan_labels.py` | Derives Trojan gate labels from a flat netlist |
| `convert_to_contest_format.py` | Rewrites a netlist into the contest format |

**The two cell libraries are not interchangeable.** They describe the same nine
primitives, but the DC copy carries extra attributes — default pin
capacitances, operating conditions, a constraint lookup template — that Design
Compiler's library reader requires and Genus does not. Each flow ships with the
version it needs.

## Running the synthesis stage

Paths come from the environment, so nothing is tied to one machine:

```bash
cd synthesis/design_compiler
DC_RTL_DIR=/path/to/rtl DC_OUT_DIR=/path/to/netlists ./batch_dc.sh
```

```bash
cd synthesis/genus
GENUS_RTL_DIR=/path/to/rtl GENUS_OUT_DIR=/path/to/netlists ./batch_genus.sh
```

Both skip designs that already have output, so an interrupted run resumes.

### Settings that matter

Two synthesis options are load-bearing and should not be "optimised":

* **Hierarchical instance names are preserved.** After flattening, a gate that
  came from the Trojan module still carries `trojan` in its escaped name. That
  is the only remaining link between RTL intent and gate-level structure, and
  the entire labelling stage depends on it.

* **Sequential constant propagation and flop merging are disabled.** Trojan
  trigger logic is *deliberately* near-constant, so an optimiser will fold it
  away and merge its state elements into the host's. The netlist still
  synthesises; the labels just no longer describe it. Silent corruption of this
  kind is far worse than a build failure.

Effort is `low` in both flows: heavy optimisation restructures the Trojan past
recognition, and this corpus needs breadth, not quality of result.

## Labelling, then anonymising — in that order

```bash
# 1. Extract labels while the hierarchy is still visible in the names
python extract_trojan_labels.py --in build/netlists --out build/labels

# 2. Only then strip the names
python convert_to_contest_format.py --flow dc \
       --in build/netlists --out build/contest_netlists
```

The order is not optional. `convert_to_contest_format.py` renames every gate to
`g0, g1, …` and every net to `n0, n1, …`; run it first and the Trojan is
unrecoverable.

**Gate numbering is the contract between the two scripts.** Both walk gate
instances in the same encounter order, so `g17` denotes the same instance in
the label file and the netlist. If you change one traversal, change the other —
a mismatch corrupts every label without producing any error.

### Two labelling modes

Chosen automatically by filename:

* Filename starts with `trojan` — the design *is* a standalone synthesised
  Trojan module, so **every** gate is labelled. This produces the corpus's pure
  positive examples.
* Otherwise — a host circuit with a Trojan injected; only gates whose instance
  name or connected nets carry the `\trojan` prefix are labelled.

### What `--flow` changes

Only the output filename suffix: `_netlist` for `dc`, `_g_netlist` for `genus`.
That suffix is how the corpus keeps the two synthesisers' outputs apart. The
normalisation itself is identical for both — verified by running each flow's
original transformation over 20,925 lines of real netlist plus targeted escaped
identifier cases, with no difference in output.

## The contest netlist format

What comes out the far end:

```verilog
module top(n0, n1, n2, n3);
    input n0, n1;
    output n2, n3;
    wire n4, n5;
    nand g0(n4, n0, n1);
    not  g1(n5, n4);
    dff  g2(.RN(n0), .CK(n1), .D(n5), .Q(n2));
endmodule
```

* Primitive gates are positional, **output first**.
* Flip-flops keep named pins, in the fixed order `.RN .SN .CK .D .Q`.
* Only the nine contest primitives appear; no hierarchy, no assignments, no
  comments.
