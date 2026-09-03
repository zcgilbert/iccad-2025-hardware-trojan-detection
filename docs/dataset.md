# The training corpus

The contest supplies ten reference Trojans as RTL. Ten examples cannot train a
network that generalises to Trojans nobody has published, so most of the
engineering effort in this project went into **manufacturing a corpus**.

This document describes what was built and why. Every number here is computed
from [`../data/dataset_manifest.csv`](../data/dataset_manifest.csv), which has
one row per generated netlist and is itself produced by parsing the files:

```bash
python tools/build_dataset_manifest.py --corpus <generated_data> --out manifest.csv
```

## What was generated

**2,300 labelled netlists, 2,150,640 gates in total.**

| Pipeline | Netlists | Median gates | Largest | Trojan gate share |
|---|---:|---:|---:|---|
| Design Compiler | 1,500 | 833 | 9,602 | 500 files at 100 %, 1,000 files at ~8 % |
| Genus | 500 | 73 | 6,866 | 100 % |
| Equivalent rewriting | 180 | 2,487 | 10,263 | 16.7 % |
| Trojan-free | 120 | 651 | 4,105 | 0 % |

Design sizes span **13 to 10,263 gates** (median 659), which matters: the hidden
test set mixes small and large circuits, and a model trained only on one scale
transfers badly to the other.

### Coverage is deliberately balanced

Every one of the ten contest Trojans appears in **exactly 150 netlists**:

```
trojan0 …  150      trojan5 …  150
trojan1 …  150      trojan6 …  150
trojan2 …  150      trojan7 …  150
trojan3 …  150      trojan8 …  150
trojan4 …  150      trojan9 …  150
```

The remaining 800 files carry no Trojan type in their name: the 120 Trojan-free
designs, and the augmentation outputs which inherit their parent's type.

### Three kinds of example, on purpose

The corpus is not uniform, and the mix is the point:

* **Pure positives** (1,000 files) — a Trojan module synthesised on its own, so
  every gate is Trojan. Teaches what Trojan logic looks like internally.
* **Host plus Trojan** (1,120 files) — a Trojan injected into a real circuit.
  Trojan gates are a **median 8.19 %** of the design (10th percentile 2.4 %,
  90th 66.7 %). This is the realistic case, and the source of the class
  imbalance that focal loss exists to handle.
* **Pure negatives** (120 files) — clean designs, so the model has something to
  say "no" about. Ten distinct base circuits — `aes`, `sha_256`, `alu_128`,
  `imagproc`, `pipeline`, `fsm`, `control`, `branch`, `register`,
  `shift_adder` — each synthesised through both flows.

## How the pipelines work

### 1. RTL → gate-level netlist (`synthesis/`)

Trojan RTL and host designs go through **Synopsys Design Compiler** and
**Cadence Genus**, both mapped to the contest's nine-primitive cell library
(`contest_cells.lib`).

Two settings are non-obvious and load-bearing:

* **Hierarchical instance names are preserved** (`verilogout_hierarchical_instance_names`).
  This is what makes labelling possible at all — see below.
* **Sequential constant propagation and flop merging are disabled.** Trojan
  trigger logic is *deliberately* near-constant, so an optimiser will happily
  fold it away. Leaving these on silently deletes the exact structure the model
  is supposed to learn, and produces netlists whose labels no longer match
  their contents.

Effort is set to `low` in both flows. Heavy optimisation restructures the
Trojan beyond recognition, and the corpus needs many diverse designs rather
than a few fast ones.

**Why two synthesisers.** Design Compiler and Genus map identical RTL to
visibly different gate structures. The hidden test set was synthesised by the
organisers, not by us, so training across two vendors is direct insurance
against the model keying on one tool's idioms. The `genus` netlists have a
median of 73 gates against Design Compiler's 833 for the same source material —
the two tools do not even agree on scale.

### 2. Labelling (`synthesis/extract_trojan_labels.py`)

The label problem: we know which *RTL module* is the Trojan, but training needs
to know which *gates* are, after flattening has erased the hierarchy.

Flattening is exactly what makes this recoverable. A gate that came out of the
Trojan module still carries `trojan` inside its escaped instance name, or drives
a net that does. So labels are extracted **at this moment** — after synthesis,
before the anonymising rename.

Two modes, chosen by filename:

* **Whole-file** — the design *is* a standalone synthesised Trojan, so every
  gate is labelled. This produces the pure positives.
* **Selective** — a host circuit with a Trojan injected; only gates whose name
  or connected nets carry the Trojan prefix are labelled.

### 3. Anonymisation (`synthesis/convert_to_contest_format.py`)

The contest format strips every semantic hint: gates become `g0, g1, …`, nets
become `n0, n1, …`, the module becomes `top`, escaped identifiers are flattened,
and continuous assignments are dropped.

Gate numbering is the contract between the two scripts. The labeller and the
converter walk instances in the **same encounter order**, so `g17` means the
same instance in both outputs. Changing one traversal without the other
silently corrupts every label in the corpus.

### 4. Equivalent rewriting (`src/augment_netlists.py`)

Each gate is rewritten into an equivalent sub-circuit with probability `--rate`:

| Original | Replacement |
|---|---|
| `NAND(a,b)` | `NOT(AND(a,b))` or De Morgan `OR(NOT a, NOT b)` |
| `NOR(a,b)` | `NOT(OR(a,b))` or `AND(NOT a, NOT b)` |
| `XOR(a,b)` | `a'b + ab'`, or the classic four-NAND construction |
| `XNOR(a,b)` | `NOT(XOR)`, or `ab + a'b'` |
| `AND` / `OR` | via `NAND`/`NOR` plus inversion |
| `NOT(a)` | `NAND(a,a)` or `NOR(a,a)` |
| `BUF(a)` | double inversion |

Every rule is a Boolean identity, so the circuit's function is preserved.
Structure changes substantially; a Trojan gate's label is inherited by *all* the
gates that replace it, so no Trojan logic goes unlabelled.

Flip-flops are never rewritten — there is no combinational identity for them,
and touching sequential elements would change the state encoding. `not` and
`buf` instances are also passed through untouched, so inverter chains stay
recognisable in the augmented output.

### Verifying it, and what verification found

"Preserved by construction" is exactly the kind of claim that should not be
taken on trust: a rule with one input on the wrong net still produces a valid
netlist that parses, trains, and silently poisons the corpus.
[`tools/check_augmentation.py`](../tools/check_augmentation.py) checks it at two
levels — every rule against the gate it replaces over its **complete truth
table** (two inputs, four rows: a proof, not a sample), and whole augmented
netlists against their sources by random simulation of every output bit.

Writing that checker turned up a real defect. The original implementation
derived each replacement gate's wiring from its *position* in the rule — take
the next unused input, else the previous gate's output. That happens to be
correct for two- and three-gate rules and is wrong for longer ones. Three rules
were affected:

| Rule | What it actually computed |
|---|---|
| `xor → not, and, not, and, or` | `(¬(a'b) ∧ a') ∨ a'b` |
| `xor → nand, nand, nand, nand` | second NAND wired as `nand(t0, t0)` |
| `xnor → and, not, not, and, or` | constant `1` |

Rules are now explicit wiring templates rather than positional chains, and all
fifteen pass exhaustively:

```
[level 1] rewrite rules vs. the gates they replace, exhaustively
  PASS  xnor rule 1 (and -> not -> not -> and -> or)  [4/4 rows]
  PASS  xor  rule 0 (not -> and -> not -> and -> or)  [4/4 rows]
  PASS  xor  rule 1 (nand -> nand -> nand -> nand)    [4/4 rows]
  ...
rules checked: 15 | netlists checked: 24 | failures: 0
```

**This matters for reproducibility, not for the reported score.** The
augmentation pipeline generated 180 of the 2,300 training netlists, and some
fraction of those — the ones where a XOR or XNOR happened to draw the broken
alternative — were not equivalent to their source. The shipped checkpoint was
trained before the fix. The contest result stands as reported, since it is the
score that submission received, but a model retrained from this repository
would not be starting from byte-identical data.

**What this is for.** It is the direct countermeasure to a network memorising
*"this exact NAND-XOR shape is a Trojan"*. The same logic, presented many
structurally different ways, forces the model onto topological features that
survive re-synthesis.

## What ships in this repository, and what does not

The raw corpus is ~106 MB of Verilog, and building graphs from it produces
another ~1.6 GB of intermediate tensors and spreadsheets. None of that is worth
distributing — the tensors are format- and version-specific, and regenerating
them is one command.

What ships instead:

| | |
|---|---|
| The pipelines that built it | `synthesis/`, `src/augment_netlists.py` |
| Complete provenance for all 2,300 files | `data/dataset_manifest.csv` |
| Representative samples | `data/public_benchmark/`, `data/holdout/` |
| The ten Trojans as RTL | `data/trojan_definitions/` |
| The trained result | `models/trojan_gnn.pt` |

The manifest carries, per netlist: source pipeline, base design, Trojan type,
variant, rewrite rate, label, gate count, Trojan gate count and file size. That
is enough to audit the corpus's composition, reproduce every statistic on this
page, and see exactly what the model was trained on — without a 106 MB
download.

## Sample data in this repository

Two small sets, kept distinct because they are different things:

* **`data/public_benchmark/`** — six designs from the organisers' released
  public benchmark, with their published answers. Larger, real contest
  circuits.
* **`data/holdout/`** — six designs from our own held-out split, with ground
  truth. These are what `predict.py` and `tools/smoke_test.py` exercise.

They are not interchangeable: the public benchmark is the organisers' data with
official answers; the hold-out split is ours.
