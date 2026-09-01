# How DiffAb Works: From Diffusion Output to Evaluation

What the model actually generates, what the sampled `.pdb` files do and do not
contain, and what has to happen to them before the numbers in the paper can be
reproduced.

Companion to [`PDB_FORMAT.md`](PDB_FORMAT.md), which covers the PDB column layout,
Chothia renumbering, the Fv crop, and how CDR ranges are located. This document picks
up after a sample has been written.

Worked example throughout:
`results/7DK2_AB_C.pdb_2026_03_03__09_37_49/` — `codesign_single` mode, 100 samples
per CDR.

## 1. The model generates three things, and none of them are atoms

The diffusion state is a per-residue triple. All three channels are diffused
**independently** — sequence is a separate categorical process, not something derived
from geometry, so any sequence can in principle pair with any set of frames.

| Symbol | Shape | What it is |
|---|---|---|
| `t` | `(L, 3)` | translation — **the CA position, and nothing else** |
| `R` | `(L, 3, 3)` | rotation — the orientation of that residue's local frame |
| `s` | `(L,)` | amino acid type, 20 classes |

`N`, `C`, and `O` are never diffused. Neither is any side-chain atom, and no χ
(side-chain torsion) angle is predicted anywhere in the model. This is the precise
sense in which DiffAb "is not full-atom generation."

⚠️ The paper's "rotation" means the **residue backbone frame orientation**. It has
nothing to do with side-chain rotamers.

## 2. From `(R, t, s)` to backbone atoms

`reconstruct_backbone_partially()`
([`geometry.py:450`](diffab/modules/common/geometry.py)), called at
[`design_for_pdb.py:206`](diffab/tools/runner/design_for_pdb.py), converts the
diffusion state into coordinates:

```python
R_new = so3vec_to_rotation(traj_batch[0][0]),   # rotation
t_new = traj_batch[0][1],                        # CA translation
aa    = aa_new,                                  # sequence
```

Inside ([`geometry.py:418-431`](diffab/modules/common/geometry.py)):

```python
bb_coords = backbone_atom_coordinates_tensor[aa]   # local N/CA/C, CA at the origin
bb_pos    = local_to_global(R, t, bb_coords)       # -> global N, CA, C
psi = bb_dihedral[..., 2]                          # O placed from the psi dihedral
```

### Intra-residue geometry is hardcoded

The local template is a constant, near-identical across all 20 residue types
([`constants.py:183-197`](diffab/utils/protein/constants.py)):

```python
AA.ALA: [(-0.525, 1.363, 0.0),   # N
         ( 0.0,   0.0,   0.0),   # CA   <- always the origin
         ( 1.526, -0.0, -0.0)]   # C
```

So N–CA = 1.46 Å, CA–C = 1.53 Å, N–CA–C ≈ 111°, always. **The model cannot deform a
residue** — it places and rotates a rigid three-atom unit. `O` is the one
semi-free atom, positioned from the ψ dihedral computed between neighboring frames.

### Only four atoms are marked valid

```python
pos_recons = F.pad(pos_recons, pad=(0, 0, 0, A-4), value=0)   # slots 5-15 zeroed
mask_bb_atoms[:, :, :4] = True                                # only N, CA, C, O
```

`save_pdb()` skips masked slots, so no side-chain lines are ever emitted for the
designed region.

## 3. What a sampled `.pdb` actually contains

**The generated CDR is backbone-only.** The boundary is exact — in
`H_CDR3/0000.pdb`, residue 94 is the last framework residue and 95 is the first
designed one:

```
94 ARG   N CA C O CB CG CD NE CZ NH1 NH2     <- 11 atoms, full side chain (copied)
95 TYR   N CA C O                            <- 4 atoms. TYR needs 12.
```

Atom accounting for CDR-H3:

| | heavy atoms |
|---|---|
| reference, residues 95–102 | 92 |
| generated, residues 95–102 | 44 (= 11 res × 4) |
| deficit | **48** |

That 48 is also the entire whole-file difference (3251 → 3203 atoms), so CDR side
chains are the *only* thing missing anywhere in the structure.

The file looks full-atom because ~406 framework and antigen residues are copied
verbatim with side chains intact, and only 11 residues are stripped.

⚠️ `H_CDR1` is the most misleading directory to inspect first: its designed residues
are Gly/Ser/Thr/Phe, and `GLY` legitimately has exactly 4 heavy atoms, so a
backbone-only Gly looks completely normal.

### Where did the rotation go?

Nowhere — it is the backbone coordinates. `R` is recoverable exactly from the written
N, CA, C positions by Gram–Schmidt, so storing it separately would be redundant, and
PDB has no field for a residue frame. Nothing was lost on the way out.

## 4. Inter-residue geometry is *not* constrained

`R` and `t` are predicted per residue with **no term tying residue *i*'s C to residue
*i+1*'s N**. Peptide-bond closure is purely learned, and it does not come out clean.

Consecutive CA–CA distances, chain A 93→104 (ideal = 3.80 Å):

```
reference       93-94:3.77  94-95:3.79  95-96:3.78  96-97:3.77  97-98:3.77  98-99:3.80 ...
H_CDR3/0000     93-94:3.77  94-95:3.31  95-96:3.72  96-97:3.82  97-98:3.62  98-99:3.76 ...
                            ^^^^^^^^^^  0.5 A short at the anchor junction
H_CDR3/0001     ...  101-102:4.01
H_CDR3/0002     ...  95-96:3.40
```

Measured over all 100 CDR-H3 samples, 1000 intra-CDR CA–CA bonds:

```
min 2.91 A    max 4.71 A    32.0% fall outside 3.6-4.0 A
```

The crystal structure holds 3.77–3.80 Å throughout. Generated loops routinely land at
3.3 Å or 4.2 Å — physically impossible peptide bonds. The worst offender is
consistently the **anchor junction** (`94-95`), where the fixed framework meets the
generated segment and the model must stitch onto a frame it did not choose.

*(The reference's `101-102` at 2.95 Å is not an error — Tyr101–Pro102 is a genuine
cis-peptide bond, which gives ~2.9 Å.)*

**Consequence: relaxation is not cosmetic polish.** It is what converts a
topologically-correct-but-physically-broken backbone into something Rosetta will
score. There is no mechanism inside the model that enforces chain closure.

## 5. Relaxation: two stages, two different jobs

Run separately from sampling; the raw output is **never** overwritten. Outputs are
suffixed by `get_in_path_with_tag()`
([`relax/base.py:19-22`](diffab/tools/relax/base.py)).

```bash
python -m diffab.tools.relax.run --root ./results
```

### Stage 1 — OpenMM: make it a molecule

[`openmm_relaxer.py:36-73`](diffab/tools/relax/openmm_relaxer.py)

```python
fixer.findMissingAtoms()
fixer.addMissingAtoms(seed=0)      # <-- SIDE CHAINS ARE BUILT HERE
fixer.addMissingHydrogens()
```

then amber99sb minimization with a harmonic restraint pinning every **non-generated**
heavy atom to its input position:

```python
force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
force.addGlobalParameter("k", self.stiffness)      # 10.0 kcal/mol/A^2
```

Two jobs: *construct* the missing atoms (side chains at ideal internal geometry, plus
the hydrogens X-ray never sees), and *repair the bond geometry* — this is what fixes
those 2.91 Å and 4.71 Å CA–CA distances. Restraining the framework while leaving the
CDR free means strain is absorbed into the designed loop instead of propagating into
the scaffold.

Output: `0000_openmm.pdb`

### Stage 2 — PyRosetta FastRelax: make it plausible

[`pyrosetta_relaxer.py:74-145`](diffab/tools/relax/pyrosetta_relaxer.py),
`ref2015`, `max_iter=1000`, `-relax:default_repeats 2`

| Setting | Effect |
|---|---|
| `RestrictToRepacking()` | **no design** — the generated sequence is frozen, only rotamers repack |
| `subset='nbrs'` | CDR (`gen_selector`) + `NeighborhoodResidueSelector` shell; all else `PreventRepacking` |
| `add_bb_action(mm_enable, gen_selector)` | **backbone moves — CDR only** |
| `add_chi_action(mm_enable, subset_selector)` | χ angles move in CDR *and* neighbor shell |

pdbfixer's side chains are at *ideal* torsions with no awareness of packing, so they
clash. FastRelax repacks them from a rotamer library against `ref2015` and
co-minimizes. The neighbor shell matters because a redesigned loop displaces framework
and antigen side chains it contacts. `RestrictToRepacking()` guarantees the diffusion
model's sequence survives intact — Rosetta chooses conformations, never identities.

Output: `0000_rosetta.pdb`

### So: does relaxation move the backbone?

**Yes — both stages do, but motion is confined to the CDR.** Framework and antigen are
restrained (stage 1) or prevented from repacking (stage 2).

### Pipeline options

[`relax/run.py:56-60`](diffab/tools/relax/run.py)

| `--pipeline` | Stages | Backbone | Output suffix |
|---|---|---|---|
| `openmm_pyrosetta` *(default)* | both | moves in CDR | `_rosetta` |
| `pyrosetta` | Rosetta only | moves in CDR | `_rosetta` |
| `pyrosetta_fixbb` | Rosetta, `move_bb=False` | **frozen** | `_fixbb` |

`pyrosetta` alone still produces full-atom output — Rosetta builds missing side chains
on `pose_from_pdb` — but bond-geometry repair is weaker without the OpenMM
minimization. Use `pyrosetta_fixbb` to preserve the diffusion backbone exactly, and
expect worse `ddG`, since the strained bonds are never fixed. For reproducing the
paper, use the default.

Tags do **not** stack: `get_in_path_with_tag()` derives from `in_path`, not
`current_path`, so stage 2 reads `0000_openmm.pdb` and writes `0000_rosetta.pdb`, not
`0000_openmm_rosetta.pdb`.

## 6. Evaluation

```bash
cd diffab && python -m tools.eval.run --root ../results     # note the cd, see gotchas
```

### Which file gets evaluated

`0000_rosetta.pdb`. `--pfx` defaults to `rosetta`, and the scanner builds its filename
pattern from it ([`eval/base.py:88-92`](diffab/tools/eval/base.py)):

```python
input_fname_pattern = f'^\d+\_{self.postfix}\.pdb$'    # ^\d+_rosetta\.pdb$
ref_fname           = f'REF1_{self.postfix}.pdb'       # REF1_rosetta.pdb
```

Raw `0000.pdb` files **do not match**. Eval walks the tree, finds nothing, and reports
zero tasks — no error. **Relaxation is effectively mandatory, not optional.**

The relax scanner's pattern is `(^\d+\.pdb$|^REF\d\.pdb$)`
([`relax/base.py:84`](diffab/tools/relax/base.py)), so **`REF1.pdb` is relaxed too**.
That matters: the reference receives identical OpenMM+Rosetta treatment, so `ddG`
compares like with like rather than crystal-vs-relaxed.

### Metrics

| Source | Metric | Meaning |
|---|---|---|
| [`similarity.py:121-124`](diffab/tools/eval/similarity.py) | `rmsd` | CA-only RMSD over the CDR, **no superposition** — computed in the antigen's frame via a DP alignment that tolerates length mismatch |
| | `seqid` | BLOSUM62 Needleman–Wunsch identity — the **AAR** in the paper |
| [`energy.py:36-42`](diffab/tools/eval/energy.py) | `dG_gen`, `dG_ref`, `ddG` | Rosetta `InterfaceAnalyzerMover` `dG_separated`, antibody-vs-antigen. **`ddG < 0` = predicted better binder than wild type** |

Results accumulate in a `shelve` DB at `results/evaluation_db`, exported to CSV by
`dump_db()`.

## 7. End-to-end

```
   diffusion sampling
   R, t, s  ──►  reconstruct_backbone_partially()
        │
        ▼
   0000.pdb  /  REF1.pdb
        │     CDR: 4 atoms/res (N, CA, C, O). No side chains, no hydrogens.
        │     framework + antigen: full-atom copies of the reference
        │     ⚠ CA-CA bonds range 2.91-4.71 A
        ▼
   ┌─ STAGE 1: OpenMM ───────────────────────────────────────┐
   │ pdbfixer.addMissingAtoms(seed=0)   <- SIDE CHAINS BUILT │
   │ pdbfixer.addMissingHydrogens()                          │
   │ amber99sb minimization, k=10 kcal/mol/A^2 restraint on  │
   │ every non-generated heavy atom                          │
   └─────────────────────────────────────────────────────────┘
        ▼
   0000_openmm.pdb     full-atom, chemically valid, ideal rotamers
        ▼
   ┌─ STAGE 2: PyRosetta FastRelax ──────────────────────────┐
   │ ref2015, max_iter=1000, default_repeats=2               │
   │ RestrictToRepacking()   <- sequence FROZEN, no design   │
   │ movemap: bb  enabled on CDR only                        │
   │          chi enabled on CDR + neighbor shell            │
   └─────────────────────────────────────────────────────────┘
        ▼
   0000_rosetta.pdb  /  REF1_rosetta.pdb    <- THE EVALUATED FILES
        ▼
   ┌─ EVAL (--pfx rosetta) ──────────────────────────────────┐
   │ eval_similarity       -> rmsd, seqid                    │
   │ eval_interface_energy -> dG_gen, dG_ref, ddG            │
   └─────────────────────────────────────────────────────────┘
        ▼
   results/evaluation_db (shelve)  ->  CSV
```

Both stages are resumable. `update_if_finished()`
([`relax/base.py:38-45`](diffab/tools/relax/base.py)) skips any sample whose output
already exists, and eval seeds its `visited` set from the shelve DB.

## 8. Gotchas

**Broken imports in eval.** [`eval/run.py:9-11`](diffab/tools/eval/run.py) and
[`energy.py:14`](diffab/tools/eval/energy.py) use `from tools.eval.base import ...`
while [`similarity.py:7`](diffab/tools/eval/similarity.py) uses
`from diffab.tools.eval.base import ...`. The bare-`tools` form only resolves when run
from inside the `diffab/` package directory. Either `cd diffab` first, or patch the
four lines to `diffab.tools.eval.*`.

**`--pfx` must match the pipeline.** Relaxing with `pyrosetta_fixbb` produces
`_fixbb.pdb`, so eval needs `--pfx fixbb`. A mismatch yields zero tasks and no error
message.

**Eval on unrelaxed files silently does nothing.** Same failure mode: no matching
files, no warning.

**CDR length is fixed input, not a design variable.** The model generates exactly as
many residues as were masked, and the mask comes from the input scaffold. The 100
samples in `H_CDR3/` are 100 sequence/structure variants at a *single* length. See
[`PDB_FORMAT.md`](PDB_FORMAT.md) § "A CDR can be shorter than its range" for how gaps
in the input silently shorten that length.

**Insertion codes are load-bearing.** The writer preserves `resSeq` + `iCode`, which is
what keeps a design alignable to the reference position-by-position. Renumbering the
output collapses `100A` onto `100`.
