# Evaluation metrics

DiffAb's original evaluation reports similarity to the reference (`rmsd`, `seqid`)
and interface energy (`dG`, `ddG`). Neither says whether a generated CDR is a
*chemically legal protein on its own terms*. This document covers the additional
reference-free metrics and how to run them.

It also documents `dG_int` / `dG_bind`, which replace `dG_gen`'s conflation of
binding with packing strain — see
[Column reference: binding](#column-reference-binding), and
[`INTERFACE_ENERGY_ANALYSIS.md`](INTERFACE_ENERGY_ANALYSIS.md) for the full
diagnosis and the corrected baseline-vs-fine-tuned comparison.

## Contents

- [Output files](#output-files)
- [Running the evaluations](#running-the-evaluations)
- [Selecting what to evaluate](#selecting-what-to-evaluate)
- [Reading the column names](#reading-the-column-names)
- [What is measured, and over which residues](#what-is-measured-and-over-which-residues)
- [Column reference: validity](#column-reference-validity)
- [Column reference: terms](#column-reference-terms)
- [Column reference: binding](#column-reference-binding)
- [Column reference: ipSAE](#column-reference-ipsae)
- [Interface contacts](#interface-contacts)
- [Thresholds and constants](#thresholds-and-constants)
- [Caveats](#caveats)
- [Why these metrics exist](#why-these-metrics-exist)

## Output files

Every artefact is written into an `evaluation/` subdirectory, never beside the
generated structures.

| File | Contents | Produced by |
| --- | --- | --- |
| `summary.csv` | `rmsd`, `seqid` (= AAR), `dG_gen`, `dG_ref`, `ddG` | `diffab.tools.eval.run` |
| `validity_per_design.csv` | 67 columns; one row per design | `diffab.tools.eval.validate` |
| `validity_summary.csv` | 31 columns; one row per `(method, structure, cdr)` | `diffab.tools.eval.validate` |
| `terms_per_design.csv` | `ref2015` term breakdown over the CDR, one row per design | `diffab.tools.eval.rosetta_terms` |
| `terms_summary.csv` | the same, averaged per `(method, structure, cdr)` | `diffab.tools.eval.rosetta_terms` |
| `binding_per_design.csv` | `dG_int`, `dG_bind`, `fa_rep_bound` + `ref_`/delta; one row per design | `diffab.tools.eval.binding` |
| `binding_summary.csv` | the same, averaged per `(method, structure, cdr)` | `diffab.tools.eval.binding` |
| `peptide_bond_distribution.png` | bond-length histogram, one panel per CDR | `diffab.tools.eval.plot_geometry` |
| `dG_distribution.png`, `dG_violin.png` | interface energy per CDR (`--metric` picks which) | `diffab.tools.eval.plot_energy` |
| `interface_reference.csv` | per-residue contact distances for the reference structures | `diffab.tools.eval.interface` |
| `ipsae_per_design.csv` | `ipsae_max`, `pdockq`, `iptm`, `sc_rmsd` + `ref_`/`neg_`; one row per design | `diffab.tools.eval.ipsae` |
| `ipsae_summary.csv` | the same, averaged per `(method, structure, cdr)` | `diffab.tools.eval.ipsae` |
| `esmfold2/` | cached ESMFold2 predictions (`.pdb`, `.json`, `.cif`) and ipsae's per-residue reports | `diffab.tools.eval.ipsae` |

The files are not joined. They all key on `(method, structure, cdr)` plus
`filename`, so they merge on those columns when needed.

See [Selecting what to evaluate](#selecting-what-to-evaluate) for `--layout`.

By default every artefact goes into the run's own `evaluation/` directory and
nothing is written at the root:

```
results/
  codesign_single/
    7DK2_AB_C.pdb_2026_03_03__09_37_49/
      metadata.json
      reference.pdb
      log.txt
      H_CDR1/ ... L_CDR3/                 <- generated structures
      evaluation/                         <- this run only
        summary.csv
        validity_per_design.csv
        validity_summary.csv
        terms_per_design.csv
        terms_summary.csv
        peptide_bond_distribution.png
        evaluation_db                     <- pipeline cache, not a result
    5XYZ_HL_A.pdb_2026_04_01__10_12_00/
      evaluation/
        ...
```

`--layout flat` or `both` additionally writes a combined roll-up across every
structure into `<--out or --root>/evaluation/`.

The directory name is `EVAL_SUBDIR` in `diffab/tools/eval/base.py`; every runner
resolves its output path through `evaluation_dir()`, so changing it there moves
all of them.

### `evaluation_db.{dat,dir,bak}`

Not results — this is the `run` pipeline's cache. `run.py` opens one `shelve`
database (a persistent dict) **per run**, keyed by output-PDB path and holding
the pickled `EvalTask` for every design already scored, so a re-run skips them.
Keeping it inside the run it describes means deleting a run takes its cache with
it, instead of leaving orphaned entries in a tree-wide database.

The filenames depend on which `dbm` backend Python selects. Python 3.13 defaults
to `dbm.sqlite3`, giving a single file named `evaluation_db`; older interpreters
falling back to `dbm.dumb` produce three — `.dir` (key index), `.dat` (the
pickled values) and `.bak` (a backup of the index). Either way the path prefix is
the same and `shelve` picks the right one automatically.

Only `run` uses it; `validate`, `rosetta_terms` and `plot_geometry` rescan the
tree every time and ignore it entirely.

It is safe to delete at any time — everything it holds is already in
`summary.csv`. Delete it whenever you want `run` to re-score existing designs,
for instance after adding a metric:

```bash
rm <run>/evaluation/evaluation_db*
```

Two things that make a stale database worse than useless: its keys are the paths
as they were when written, so running from a different working directory
invalidates every one of them, and its values are pickles that name the module
they came from — a stale database written before an import path changed raises
`ModuleNotFoundError` when `dump_db` tries to read it. Delete rather than
migrate.

## Running the evaluations

### 0. Prerequisites

Relaxation must have run first: everything below reads `NNNN_<pfx>.pdb` and
compares against the raw `NNNN.pdb`.

```bash
# Rosetta FastRelax only (produces *_rosetta.pdb)
python -m diffab.tools.relax --root ./results --pipeline pyrosetta

# OpenMM minimisation then Rosetta relax (produces *_openmm.pdb then *_rosetta.pdb)
python -m diffab.tools.relax --root ./results --pipeline openmm_pyrosetta
```

Dependencies by stage:

| Stage | Needs |
| --- | --- |
| `validate`, `plot_geometry` | numpy, biopython, matplotlib |
| `rosetta_terms` | PyRosetta |
| `run` (full pipeline) | ray, biopython; PyRosetta unless `--no_energy` |
| `relax` | PyRosetta; OpenMM for the `openmm_*` pipelines |

### 1. Validity metrics (no PyRosetta)

```bash
python -m diffab.tools.eval.validate --root ./results --pfx rosetta
```

Writes `validity_per_design.csv` and `validity_summary.csv`, and prints the
summary. Rescans the result tree on every invocation and does **not** use
`evaluation_db`, so it can be re-run after adding a metric without clearing
anything.

### 2. Bond-length distribution plot

```bash
python -m diffab.tools.eval.plot_geometry --root ./results --pfx rosetta
```

Writes `peptide_bond_distribution.png` and prints per-CDR bond statistics plus a
sweep of the violation rate against the tolerance.

### 3. Rosetta term breakdown (needs PyRosetta)

```bash
python -m diffab.tools.eval.rosetta_terms --root ./results --pfx rosetta
```

### 4. Interface contacts (no PyRosetta)

```bash
python -m diffab.tools.eval.interface --root ./results --pfx rosetta
```

Analyses the **reference** structures and writes `interface_reference.csv`.
Accepts the same `--method` / `--structure` / `--cdr` filters and `--layout` as
the other runners.

### 5. Binding energy without the strain artefact (needs PyRosetta)

```bash
python -m diffab.tools.eval.binding --root ./results --pfx rosetta --workers 12
```

Writes `binding_per_design.csv` and `binding_summary.csv`. See
[Column reference: binding](#column-reference-binding) for why these exist
alongside `dG_gen` — in short, `dG_gen` is dominated by packing strain rather
than by binding.

Costs ~4.0 s per design, so ~9 min per 600 designs on 12 cores. `--workers 1`
runs serially without Ray. Accepts the usual `--method` / `--structure` / `--cdr`
filters and `--layout`.

### 6. Energy distribution plots

```bash
# one figure per run, from summary.csv
python -m diffab.tools.eval.plot_energy --root ./results

# overlay two runs, violins on one shared axis, using dG_bind
python -m diffab.tools.eval.plot_energy \
  --runs <runA> <runB> --labels baseline fine-tuned --out <runB> \
  --kind violin --metric dG_bind --clip 100
```

| Flag | Effect |
| --- | --- |
| `--metric` | `dG_separated` (default, reads `summary.csv`), `dG_bind`, `dG_int` |
| `--kind` | `hist` (default, one panel per CDR) or `violin` (all CDRs, one shared axis) |
| `--runs` / `--labels` | overlay several runs in a single figure |
| `--clip` | hide the design tail above this percentile when drawing (default 99; 100 = no clipping) |
| `--ylim LO HI` | explicit energy-axis limits for `--kind violin` |

Output filenames derive from the metric (`dG_*`, `dG_bind_*`, `dG_int_*`), so
metrics never overwrite each other. Runs are never pooled — each design
distribution keeps its own reference, colour-matched.

### 7. ipSAE self-consistency (needs a GPU)

```bash
# once per machine: downloads weights and confirms the ESMFold2 result fields
python -m diffab.tools.eval.ipsae --probe

python -m diffab.tools.eval.ipsae --root ./results --top-n 5 --workers 2
```

Refolds the designed **sequences** with ESMFold2 and scores the antibody-antigen
interface of the prediction with ipSAE. This is the only metric here that does
not read DiffAb's coordinates; see
[Column reference: ipSAE](#column-reference-ipsae) for what that buys.

Needs `binding_per_design.csv` from section 5 first: designs are ranked by
`ddG_int` and only the best `--top-n` per CDR are folded. A run without that
file is skipped with a warning rather than folding an arbitrary subset. Folding
is minutes per complex, so scoring all 600 designs of a run is not practical --
600 designs x 3 arms would be days.

| Flag | Effect |
| --- | --- |
| `--probe` | fold a toy dimer, print the result object's fields, and exit |
| `--top-n <n>` | designs per CDR to fold, best first (default 5) |
| `--rank-by <col>` | column of `binding_per_design.csv` to rank by (default `ddG_int`, ascending) |
| `--arms <list>` | subset of `designs,native,negative` |
| `--negative {shuffle,mismatch}` | shuffle the CDR in place, or swap in another target's antigen |
| `--pae-cutoff`, `--dist-cutoff` | ipSAE cutoffs in A (default 10, 10); recorded in the CSV |
| `--checkpoint <name>` | default `biohub/ESMFold2-Fast`; `biohub/ESMFold2` is the MSA-capable model |
| `--workers <n>` | one GPU per worker (2 on this workstation); 1 runs serially without Ray |

Predictions are cached in `<run>/evaluation/esmfold2/` keyed by a hash of the
sequences folded, so an interrupted run resumes without refolding. Accepts the
usual `--method` / `--structure` / `--cdr` filters and `--layout`.

### 8. Full pipeline

```bash
python -m diffab.tools.eval.run --root ./results --pfx rosetta
```

Runs similarity and interface energy only — `summary.csv` keeps the original
DiffAb columns (`rmsd`, `seqid`, `dG_gen`, `dG_ref`, `ddG`). The validity and
term metrics are deliberately kept out of it; they have their own files. Flags:

| Flag | Effect |
| --- | --- |
| `--pfx <tag>` | which relaxed structures to read (default `rosetta`; `''` for raw) |
| `--no_energy` | skip interface energy — PyRosetta is then never imported |
| `--once` | stop after one pass instead of watching for new results |

Writes each run's `summary.csv` into that run's `evaluation/`. Keys off that
run's `evaluation_db` and skips anything already recorded there — to re-evaluate
existing designs, `rm <run>/evaluation/evaluation_db*` first. No filter or
`--layout` flags; scope it with `--root`. Add `--once` to stop after a single
pass instead of watching for new results.

### Installing PyRosetta

```bash
python -m pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

Two failure modes worth knowing. On macOS builds from python.org, the download
fails with `CERTIFICATE_VERIFY_FAILED` — run `/Applications/Python
3.x/Install Certificates.command`, or `export
SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")`. And the
installer shells out to bare `pip`; if only `python -m pip` is on PATH it prints
the resolved wheel URL, exits 0, and installs nothing. In that case install the
printed URL directly with `python -m pip install <url>`.

## Selecting what to evaluate

By default every runner walks the whole `--root` tree and evaluates everything it
finds. There are three ways to narrow that.

### 1. Point `--root` at a subdirectory

Works at any depth, because each structure's metadata is resolved relative to the
PDB file rather than to the root:

```bash
# one run
python -m diffab.tools.eval.validate --root ./results/codesign_single/7DK2_AB_C.pdb_2026_03_03__09_37_49

# a single CDR of one run
python -m diffab.tools.eval.validate --root ./results/codesign_single/7DK2_AB_C.pdb_2026_03_03__09_37_49/H_CDR3
```

### 2. Filter flags

`validate` and `rosetta_terms` accept comma-separated, case-insensitive
substring filters. They compose (all must match):

| Flag | Matches against | Example |
| --- | --- | --- |
| `--method` | method directory name | `--method codesign_single` |
| `--structure` | `metadata['identifier']` | `--structure 7DK2,5XYZ` |
| `--cdr` | CDR tag | `--cdr H_CDR3,L_CDR1` |

```bash
# only CDR-H3 and L-CDR1, across every structure
python -m diffab.tools.eval.validate --root ./results --cdr H_CDR3,L_CDR1

# one structure, one CDR
python -m diffab.tools.eval.validate --root ./results --structure 7DK2 --cdr H_CDR3
```

The runner reports both counts, e.g. `Found 600 structures, 200 selected.`

### 3. `--layout` — where the output goes

| Value | Behaviour |
| --- | --- |
| `per_run` *(default)* | only inside each run's `evaluation/` — nothing at the root |
| `both` | the per-run files **and** a combined roll-up under `--out` / `--root` |
| `flat` | only the combined roll-up under `--out` / `--root` |

```bash
# keep results with their runs, no root-level files
python -m diffab.tools.eval.validate --root ./results --layout per_run

# one combined table somewhere else entirely
python -m diffab.tools.eval.validate --root ./results --layout flat --out ./analysis
```

Per-run files contain only that run's rows, with their own summary computed over
that subset. Filters apply before writing, so a filtered run produces filtered
files in both layouts.

In every case the files land in an `evaluation/` subdirectory, not directly in
the chosen location.

Note `diffab.tools.eval.run` (the ray pipeline) has **no** filter or `--layout`
flags — it always writes per-run only. Scope it with `--root`.

## Reading the column names

Every column is **prefix + metric + aggregation**.

**Prefix — which structure was measured:**

| Prefix | File read | Meaning |
| --- | --- | --- |
| *(none)* | `NNNN_<pfx>.pdb` | the design, post-relax — what the evaluation scores |
| `pre_` | `NNNN.pdb` | the same design before relaxation — raw model output |
| `ref_` | `REF1_<pfx>.pdb` | the **native** CDR through the identical protocol |

`ref_*` is the control. Same parser, same thresholds, same residue range, same
relax protocol — so any difference is attributable to the designed residues.

**Aggregation suffix — `validity_summary.csv` only**, applied over the designs in
one `(method, structure, cdr)` group: `_mean`, `_max`, `_min`, `_std`.

`ref_*` columns in the summary carry **no** suffix: the reference is one value
repeated across every design in the group, so it is taken as-is. Each `ref_*`
column sits immediately after the generated column it controls.

## What is measured, and over which residues

Every metric covers **only the designed CDR residues**, sliced
`residue_first` → `residue_last` from `metadata.json` by
`similarity.extract_reslist` — the same function `rmsd` and `seqid` use, so the
residue set is identical across all output files.

```
cdr       n_res   n_bonds
H_CDR1      7        6
H_CDR2      6        5
H_CDR3     11       10     (95-102 plus insertion codes 100A/100B/100C)
L_CDR1     11       10
L_CDR2      7        6
L_CDR3      9        8
```

`n_bonds = n_res - 1`: a peptide bond lies *between* consecutive residues. With
5–10 bonds per CDR, one bond is 10–20% of any fraction — the fractions below are
coarse-grained by construction.

## Column reference: validity

`d` denotes one peptide bond length; `N` the number of designs in a group.

### Group keys

| Column | Meaning |
| --- | --- |
| `method` | method directory name, e.g. `codesign_single` |
| `structure` | `metadata['identifier']`, e.g. `7DK2_AB_C.pdb` |
| `cdr` | CDR tag, e.g. `H_CDR3` |
| `filename` | per-design file, e.g. `0096_rosetta.pdb` (per-design file only) |

### Residue counts

| Column | How it is computed |
| --- | --- |
| `n_res` | residues in the CDR slice with all of N, CA, C, O present |
| `n_incomplete_res` | residues in the slice missing a backbone atom (should be 0) |

### Peptide bond — `pep_*`

The peptide bond is the distance from the **carbonyl `C` of residue i** to the
**amide `N` of residue i+1**, for every consecutive pair. Ideal 1.329 Å. This is
an *inter-residue* bond, distinct from the intra-residue N–CA–C angle.

| Column | How it is computed |
| --- | --- |
| `pep_mean`, `pep_min`, `pep_max` | mean / min / max of `d` within one design |
| `pep_rmsd_ideal` | `sqrt(mean((d - 1.329)^2))` over the design's bonds |
| `pep_n_viol` | count of bonds with `abs(d - 1.329) > 0.05` |
| `pep_frac_viol` | `pep_n_viol / n_bonds` |
| `pep_n_break` | count of bonds with `abs(d - 1.329) > 0.30` — effectively severed |

In the summary:

| Column | Meaning |
| --- | --- |
| `pep_rmsd_ideal_mean` | mean over the `N` designs |
| `pep_frac_viol_mean` | mean over the `N` designs |
| `pep_n_break_mean`, `_max` | mean and worst-case count |
| `pep_min_mean` | mean of the per-design **minimum** — "the typical design's worst bond" |
| `pep_min_min` | the single shortest bond anywhere in the group |

`pep_min_mean` and `pep_min_min` are different statements; do not conflate them.

**`pep_rmsd_ideal` is the metric to prefer.** It needs no threshold, does not
saturate near 100%, and is stable: comparing pre- and post-relax, `pep_mean` and
`pep_rmsd_ideal` agree to 0.0005 Å while `pep_frac_viol` can differ by 0.2,
because a bond sitting at the tolerance boundary flips category. Use
`pep_frac_viol` for communication, not for ranking designs.

### Intra-residue bonds and angle

Computed within each residue. DiffAb builds N/C/O from ideal internal geometry
given each residue's predicted frame, so these are near-zero by construction and
carry almost no information — they exist to demonstrate that the *intra*-residue
geometry is fine and the defect is specifically in chain connectivity.

| Column | How it is computed |
| --- | --- |
| `bond_nca_dev_max` | max `abs(d(N,CA) - 1.458)` |
| `bond_cac_dev_max` | max `abs(d(CA,C) - 1.525)` |
| `bond_co_dev_max` | max `abs(d(C,O) - 1.231)` |
| `nca_c_dev_max` | max `abs(angle(N,CA,C) - 111.2)`, degrees |

### Peptide planarity — `omega_*`

`omega` is the dihedral **CA(i) → C(i) → N(i+1) → CA(i+1)**. A planar *trans*
peptide is ±180°, *cis* (legitimate for proline) is 0°. Deviation is
`min(180 - abs(omega), abs(omega))`, so 0 means ideally planar in whichever state
the bond is in and cis-proline is not penalised.

| Column | How it is computed |
| --- | --- |
| `omega_dev_mean` | mean deviation over the design's bonds, degrees |
| `omega_dev_max` | worst deviation in the design |
| `omega_n_viol` | count of bonds deviating more than 30° |

In the summary, `omega_dev_mean_mean` is the two levels of averaging (per design,
then over designs) — not a typo.

Weak signal in practice: native CDRs themselves deviate 2.6–7.7°, so designs at
1.5–2.4× native are barely distinguishable next to a 30–70× ratio on bond
precision. Not worth a headline.

### Clashes — `clash_*`, `pre_clash_*`

Atom-level, not residue-level. For every **backbone heavy atom** (N, CA, C, O) of
the CDR, find all heavy atoms in the entire complex within 2.2 Å
(`Bio.PDB.NeighborSearch`, a KD-tree), then exclude:

- the atom itself,
- any atom in the same residue,
- any atom in a residue adjacent in the same chain (covalently bonded, not a clash),

and de-duplicate pairs. Surviving pairs are clashes. Included are CDR-internal
pairs two or more residues apart, and CDR-to-framework and CDR-to-antigen pairs.

| Column | How it is computed |
| --- | --- |
| `clash_n` | clash count, post-relax |
| `clash_n_severe` | subset closer than 1.5 Å |
| `clash_min_dist` | closest non-bonded contact, Å |
| `pre_clash_n`, `pre_clash_n_severe`, `pre_clash_min_dist` | the same on the raw file |

Two deliberate asymmetries:

- **Heavy atoms only.** Relaxed files carry hydrogens, raw files do not; without
  this the two would not be comparable.
- **Backbone only on the CDR side.** The raw file physically contains no
  side-chain atoms for designed residues. The *partner* side is unrestricted —
  every heavy atom in the complex, framework and antigen side chains included.

This is a comparability constraint, not a claim that side-chain clashes are
unimportant; post-relax side-chain strain is what `term_fa_rep` covers.

Read `pre_clash_*` as the honest measure of what the model produced: FastRelax
genuinely does relieve steric overlap, so post-relax clash counts are near zero
across the board.

### Relaxation shift — `shift_*`

How far relaxation had to move the generated backbone. **No superposition** —
coordinates are subtracted directly between `NNNN.pdb` and `NNNN_<pfx>.pdb`,
which is valid because DiffAb holds the framework and antigen fixed, so both
files already share one coordinate frame.

| Column | How it is computed |
| --- | --- |
| `shift_ca_rmsd` | `sqrt(mean(d^2))` over CA atoms only (`n_res` terms) |
| `shift_bb_rmsd` | the same over all four backbone atoms (`4 * n_res` terms) |
| `shift_bb_max` | the largest single per-atom displacement in the design |

In the summary, `shift_bb_max_max` is the worst per-atom displacement across the
whole group. Side chains are excluded by necessity — they do not exist in the
pre-relax file, so all-atom is impossible here.

The rationale: relaxation moves only what it must to reduce energy, so a large
shift means the generated backbone was far from any physically reasonable
conformation. `ref_shift_*` — the native loop through the same protocol — is the
null this is read against.

### Reference controls

Every `ref_*` column is the corresponding metric computed on `REF1_<pfx>.pdb`
over the same residue range: `ref_pep_rmsd_ideal`, `ref_pep_frac_viol`,
`ref_pep_min`, `ref_omega_dev_mean`, `ref_clash_n`, `ref_shift_ca_rmsd`,
`ref_shift_bb_rmsd`. The per-design file additionally carries the full `ref_*`
geometry block.

`ref_shift_*` is measured `REF1.pdb` → `REF1_<pfx>.pdb`.

## Column reference: terms

`terms_summary.csv` — one row per `(method, structure, cdr)`, every column a
**mean over the designs** in that group. `terms_per_design.csv` has the same
columns unaggregated, one row per design.

There are no aggregation suffixes here: every summary column is a mean, so
`_mean` everywhere would carry no information. The prefix convention matches
validity — `term_` is the design (`NNNN_<pfx>.pdb`), `ref_term_` is the native
CDR (`REF1_<pfx>.pdb`), and each control sits directly after the column it
controls.

### What a "term" is

`ref2015` is not a single formula but a **weighted sum of about nineteen
components**, each one a *score term*:

```
total_score = sum over terms of  weight[term] * value[term]
```

Lower is more favourable. The columns here are parts of one score, not
alternative ways of scoring, and they add up to `total`.

Terms come in two kinds, which is only a statement about how each is computed:

- **one-body** — depends on a single residue's own conformation:
  `fa_dun`, `rama_prepro`, `p_aa_pp`, `omega`, `ref`, `fa_intra_*`
- **two-body** — depends on a pair of residues: `fa_atr`, `fa_rep`, `fa_sol`,
  `fa_elec`, `lk_ball_wtd`, `hbond_*`

`fa` stands for **full-atom**, as opposed to Rosetta's coarse-grained *centroid*
mode where a side chain collapses to one pseudo-atom.

### How each value is computed

In `cdr_term_scores` (`diffab/tools/eval/rosetta_terms.py`):

1. `pose_from_pdb` loads the complex, `scorefxn(pose)` scores it with `ref2015`.
2. `pdb2pose` maps `residue_first`/`residue_last` to pose indices — the same CDR
   slice the validity metrics use.
3. Per term: `sum(residue_total_energies(i)[term] for i in CDR) * weight`.

Three things follow.

**Values are weighted**, so all columns are in comparable Rosetta Energy Units
and sum to `total`. The weights are not all 1:

```
fa_atr 1.0   fa_sol 1.0   fa_elec 1.0   hbond_* 1.0   ref 1.0
fa_rep 0.55  fa_dun 0.7   p_aa_pp 0.6   rama_prepro 0.45   omega 0.4
```

`fa_atr` and `fa_rep` are the attractive and repulsive halves of one
Lennard-Jones potential, split so they can be weighted separately — Rosetta
deliberately softens repulsion.

**These are the CDR's share, not its total interaction energy.** A two-body
energy is computed once for the pair and then attributed half to each partner,
so a CDR-to-antigen contact contributes half here and half to the antigen
residue outside the slice. (Verifiable: summing any term over *all* residues
reproduces the whole-pose value exactly, which only works if the split is even.)

**Sign convention:** negative is favourable, positive is a penalty.

### Columns

| Column | Meaning |
| --- | --- |
| `term_total_no_ref` | `total` minus the `ref` term. **The column to compare designs on** |
| `term_total` | the CDR's share of the pose score, weighted sum over all terms |
| `term_total_per_res` | `total / n_res`, comparable across CDRs of different length |
| `term_rama_prepro` | Ramachandran, pre-proline aware — are phi/psi in a populated region. The most direct "is this backbone legal" signal here |
| `term_fa_rep` | Lennard-Jones repulsion: steric strain. Soft and longer-range, so more sensitive than the hard 2.2 A cutoff behind `clash_n` |
| `term_p_aa_pp` | `P(amino acid | phi, psi)` — is the designed residue type compatible with the backbone it was given. The one term that probes sequence/structure consistency directly |
| `term_n_res` | residues in the slice; `ref_term_n_res` always equals it |

Why `total_no_ref` rather than `total`: `ref` is a per-amino-acid constant
depending only on composition, so a design that chose bulkier residues carries a
different baseline for reasons with no structural content. Subtracting it removes
that offset. In this dataset the difference is worth up to ~6.6 REU.

### Reporting more terms

Every term is computed on the same scoring pass, so the unreported ones cost
nothing. `REPORT_ORDER` at the top of `rosetta_terms.py` controls what is
written — add a name and re-run.

Two are deliberately excluded and should not be added: **`hbond_sr_bb` and
`hbond_lr_bb`**. Rosetta keeps those in separate whole-pose energy containers and
never attributes them to individual residues, so a per-residue query returns
`0.0` for any structure, loop or helix. They previously appeared as columns of
zeros, which read like a finding about the loops and was not one.

### `total` is not `dG`

The most common confusion, since both are `ref2015` in REU:

| | `term_total` | `dG_gen` (in `summary.csv`) |
| --- | --- | --- |
| Kind | one structure's score | a **difference** of two scores |
| Definition | score of the complex, CDR residues only | `score(complex) - score(chains separated and repacked)` |
| Answers | how strained is this loop | how much does binding gain |
| Scope | the CDR slice | the whole antibody-antigen interface |

`total_score` is like an altitude; `dG` is a height difference between two
points. A design can be badly strained (poor `total`) yet still grip the antigen
(acceptable `ddG`), or the reverse — neither is derivable from the other.

One important qualification to the "Definition" row: only the *separated* side is
repacked, so `dG_gen` is not a clean height difference — it also carries the
structure's repackable strain. That makes the independence claim above weaker
than it looks, since `dG` partly measures strain too. See
[Column reference: binding](#column-reference-binding).

### Caveats

- **`total` excludes backbone hydrogen bonding.** It is built from
  `residue_total_energy`, which omits the two undecomposable `hbond_*_bb` terms.
  Design and native are both missing the same contribution, so comparisons hold,
  but the absolute value is not a complete energy. Arithmetically:
  `sum(residue_total_energy) - total_score = hbond_sr_bb + hbond_lr_bb`.
- **`ref2015` has no bond-length or bond-angle term.** Every column here is blind
  to the connectivity defects the validity metrics measure; a 0.38 A peptide bond
  registers nowhere in this file. `omega` penalises a *twisted* peptide, not a
  broken one. The two files measure genuinely different things.
- **These are post-relax numbers.** FastRelax minimised exactly these terms, so
  what remains is *residual* strain that survived local optimisation — a floor,
  not the strain the model produced.
- **`fa_dun` reflects relaxation, not generation** (if you re-enable it). DiffAb
  emits no side chains; their rotamers were chosen by the relax stage.

## Column reference: binding

`binding_per_design.csv` and `binding_summary.csv`, from
`diffab/tools/eval/binding.py`.

### Why these exist: `dG_gen` is not a binding energy

`InterfaceAnalyzerMover` computes `E(bound) - E(unbound)`, and each side can
optionally be repacked first. `energy.py:33` enables `set_pack_separated(True)`
and leaves `pack_input` at its `False` default, so the **unbound** state is
allowed to relieve its side-chain strain and the **bound** state is not. The
reported quantity is therefore

```
dG_gen  =  interaction energy  +  R
```

where `R >= 0` is whatever strain repacking can remove. Because DiffAb relaxes
only a ~10 A shell around one CDR (`subset='nbrs'`), the other ~250 residues keep
raw input side chains whose strain never cancels. Measured on the 7DK2 tree:

| | range across the 6 CDR references |
| --- | --- |
| `dG_gen`-style value | 37.7 … 104.3 (spread **66.6** REU) |
| interaction energy | −33.3 … −44.9 (spread **11.5** REU) |
| `R` | 79.3 … 138.1 (spread **58.8** REU) |

`corr(dG_ref, R) = +0.985` versus `corr(dG_ref, interaction) = +0.488`. So the
column is closer to a *strain meter* than to an affinity. Full derivation, the
2x2 flag table, and the term-by-term cancellation proof are in
[`INTERFACE_ENERGY_ANALYSIS.md`](INTERFACE_ENERGY_ANALYSIS.md).

### Columns

Each appears three times: bare (the design), `ref_`-prefixed (the native
control), and as a delta. Lower is better for all three.

| Column | Meaning |
| --- | --- |
| `dG_int` | `dG_separated` with **packing off on both sides** |
| `ref_dG_int` | the same for the native; **one cached value per CDR** |
| `ddG_int` | `dG_int - ref_dG_int` |
| `dG_bind` | `dG_separated` with **packing on both sides** |
| `ref_dG_bind` | the same for the native, averaged over `REF_REPEATS = 3` |
| `ddG_bind` | `dG_bind - ref_dG_bind` |
| `fa_rep_bound` | weighted `fa_rep` of the bound complex, no repacking |
| `ref_fa_rep_bound` | the same for the native |
| `delta_fa_rep_bound` | `fa_rep_bound - ref_fa_rep_bound` |

### `dG_int` — interaction energy, exactly strain-free

Packing off on both sides means the separated pose is a pure rigid-body
translation of the bound one, so **every one-body and intra-chain term is bit
identical and cancels to exactly zero** — verified: 11 of 19 `ref2015` terms,
including all 972 REU of `fa_dun`. Only the eight genuine inter-chain terms
survive (`fa_atr`, `fa_sol`, `fa_rep`, `fa_elec`, `lk_ball_wtd`, `hbond_bb_sc`,
`hbond_lr_bb`, `hbond_sc`).

Consequences:

- **Deterministic.** No packer runs, so no RNG is consumed: repeated scorings of
  the same file give `sd = 0.00`. Contrast `dG_ref`, which takes 6–16 distinct
  values per 100 draws purely from the packer's seed.
- **Immune to `R`.** The relaxation-quality artefact is gone by construction.
- **Comparable across CDRs**, which `ddG` is not.

It is an **interaction energy, not a free energy.** The unbound state stays
frozen in its bound conformation, so it omits side-chain reorganisation, backbone
relaxation, and conformational entropy on unbinding — all of which make it
overstate affinity. It is also not fully relax-independent: the designed CDR's
own side chains were built by relax and sit on the interface.

### `dG_bind` — closer to a real ΔG

Both states get the same side-chain optimisation, so strain cancels
approximately rather than exactly, but the unbound state is now allowed the
reorganisation that genuinely accompanies unbinding — which is the correct
physics `pack_separated=True` was designed to capture.

Stochastic, but far less so than `dG_gen`: `sd ~0.25` REU on relaxed structures,
so one sample per design suffices. Costs ~2.4 s per structure against 1.7 s for
`dG_int`.

### `fa_rep_bound` — the strain your validity metrics cannot see

`fa` = full-atom, `rep` = repulsive. Rosetta splits the Lennard-Jones 6–12
potential into `fa_atr` (attractive r⁻⁶, ≤ 0, weight 1.00) and `fa_rep`
(repulsive r⁻¹², ≥ 0, weight 0.55). It penalises every atom pair closer than the
sum of their van der Waals radii, and is linearised at very short range rather
than diverging — which is why a badly clashing structure scores 1100 instead of
infinity.

Not a binding term. It is recorded because it is what actually drives the spread
in `dG_gen` (Spearman **0.72** within H_CDR3, against 0.25 for the interaction
energy), and because **`clash_n` is blind to it**: `clash_n` counts only
backbone–backbone pairs under 2.2 A inside the CDR, while `fa_rep` is all-atom
repulsion across all 311 residues — overwhelmingly side chains, which DiffAb
never generated. Designs with `fa_rep_bound` of 489 and 915 both have
`clash_n = 0`.

Baseline is ~517–532 for five CDRs (the shared, never-relaxed bulk of the
structure). H_CDR3 reaches a median of 612 and a max of 1100.

### Caveats

- **The reference is scored once per CDR, not per design.** This is deliberate —
  it is what removes the seed noise — but it means `ref_*` columns are constant
  within a CDR, and the reference spread that `plot_energy` draws for
  `dG_separated` collapses to a single line.
- **Different references per CDR remain.** `binding.py` reads the same
  `REF1_rosetta.pdb` files, so the six CDRs still have six differently relaxed
  natives. `dG_int` removes the *strain* consequence of that (spread falls from
  66.6 to 11.5 REU) but not the underlying difference. A single `subset='all'`
  relax of the native would.
- **`dG_bind` does not rescue a clashed design.** For H_CDR3 its median is
  +32.7 (baseline) and +6.9 (fine-tuned) against −19 to +3 for every other CDR;
  repacking cannot fix strain that severe.
- **`summary.csv` is untouched.** `dG_gen` / `dG_ref` / `ddG` keep their original
  meaning and values; these are additional columns in a separate file.

## Column reference: ipSAE

`ipsae_per_design.csv`, `ipsae_summary.csv`. Produced by
`python -m diffab.tools.eval.ipsae`.

### Why this exists

Every other metric in this document reads the coordinates DiffAb produced.
`rmsd` compares them to the native, `dG_int` scores their interface, `n_contact`
counts their contacts. All of them take the pose as given, so none can
distinguish a design that would actually fold and bind from one that merely
scores well under the generator's own assumptions.

This metric asks the orthogonal question: **hand only the designed sequence to
an independent structure predictor -- does it rebuild the complex, confidently?**
That is the self-consistency test binder-design papers use to filter designs
before synthesis, and it is the one number here that DiffAb cannot influence
except through the sequence it wrote.

ipSAE rather than ipTM because ipTM averages over whole chains. An Fv plus
antigen is mostly non-interface residues, so a genuinely confident epitope
contact is diluted by several hundred residues that were never going to touch
anything. ipSAE restricts the sum to residue pairs under a PAE cutoff and
rescales `d0` to match, which is exactly the correction this geometry needs.

ESMFold2 (CZI Biohub, May 2026) is the backend because it is reported to beat
AlphaFold3 on antibody-antigen binding-pose accuracy -- the hardest complex
class and the only one that matters here -- and because it is MIT-licensed,
pip-installable, and takes multi-chain input directly.

### The three arms

An ipSAE value on its own means nothing, so three complexes are folded per
design and reported side by side under the usual prefix grammar:

| Prefix | Arm | What is folded |
| --- | --- | --- |
| *(none)* | design | the designed H + L + antigen sequences |
| `ref_` | native | the crystal antibody, same antigen -- the ceiling |
| `neg_` | negative | the designed chains with the CDR residues shuffled in place, same antigen -- the floor |

The shuffle preserves length and amino-acid composition and touches only the
residues DiffAb designed, so the negative differs from the design in exactly the
variable under test. `--negative mismatch` instead pairs the antibody with a
different structure's antigen, which is a harder floor.

**Read the controls before reading the designs.** The runner prints the three
means and warns when `ref_` and `neg_` differ by less than 0.1. If they do not
separate, ipSAE is blind on that target and no difference between design methods
can be read from it -- that is a result about the metric, not about the designs.

### Columns

| Column | Meaning |
| --- | --- |
| `ipsae_max` | the headline: best ipSAE between either antibody chain and the antigen |
| `ipsae_H_Ag` | ipSAE between the heavy chain and the antigen |
| `ipsae_L_Ag` | ipSAE between the light chain and the antigen |
| `pdockq` | pDockQ for the same interface, as a cross-check |
| `iptm`, `ptm` | ESMFold2's own whole-complex confidences |
| `sc_rmsd` | CA-RMSD, CDR only, between ESMFold2's prediction and DiffAb's design |
| `pae_cutoff`, `dist_cutoff` | the ipSAE cutoffs these rows were computed at |

Each of the first six appears three times (`x`, `ref_x`, `neg_x`). `sc_rmsd` is
design-only -- there is no native or negative counterpart to compare against.

ipSAE is asymmetric; the value taken is ipsae.py's `max` row, which is the
maximum over both directions and the value the paper reports. Where an antigen
has several chains the best-scoring one wins, since a design only has to bind
somewhere on the target.

### `sc_rmsd` is the other half of self-consistency

ESMFold2 folds from sequence, so it discards DiffAb's backbone entirely. That
means ipSAE grades the designed **sequence**, not the designed **structure** --
a design whose sequence folds into a good binder with a completely different CDR
conformation still scores well. `sc_rmsd` closes that gap: the prediction is
superposed on the antibody framework (every antibody CA except the CDR under
test) and the CDR CA-RMSD is measured. Low ipSAE with low `sc_rmsd` means the
conformation was reproduced but the interface was not; high `sc_rmsd` means the
predictor disagrees with DiffAb about the CDR entirely.

### Caveats

- **Training-set contamination.** ESMFold2's data cutoff is September 2021, which
  covers 7DK2 and most of the SAbDab test set. The `ref_` arm is therefore partly
  memorised and is an optimistic ceiling, not a neutral reference. Design-versus-
  negative separation is the defensible comparison.
- **Only the best few designs are scored.** `ipsae_summary.csv` averages over
  `--top-n` per CDR, not over all 600, so it is not comparable to
  `binding_summary.csv`, which averages over everything. Compare like with like
  by merging on `filename`.
- **One diffusion sample per complex.** ESMFold2 is a diffusion model; a single
  sample is taken for cost. Expect run-to-run variation, and re-fold with a
  different `--negative-seed` if a number looks decisive.
- **ipSAE cutoffs are a choice.** Upstream examples use 15 for AF2-style input
  and 10 for AF3/Boltz. The default here is 10/10 and is recorded in every row,
  so runs at different cutoffs stay distinguishable.
- **`ipsae.py` mangles paths containing `.pdb`.** It derives its output filenames
  with `pdb_path.replace(".pdb", "")`, a global replace, and every DiffAb run
  directory is named `<target>.pdb_<timestamp>`. `run_ipsae` therefore scores a
  copy in a scratch directory and copies the reports back; do not "simplify" it
  to run in place.

## Interface contacts

`interface_reference.csv` (one row per residue) and `interface_summary.csv` (one
row per region), from `diffab/tools/eval/interface.py`.

Chain assignment is **not hardcoded**: antibody chains come from the metadata
(`task.ab_chains`), every other chain is treated as antigen — the same convention
as `energy.eval_interface_energy`.

### Which reference file is read

The reference is read **unrelaxed** by default (`REF1.pdb`), chosen by `--ref-pfx`
independently of `--pfx`. References are then deduplicated **by file content**, so
the six byte-identical copies of `REF1.pdb` — one per CDR directory, and identical
to `reference.pdb` at the run root — collapse to a single structure.

This matters. The relaxed `REF1_<pfx>.pdb` copies are **not six replicates of one
measurement**, and must not be treated as such. Each was produced by relaxing a
*different single CDR*, which:

- moves that CDR's backbone (0.51–0.99 A);
- drags every residue **downstream in the same chain** through Rosetta's
  torsion-space fold tree, even though the movemap never enabled them — relaxing
  H_CDR1 shifts H_CDR2 by 0.31 A and H_CDR3 by 0.30 A, while relaxing H_CDR3
  shifts neither;
- repacks spatial neighbours, which reaches across chains because CDR loops
  cluster (side chains move up to 4.8 A);
- leaves everything else **bit-identical** to the native.

So in the H_CDR3 reference, L_CDR1's backbone was never touched. Asking whether a
residue contacts "in all six references" compares one file where its CDR was
relaxed against five where it was not — a mixture, not a replicate count. Use
`--ref-pfx rosetta` only if you specifically want to inspect a relaxed reference;
the summary then reports each reference on its own rows and never merges them.

### The cutoff: 4.5 A minimum heavy-atom distance

A contact is a pair of residues whose closest **heavy atoms** (hydrogens excluded)
lie within 4.5 A. Measured on the 7DK2 native — 226 antibody residues in chains
A/B, 191 antigen residues in chain C:

```
  <= 4.0 A : 28 antibody / 26 antigen
  <= 4.5 A : 31 antibody / 29 antigen     <- primary
  <= 5.0 A : 37 antibody / 33 antigen
  <= 6.0 A : 43 antibody / 44 antigen
  closest contact: 2.66 A
```

Three reasons for 4.5:

1. **It sits in the measured density trough.** Binning the antibody-side minimum
   distances at 0.5 A gives 13 residues in 3.0–3.5, 10 in 3.5–4.0, then only
   **3 in 4.0–4.5**, then 6 in 4.5–5.0. The distribution is otherwise continuous,
   so this is its thinnest point rather than a discovered boundary — the cutoff
   is a convention, just a well-placed one. `interface.py` prints the histogram on
   every run so this stays auditable.
2. **It cross-validates against a definition using no distance at all.** Bio.PDB
   `ShrakeRupley` SASA burial (dSASA > 1 A^2, antibody alone minus antibody in
   complex) flags 40 residues and agrees with the 4.5 A cutoff on 96% of residues.
3. It is the conventional antibody–antigen interface definition in the literature.

4.0, 5.0 and 6.0 are reported alongside as a sensitivity check.

### Do not use `CA-CA <= 6.0`

`diffab/utils/transforms/mask.py:194` defines `contact_flag = (nn_ab_dist <= 6.0)`
on Cα–Cα distances. That threshold exists to sample a single **patch anchor** for
cropping, not to define an interface, and it is the wrong tool here: on this
structure it selects **9** antibody residues against 31 at 4.5 A heavy-atom. Cα
positions ignore side-chain reach, and this interface is overwhelmingly
side-chain mediated — TYR, GLU, TRP, PHE and ARG dominate the contact list.

### `interface_reference.csv`

| Column | Meaning |
| --- | --- |
| `method`, `structure` | which run this row came from |
| `reference` | the reference file, relative to the run directory (e.g. `H_CDR1/REF1.pdb`) |
| `side` | `antibody` or `antigen` |
| `region` | CDR tag (`H_CDR3`, ...), `framework-<chain>`, or `antigen-<chain>` |
| `chain`, `resseq`, `icode`, `resname` | residue identity; `icode` is the insertion code, blank for most |
| `min_dist` | minimum heavy-atom distance to any residue on the opposite side, A |
| `dsasa` | solvent-accessible surface area lost on binding, A^2 |
| `contact` | `min_dist <= 4.5` |

`min_dist` is `inf` for residues with no partner atom within 14 A
(`SEARCH_RADIUS`). Constants are at the top of `interface.py`.

### `interface_summary.csv`

| Column | Meaning |
| --- | --- |
| `method`, `structure`, `reference` | as above |
| `side`, `region` | as above |
| `n_contact` | contacting residues in this region |
| `residues` | their identities, `A26 GLY; A27 PHE; ...` |

One row per `(reference, side, region)`. Rows from different references are never
combined, for the reason given above.

### The 7DK2 native interface

```
region           n  residues
H_CDR1           5  A26 GLY, A27 PHE, A28 THR, A31 SER, A32 TYR
H_CDR2           2  A52 LYS, A56 GLU
H_CDR3           5  A96 LEU, A97 GLY, A98 ILE, A100 TRP, A100C ASP
L_CDR1           2  B31 ASN, B32 SER
L_CDR2           5  B50 ALA, B53 THR, B54 LEU, B55 GLU, B56 SER
L_CDR3           5  B91 PHE, B92 TYR, B93 SER, B94 THR, B96 ARG
framework-A      4  A1 GLU, A3 GLN, A33 TRP, A58 TYR
framework-B      3  B49 TYR, B57 GLY, B60 SER
                31  total  (24 CDR, 7 framework)
antigen-C       29
```

Chain A is the heavy chain, B the light (from `metadata.json`: `H_CDR*` map to A,
`L_CDR*` to B). These are Fv-only files, so "framework" means the Fv scaffold.

Points worth knowing:

- **All six CDRs bind and H3 does not dominate.** The single closest contact is
  L_CDR2 B55 GLU at 2.66 A. H_CDR2 and L_CDR1 contribute only 2 residues each.
- **The framework contributes 7 of 31 contacts.** Most are a CDR-definition
  artifact: DiffAb uses Chothia ranges, and A33 falls inside *Kabat* CDR-H1, A58
  inside Kabat CDR-H2. **A1 GLU (2.76 A) is not** — that is the chain N-terminus
  touching the antigen, worth inspecting before trusting it.
- **Contacts are many-to-many.** The 31 and 29 residues form **62 contacting
  pairs**: 2.00 partners per Fv residue, 2.07 per antigen residue. About half of
  each side touches exactly one partner, the rest two to five. The most connected
  are A1 and B56 (5 each) on the Fv side, C417 and C486 (5 each) on the antigen.
  The counts differ by 2 simply because the network is slightly lopsided, not for
  any deeper reason.
- **The epitope is the ACE2-binding receptor-binding motif of the SARS-CoV-2
  spike RBD**, including the well-known variant positions K417, E484, N501 and
  Y505.

### Not yet implemented: per-design contacts

This module analyses references only. The intended per-design metrics — contacts
made by the designed CDR, overlap of its epitope with the native one, and closest
approach to the antigen — are deferred. Note that generated structures *are*
relaxed, so comparing them against the relaxed reference (`--ref-pfx rosetta`) is
the like-for-like choice there, even though the unrelaxed native is the right
answer to "what is the native interface".

## Thresholds and constants

All at the top of `diffab/tools/eval/geometry.py`; change and re-run.

```python
IDEAL_PEP_CN = 1.329   # Engh & Huber (1991), sigma = 0.014
IDEAL_N_CA   = 1.458
IDEAL_CA_C   = 1.525
IDEAL_C_O    = 1.231
IDEAL_N_CA_C = 111.2   # degrees

PEP_TOL      = 0.05    # A, ~3.6 sigma
BREAK_TOL    = 0.30    # A, chain effectively severed
OMEGA_TOL    = 30.0    # degrees from planarity
CLASH_DIST   = 2.2     # A, non-bonded heavy-atom clash
CLASH_SEVERE = 1.5     # A
```

**Why `PEP_TOL = 0.05`.** Engh & Huber give the peptide C–N as 1.329 ± 0.014 Å,
and structure validation conventionally flags outliers at 3σ = 0.042 Å. So 0.05 Å
is 3.6σ — slightly *more permissive* than the standard criterion, not stricter.
`plot_geometry` prints the sensitivity sweep so this is always auditable.

**Why `BREAK_TOL = 0.30`.** The resulting window, 1.03–1.63 Å, brackets the range
in which a C–N bond can physically exist (C–N single ≈ 1.47, C=N double ≈ 1.28,
peptide ≈ 1.33 as a partial double bond). Outside it the atoms are not bonded.
This is the softer-justified of the two thresholds: a symmetric tolerance is not
really right, since compressing a bond is harder than stretching it, and an
absolute window such as "outside [1.0, 1.7] Å" would be more principled.

**Why `CLASH_DIST = 2.2`.** The sum of van der Waals radii for C/N/O is
~3.0–3.2 Å, but a hydrogen bond puts N···O at 2.8–3.0 Å. A threshold at vdW
contact would count every H-bond and every normal packing contact as a clash.
MolProbity resolves this by measuring overlap with explicit hydrogens and
excluding H-bonded donor/acceptor pairs; with no H-bond detection here, the
threshold is instead placed safely below H-bond distance, so anything caught is
unambiguous overlap. The cost is that mild clashes are under-counted — which is
why `term_fa_rep`, a soft longer-range repulsive, is the more sensitive measure
of residual strain.

## Caveats

- **Bond geometry is identical pre- and post-relax.** FastRelax samples torsions,
  which preserves every bond length and bond angle by construction; `ref2015` has
  no bond-length or bond-angle term, so it cannot penalise a violation either.
  Measured across 600 designs, the largest change in any bond length through
  relaxation is 0.0013 Å — the quantisation floor of the PDB format's three
  decimal places — while backbone atoms move a mean of 0.82 Å. Consequently
  `pre_pep_*` is redundant with `pep_*` and is omitted from the summary. Enabling
  `basic_idealize` (commented out in `relax/pyrosetta_relaxer.py`) or switching to
  cartesian relax with the `cart_bonded` term would change this.
- **Clashes are *not* preserved by relaxation** — `fa_rep` is torsion-accessible,
  so relax genuinely fixes them. Pre-relax is the only place they can be measured
  as generated.
- **Distributions matter more than violation rates.** Designed bond lengths form
  broad, roughly unimodal distributions centred near ideal (mode 1.23–1.31 Å)
  with a standard deviation of 0.15–0.31 Å against a native 0.003–0.008 Å. The
  failure is a 30–100× loss of *precision*, not occasional catastrophic outliers,
  though the tails do reach impossible values (0.38 Å, 3.82 Å).
- **`validate` ignores `evaluation_db`; `run` does not.** If `run` reports no new
  tasks, its shelve database already lists those files as visited.
- **`dG_ref` in `summary.csv` is not a constant.** Two independent causes. (a)
  `energy.py` scores `REF1_rosetta.pdb`, and the six CDRs have six differently
  relaxed natives — a 66.6 REU spread, so **`ddG` is not comparable across
  CDRs**. (b) `set_pack_separated(True)` repacks stochastically from the process
  RNG, giving 6–16 distinct values per 100 draws (`sd` 0.5–2.3 REU); `dG_ref` is
  recomputed for every design rather than once per CDR, which is what turns a
  fixed reference into 1200 noisy draws. `-constant_seed` reproduces a run
  exactly. `dG_int` in `binding_per_design.csv` has neither problem.
- **IMP% (`ddG < 0`) is fragile near zero.** For L_CDR3 a 2 REU shift in the
  threshold moves IMP% by 20 points, because designs pile up right where the
  reference sits — and the two 7DK2 runs' L_CDR3 references differ by 3.52 REU,
  six times the within-run `sd`. Scoring both runs against a common per-CDR
  threshold moved the pooled comparison from +6.8 to −1.7. Prefer a margin
  (`ddG < -5`) or `ddG_int`.

## Appendix: summary column order

### `validity_summary.csv`

31 columns, in file order. Each `ref_*` control follows the column it controls.

| # | Column | | # | Column |
| --- | --- | --- | --- | --- |
| 1 | `method` | | 17 | `clash_n_max` |
| 2 | `structure` | | 18 | `ref_clash_n` |
| 3 | `cdr` | | 19 | `pre_clash_n_mean` |
| 4 | `pep_rmsd_ideal_mean` | | 20 | `pre_clash_n_max` |
| 5 | `ref_pep_rmsd_ideal` | | 21 | `pre_clash_n_severe_mean` |
| 6 | `pep_frac_viol_mean` | | 22 | `pre_clash_n_severe_max` |
| 7 | `ref_pep_frac_viol` | | 23 | `shift_ca_rmsd_mean` |
| 8 | `pep_n_break_mean` | | 24 | `shift_ca_rmsd_std` |
| 9 | `pep_n_break_max` | | 25 | `shift_ca_rmsd_max` |
| 10 | `pep_min_mean` | | 26 | `ref_shift_ca_rmsd` |
| 11 | `pep_min_min` | | 27 | `shift_bb_rmsd_mean` |
| 12 | `ref_pep_min` | | 28 | `shift_bb_rmsd_std` |
| 13 | `omega_dev_mean_mean` | | 29 | `shift_bb_rmsd_max` |
| 14 | `ref_omega_dev_mean` | | 30 | `ref_shift_bb_rmsd` |
| 15 | `omega_n_viol_mean` | | 31 | `shift_bb_max_max` |
| 16 | `clash_n_mean` | | | |

The layout is defined by `SUMMARY_SPEC` at the top of
`diffab/tools/eval/validate.py` as `(column, aggregations, reference column)`
triples; edit that list to add, drop or reorder columns. `validity_per_design.csv`
always carries the full 67 columns regardless.

### `terms_summary.csv`

17 columns. `terms_per_design.csv` carries the same set plus `filename`.

| # | Column | | # | Column |
| --- | --- | --- | --- | --- |
| 1 | `method` | | 10 | `term_rama_prepro` |
| 2 | `structure` | | 11 | `ref_term_rama_prepro` |
| 3 | `cdr` | | 12 | `term_fa_rep` |
| 4 | `term_total_no_ref` | | 13 | `ref_term_fa_rep` |
| 5 | `ref_term_total_no_ref` | | 14 | `term_p_aa_pp` |
| 6 | `term_total` | | 15 | `ref_term_p_aa_pp` |
| 7 | `ref_term_total` | | 16 | `term_n_res` |
| 8 | `term_total_per_res` | | 17 | `ref_term_n_res` |
| 9 | `ref_term_total_per_res` | | | |

Order comes from `REPORT_ORDER` in `rosetta_terms.py`.

### `binding_summary.csv`

12 columns. `binding_per_design.csv` carries the same set plus `filename`.

| # | Column | | # | Column |
| --- | --- | --- | --- | --- |
| 1 | `method` | | 7 | `dG_bind` |
| 2 | `structure` | | 8 | `ref_dG_bind` |
| 3 | `cdr` | | 9 | `ddG_bind` |
| 4 | `dG_int` | | 10 | `fa_rep_bound` |
| 5 | `ref_dG_int` | | 11 | `ref_fa_rep_bound` |
| 6 | `ddG_int` | | 12 | `delta_fa_rep_bound` |

Order comes from `REPORT_ORDER` and `DELTA_NAMES` in `binding.py`.

## Why these metrics exist

`rmsd` and `seqid` measure similarity to the reference. `dG`/`ddG` measure the
interface. None of them can see whether the generated backbone is chemically
possible:

- `rmsd` uses **CA atoms only**, so it never touches a bond length.
- `seqid` (AAR) discards coordinates entirely.
- `ref2015`, and therefore `dG`, `ddG` and `total_score`, has **no bond-length or
  bond-angle term**.
- FastRelax works in torsion space, so it **cannot** repair bond geometry.

The metrics here close that gap. `ref_*` controls are computed throughout, on the
native structure through the same protocol, so every number can be read against
the value a real antibody scores.

`dG` turned out to have a second, separate problem: it charges the whole
structure's side-chain packing strain to the binding term, so it measures
relaxation quality as much as affinity, and `ddG` cannot be compared across CDRs.
`dG_int` and `dG_bind` measure binding without that confound, and
`fa_rep_bound` measures the all-atom packing strain the backbone-only validity
metrics cannot see. The three together separate what `dG_gen` had merged.

All of that still reads coordinates DiffAb produced. `ipsae_max` and `sc_rmsd`
are the one check that does not: they refold the designed sequence with an
independent predictor and ask whether the complex comes back. A design can
satisfy every metric above and still fail that, which is why the native and
scrambled controls are folded alongside it rather than assumed.
