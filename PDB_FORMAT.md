# Understanding the PDB File Format

Notes on reading the protein structure files used by DiffAb, using
[`data/examples/7DK2_AB_C.pdb`](data/examples/7DK2_AB_C.pdb) as the worked example.
Files in the SAbDab dataset (`data/all_structures/chothia/*.pdb`) and the model's
own output (`results/.../0000.pdb`) use the same format.

For what the model puts *inside* those files — the `(R, t, s)` diffusion output,
why sampled CDRs are backbone-only, and the relax/eval pipeline — see
[`HOW_DIFFAB_WORKS.md`](HOW_DIFFAB_WORKS.md).

## The one rule that matters

**PDB is fixed-width, defined by byte position.** Every `ATOM` line is exactly 80
characters — the width of a 1970s punch card, which is where the format comes from.
Fields are located by column number, *not* by splitting on whitespace.

## Anatomy of an ATOM / HETATM line

Line 1694 of the example file:

```
HETATM 1692  O   HOH A 313      17.555 -26.764  20.869  1.00 31.87           O
```

| Cols | Field | Value here | Meaning |
|---|---|---|---|
| 1–6 | record name | `HETATM` | line type (see [Record types](#record-types)) |
| 7–11 | **serial** | `_1692` | atom index, unique in the file, right-justified |
| 12 | *(blank)* | `_` | — |
| 13–16 | **atom name** | `_O__` | which atom within the residue |
| 17 | **altLoc** | `_` | alternate conformation marker (`A`/`B` if the atom was modeled in two positions) |
| 18–20 | **resName** | `HOH` | residue type — 3-letter amino acid, or `HOH` = water, `NAG` = a sugar |
| 21 | *(blank)* | `_` | — |
| 22 | **chainID** | `A` | which chain |
| 23–26 | **resSeq** | `_313` | residue number within the chain |
| 27 | **iCode** | `_` | insertion code — the antibody-numbering letter (e.g. `100A`) |
| 28–30 | *(blank)* | `___` | — |
| 31–38 | **x** | `__17.555` | coordinate in **ångströms** (1 Å = 10⁻¹⁰ m) |
| 39–46 | **y** | `_-26.764` | |
| 47–54 | **z** | `__20.869` | |
| 55–60 | **occupancy** | `__1.00` | fraction of copies where the atom sits here. `1.00` = always; `0.50` = split between two conformations |
| 61–66 | **B-factor** | `_31.87` | temperature factor — positional blur, in Å². Low ≈ rigid / well-resolved, high ≈ floppy or poorly resolved |
| 67–76 | *(blank)* | | segment ID, obsolete |
| 77–78 | **element** | `_O` | element symbol, right-justified |
| 79–80 | **charge** | `__` | formal charge, e.g. `1+`, `1-` |

(`_` marks a significant space.)

Note that `resSeq 313` is in the 300s while chain A's protein runs 1–223.
Crystallographers park waters and ligands in a high number range so they don't
collide with real residues. This particular line is one of the 41 `HETATM`
records DiffAb discards — it never reaches the model.

## Two traps this file demonstrates

### 1. You cannot split on whitespace

Line 1678:

```
ATOM   1677  O   CYS A 223     -28.075  28.139  28.445  1.00113.08           O
                                                        ^^^^^^^^^^
```

That B-factor is 113.08 — over 100, so it fills all six of its columns and
**collides with occupancy**. Naive splitting yields:

```
[9]=28.445   [10]=1.00113.08   [11]=O      <- one mangled field, everything after shifts
```

Negative coordinates cause the same failure: `-28.075  28.139` reads fine, but
`-128.075-28.139` would not. This is why the codebase always slices —
`line[21]`, `line[22:26]` — and why you should use Biopython rather than
hand-rolling a parser.

### 2. Atom names are right-justified from column 14, not 13

The element symbol occupies columns 13–14. So:

```
cols 13-16:  "_CA_"   = alpha Carbon (a protein backbone atom)
cols 13-16:  "CA__"   = CAlcium ion
```

One space of difference, completely different chemistry.

**Naming convention inside a residue:** `N, CA, C, O` are the backbone. Side-chain
atoms are lettered outward from the backbone by Greek letter — **B**eta, **G**amma,
**D**elta, **E**psilon, **Z**eta, **H**eta — giving `CB` → `CG` → `CD` → `CE`, with
branches numbered (`CG1`/`CG2`, `OE1`/`OE2`). So `ARG`'s `NH1` is a nitrogen far out
at the eta position, and `CYS`'s `SG` is its sulfur at the gamma position:

```
ATOM   1674  N   CYS A 223  ...   backbone nitrogen
ATOM   1675  CA  CYS A 223  ...   alpha carbon
ATOM   1676  C   CYS A 223  ...   backbone carbonyl carbon
ATOM   1677  O   CYS A 223  ...   backbone carbonyl oxygen
ATOM   1678  CB  CYS A 223  ...   side chain, beta position
ATOM   1679  SG  CYS A 223  ...   side chain sulfur, gamma position
```

Six lines = one cysteine residue. Hydrogens are absent, which is normal for X-ray
structures — they scatter X-rays too weakly to be seen.

## Charge column in action

Column 79–80 is populated for ionized groups:

```
ATOM      9  OE2 GLU A   1   ...   O1-      glutamate carboxyl, deprotonated
ATOM    130  NH1 ARG A  19   ...   N1+      arginine guanidinium, protonated
```

56 `N1+` and 49 `O1-` in this file. DiffAb ignores these — it works from atom
identity and position only.

## Record types

The example file contains:

```
1 CRYST1  |  4828 ATOM  |  41 HETATM  |  3 TER  |  15 CONECT  |  1 END
```

### `ATOM` vs `HETATM`

The only real distinction is *standard biopolymer residue* vs *everything else*.
Same 80-column layout, different first word. Here: 4828 protein atoms vs 41 hetero
atoms (27 waters + 14 atoms of one NAG sugar).

### `CRYST1`

Crystal lattice parameters — line 1 of the file:

```
CRYST1  113.730  102.810  163.420  90.00 107.16  90.00 P 1 21 1      1
        a        b        c        alpha beta   gamma  space group   Z
```

Unit cell edge lengths in Å, the three angles in degrees, and `P 1 21 1` = the
lattice symmetry group. Pure crystallography metadata. This is the record that
reveals that the 4 copies of the complex in PDB entry 7DK2 (chains A/B/C, D/E/F,
G/H/I, J/K/L) are a crystal-packing artifact rather than biology. Irrelevant to
structure prediction; DiffAb never reads it.

### `TER`

Chain terminator, marking the end of a connected polymer. Three of them, placed
exactly where each protein chain stops and its solvent begins:

```
ATOM   1679  SG  CYS A 223  ...      <- last real residue of chain A
TER
HETATM 1680  O   HOH A 301  ...      <- waters, still labeled chain A but not part of the polymer
```

Note the waters carry chain ID `A` even though they aren't chain A — `TER` is what
disambiguates.

### `CONECT`

Explicit bond list, given only for non-standard residues where bonding can't be
inferred from a residue template:

```
CONECT 3402 4854            <- atom 3402 (a protein Asn side chain) bonded to 4854 (NAG C1)
CONECT 4854 3402 4855 4862
```

Format: one anchor atom serial, then the serials it bonds to. That first line
records the glycosylation — the sugar covalently attached to the spike protein.
Standard amino acids get no `CONECT`; their internal bonds are looked up by
residue name.

### `END`

Last line, nothing more.

## What DiffAb actually reads

Out of everything above, `parse_biopython_structure()` in
[`diffab/utils/protein/parsers.py`](diffab/utils/protein/parsers.py) keeps exactly:

| Column | → tensor |
|---|---|
| 22 chainID | `chain_id` |
| 23–26 resSeq + 27 iCode | `resseq`, `icode` — kept so output can be mapped back to input |
| 18–20 resName | `aa` (integer 0–20) |
| 13–16 atom name | slot index into the fixed 15-atom layout |
| 31–54 x/y/z | `pos_heavyatom`, shape `[N, 15, 3]` |
| *(absence of a line)* | `mask_heavyatom`, shape `[N, 15]`, set `False` |

Dropped: serial, altLoc, occupancy, B-factor, element, charge, all `HETATM` lines,
`CRYST1`, and `CONECT`.

The 15-atom layout is defined by `restype_to_heavyatom_names` in
[`diffab/utils/protein/constants.py`](diffab/utils/protein/constants.py) — a fixed
padded layout (`N, CA, C, O, CB, ..., OXT`) where Ala fills 5 slots and Arg fills 11,
so `mask_heavyatom` records which slots are real. Atoms missing from the crystal are
also masked `False`.

On the way out, `save_pdb()` in
[`diffab/utils/protein/writers.py`](diffab/utils/protein/writers.py) hardcodes
`occupancy=1.00, B-factor=0.00` — a generated model has no experimental uncertainty
to report. Generated files therefore contain no waters, no glycans, and no `CRYST1`,
but preserve chain IDs and residue numbers so a design can be diffed against
`reference.pdb` position by position.

## Antibody numbering: locating the Fv and the CDRs

The columns above say *where* a residue number sits. They say nothing about what it
means. For antibodies the numbers themselves carry meaning — but only after
renumbering.

### The input is a Fab; `reference.pdb` is the Fv

DiffAb does not model what you hand it. Compare the example input against the
`reference.pdb` written into the results directory:

| | `data/examples/7DK2_AB_C.pdb` | `results/.../reference.pdb` |
|---|---|---|
| chain A (heavy) | 223 residues, numbered 1–223 | 120 residues, Chothia 1–113 |
| chain B (light) | 214 residues, numbered 1–214 | 106 residues, Chothia 1–106 |
| chain C (antigen) | 191 residues, 336–526 | 191 residues, **336–526 unchanged** |

Chain A lost its CH1 domain, chain B lost CL, and the antigen passed through
untouched — it keeps its deposited numbering, so antigen and antibody residue
numbers in the same file come from different coordinate systems entirely.

`113` and `106` are *last residue numbers*, not counts. Chain A holds 120 residues
within 113 numbers because Chothia numbering absorbs the extra ones as insertion
codes in column 27 — here `52A`, `82A/82B/82C`, `100A/100B/100C`. **Never infer a
residue count from the final residue number in an antibody file**, and never count
residues by numbering alone.

### Chothia renumbering is an external standard, not DiffAb's

DiffAb contributes only the ~80-line wrapper in
[`diffab/tools/renumber/run.py`](diffab/tools/renumber/run.py). Underneath:

- **Chothia numbering** — Chothia & Lesk (1987), refined by Al-Lazikani et al. (1997).
  One of roughly five established schemes (Kabat, Chothia, Martin/"enhanced Chothia",
  IMGT, AHo), each with *different* CDR boundaries. Chothia is structure-based;
  Kabat is sequence-hypervariability-based.
- **ANARCI** — Dunbar & Deane (2016), Oxford OPIG. HMMER germline alignment, the de
  facto tool for applying any of those schemes. Same group maintains SAbDab, which is
  why SAbDab ships pre-renumbered `chothia/` and `imgt/` directories.
- **abnumber** — the Python wrapper over ANARCI that DiffAb actually calls.

⚠️ The hardcoded ranges below are **Chothia-specific**. Passing `--no_renumber` on an
IMGT- or Kabat-numbered file masks the wrong residues silently — no error, just a
design in the wrong place. `scheme='chothia'` is Chothia proper, not Martin.

### How the Fv boundary is found

One `abnumber` call does three jobs ([`run.py:16`](diffab/tools/renumber/run.py)):

```python
abchain = abnumber.Chain(seq, scheme='chothia')
offset  = seq.index(abchain.seq)
```

| Return value | Used for |
|---|---|
| `abchain.seq` | the recognized Fv subsequence. Residues outside it get `numbers[i] = None` and are **dropped** by `renumber_biopython_chain()` (`run.py:36-37`) — this is the actual Fv crop, no fixed cutoff involved |
| `abchain.chain_type` | `'H'` vs `'K'/'L'`, so heavy/light are auto-detected (`run.py:57-61`) and `--heavy`/`--light` get defaulted (`design_for_pdb.py:110-113`) |
| `ChainParseError` | chain has no valid Fv → copied unchanged, classified `other_chains` → **the antigen** |

The renumbered intermediate is written next to the input as
`<name>_chothia.pdb` (`design_for_pdb.py:107`), then that file — not the original —
is what gets parsed.

A **second, redundant crop** lives in
[`diffab/datasets/custom.py:42,51`](diffab/datasets/custom.py):

```python
max_resseq = 113    # Chothia, end of Heavy chain Fv
max_resseq = 106    # Chothia, end of Light chain Fv
```

This one matters on the `--no_renumber` path: SAbDab's `chothia/*.pdb` files number
the *whole* Fab, constant domain continuing past 113, so this cut is what removes
CH1/CL when ANARCI hasn't already done it.

### CDR positions are fixed constants

Native CDR indices and lengths differ per antibody. Renumbering is what erases that:
after Chothia renumbering, the boundaries are the same integers for every antibody,
so they can be hardcoded
([`constants.py:13-19`](diffab/utils/protein/constants.py)):

```python
class ChothiaCDRRange:
    H1 = (26, 32);  H2 = (52, 56);  H3 = (95, 102)
    L1 = (24, 34);  L2 = (50, 56);  L3 = (89, 97)
```

What varies between antibodies is CDR **length**, absorbed by insertion codes rather
than by shifting boundaries. For 7DK2:

| CDR | Chain | Range | Slots | Residues | Insertions |
|---|---|---|---|---|---|
| H_CDR1 | A | 26–32 | 7 | 7 | — |
| H_CDR2 | A | 52–56 | 5 | 6 | `52A` |
| H_CDR3 | A | 95–102 | 8 | 11 | `100A`, `100B`, `100C` |
| L_CDR1 | B | 24–34 | 11 | 11 | — |
| L_CDR2 | B | 50–56 | 7 | 7 | — |
| L_CDR3 | B | 89–97 | 9 | 9 | — |

### `metadata.json` is generated at runtime, not supplied

It does **not** come from SAbDab. `data/sabdab_summary_all.tsv` carries
`pdb, Hchain, Lchain, antigen_chain, resolution, date, …` — chain IDs and split
metadata, **no CDR positions at all**.

`create_data_variants()` (`design_for_pdb.py:22-93`) applies `MaskSingleCDR` per CDR,
then reads the boundaries back off the resulting mask
([`inference.py:28-34`](diffab/utils/inference.py)):

```python
loop_idx = torch.arange(loop_flag.size(0))[data['generate_flag']]
idx_first, idx_last = loop_idx.min().item(), loop_idx.max().item()
```

So `residue_first`/`residue_last` are a *readback of what was actually diffused* —
the **observed** endpoints, not the constants. If Chothia position 26 were absent from
the structure, `residue_first` would read `A 27`.

### Reading the output: only the target CDR changes

Every sample file (`H_CDR1/0000.pdb`, …) contains the **entire complex**. The antigen
and the non-target antibody chain are byte-identical copies of `reference.pdb`; so is
all of the target chain outside the CDR. Diffing a sample against `reference.pdb`
yields changes confined exactly to that CDR's range — both coordinates and residue
identities, since `codesign` mode designs sequence and structure jointly.

To extract the designed loop, drive it off `metadata.json` rather than hardcoding, and
slice by residue number *plus* insertion code — a `95–102` range covers 11 residues
when `100A/100B/100C` are present:

```python
import json, glob
m = json.load(open('metadata.json'))
for it in m['items']:
    ch, lo, hi = it['residue_first'][0], it['residue_first'][1], it['residue_last'][1]
    for f in sorted(glob.glob(f"{it['tag']}/*.pdb")):
        seen = []
        for l in open(f):
            if l.startswith('ATOM') and l[21] == ch and lo <= int(l[22:26]) <= hi:
                if l[22:27] not in seen:      # cols 23-27 = resSeq + iCode
                    seen.append(l[22:27])
        print(it['tag'], f, len(seen))
```

### A CDR can be *shorter* than its range — and nothing warns you

Insertion codes handle longer loops; **gaps** handle shorter ones. The numbering
scheme does not require every slot in a range to be occupied. Two causes, which the
code cannot distinguish:

1. **Germline deletions** — a 5-residue CDR-H3 simply leaves 96–100 empty.
2. **Unresolved density** — flexible loop tips, CDR-H3 especially, are often missing
   from crystal structures.

The masking code is gap-tolerant by construction (`sabdab.py:79-83`), because it
iterates over *observed* residues rather than over the range:

```python
for position, idx in seq_map.items():
    cdr_type = constants.ChothiaCDRRange.to_cdr('H', position[1])
    if cdr_type is not None:
        cdr_flag[idx] = cdr_type
```

An absent position never gets flagged. No error, no warning. The only filters are
`cdr3_length > 30` → drop and `cdr3_length == 0` → drop (`sabdab.py:92-101`); **there
is no minimum length beyond zero**, so a 3-residue H3 passes silently.

**The consequence:** DiffAb generates exactly as many residues as were masked, and
gaps shrink the mask. A CDR-H3 with three disordered residues gets a *9*-residue loop
designed where the real one has 12 — the model does not fill the gap or rebuild the
missing backbone. In `codesign_single` mode the length is pinned to the mask, since
`augmentation=False` (`design_for_pdb.py:31`) means `random_shrink_extend` never fires
(`mask.py:62-63` — training-time only).

So check occupancy before trusting a design: compare the `residue_first`→`residue_last`
span in `metadata.json` against the residue count actually present in that range. If
the count falls short of the span, you have gaps — pick a structure with a complete
loop, or pre-fill it with a loop modeler.

## Reference

PDB format: [official specification v3.3](https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html)

Antibody numbering:

- Chothia, C. & Lesk, A. M. (1987) *Canonical structures for the hypervariable regions of immunoglobulins.* J Mol Biol 196:901-917.
- Al-Lazikani, B., Lesk, A. M. & Chothia, C. (1997) *Standard conformations for the canonical structures of immunoglobulins.* J Mol Biol 273:927-948.
- Dunbar, J. & Deane, C. M. (2016) *ANARCI: antigen receptor numbering and receptor classification.* Bioinformatics 32:298-300.
- [abnumber](https://github.com/prihoda/AbNumber) — the Python wrapper DiffAb calls.
- [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab) — source of `data/all_structures/chothia/`.
