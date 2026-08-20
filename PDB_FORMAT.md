# Understanding the PDB File Format

Notes on reading the protein structure files used by DiffAb, using
[`data/examples/7DK2_AB_C.pdb`](data/examples/7DK2_AB_C.pdb) as the worked example.
Files in the SAbDab dataset (`data/all_structures/chothia/*.pdb`) and the model's
own output (`results/.../0000.pdb`) use the same format.

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

## Reference

Official specification: [PDB File Format v3.3](https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html)
