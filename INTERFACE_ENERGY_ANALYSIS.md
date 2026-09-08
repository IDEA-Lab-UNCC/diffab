# What `dG` actually measures

A diagnostic investigation into DiffAb's `dG_gen` / `dG_ref` / `ddG`, run on the
7DK2 anti-SARS-CoV-2 codesign results. Every number below was measured on this
tree; nothing is quoted from literature.

**Conclusion up front.** `dG_gen` is not a binding energy. It is
`interaction energy + packing strain`, and the strain term is roughly 10x larger
than the binding term, so it dominates. This is caused by one asymmetric flag in
`energy.py`, interacts badly with DiffAb's *local* relax, and is the reason
`dG_ref` is neither a single number nor comparable across CDRs.

Two new columns fix it. See [6. The fix](#6-the-fix) and
[`EVALUATION_METRICS.md`](EVALUATION_METRICS.md) → *Column reference: binding*.

## Contents

- [1. `dG_ref` is not one number](#1-dg_ref-is-not-one-number)
- [2. Why `dG` is positive](#2-why-dg-is-positive)
- [3. The identity that explains everything](#3-the-identity-that-explains-everything)
- [4. What "repackable strain" is](#4-what-repackable-strain-is)
- [5. Consequences for `ddG` and IMP%](#5-consequences-for-ddg-and-imp)
- [6. The fix](#6-the-fix)
- [7. Corrected baseline vs fine-tuned comparison](#7-corrected-baseline-vs-fine-tuned-comparison)
- [8. Reproducing every number here](#8-reproducing-every-number-here)
- [9. Open items](#9-open-items)

## 1. `dG_ref` is not one number

There is exactly **one** native structure. Verified: all six `REF1.pdb` files in
a run and the top-level `reference.pdb` share one MD5 (`8840f8d3…`), and the two
7DK2 runs are **coordinate-identical** — max displacement `0.0000 A` across all
3251 atoms, matched by `(chain, resseq, icode, atom name)`. Their MD5s differ
only because the runs write chains in different order (`A,B,C` vs `C,B,A`), which
`InterfaceAnalyzerMover` does not care about (it parses by chain id).

But `dG_ref` is **not computed on that file**. `base.py:92` sets
`ref_fname = f'REF1_{postfix}.pdb'`, so `energy.py:49` scores the *relaxed*
reference — and those six files have six different MD5s.

### Cause A: six different relaxed references (66.6 REU)

`pyrosetta_relaxer.py` uses `subset='nbrs'`. Backbone moves only inside
`gen_selector` (the one CDR); side chains repack only in a
`NeighborhoodResidueSelector` shell around it; **everything else keeps its raw
input side chains**. Each CDR sits somewhere different, so each reference is the
native with a *different patch* cleaned up.

| | H_CDR1 | H_CDR2 | H_CDR3 | L_CDR1 | L_CDR2 | L_CDR3 |
| --- | --- | --- | --- | --- | --- | --- |
| residues moved > 0.1 A | 109 | 74 | 75 | 94 | 66 | 58 |
| of the 31 interface residues | 16 | 11 | 17 | 17 | 16 | 11 |
| mean `dG_ref` | 71.13 | 101.32 | 73.42 | 88.98 | **37.72** | **104.30** |

A **66.6 REU spread across references that all descend from the same native.**
The unrelaxed native scores `135.31` REU, and every partial relaxation improves
on it by 32–99 REU. How much tracks how much of the *interface* was repacked
(`r = -0.65` against interface residues repacked; `r = -0.07` against total
residues moved — so it is specifically the interface that matters).

### Cause B: the packer's random seed (~2 REU)

`energy.py:33` sets `set_pack_separated(True)`, a Monte Carlo simulated anneal
drawing from the process RNG. Scoring the *same file* 12 times:

```
73.1337  73.1337  68.6578  68.8057  72.3205  73.1337
73.2815  69.4038  73.2815  73.1337  72.4684  69.4038
   mean 71.680   sd 1.879   7 distinct values
```

`sd 1.879` matches the `1.89` in `summary.csv` and the value set is the same, so
this fully accounts for the within-CDR scatter. Confirmed to be the seed — with
`-constant_seed`, two independent processes produce a bit-identical sequence:

```
default seed, run A:  69.552 72.468 69.552 69.552 73.134 73.134
default seed, run B:  73.134 73.282 73.134 72.517 69.552 73.282
-constant_seed, A:    73.282 68.658 68.041 69.404 69.552 73.282
-constant_seed, B:    73.282 68.658 68.041 69.404 69.552 73.282   <- identical
```

Two details. **Even with a fixed seed the six values differ from each other** —
it is one RNG stream that advances per packing run, so call #2 starts where #1
stopped. And the values collapse to only ~10 distinct outcomes per 100 because
the rotamer library is discrete: different random paths converge on the same
handful of local minima.

`energy.py` recomputes `dG_ref` for **every design** (1200 times across the two
runs) rather than once per CDR, which is what turns a fixed reference into 1200
noisy draws.

## 2. Why `dG` is positive

A real antibody–antigen complex should score negative. It does. The positive sign
is a protocol artefact.

`InterfaceAnalyzerMover` computes one subtraction:

```
dG_separated  =  E(bound state)  -  E(unbound state)
```

Each side can optionally be **repacked** before being scored, with one flag each:

- `pack_input` → the **bound** state (the input complex as loaded)
- `pack_separated` → the **unbound** state (after the chains are pulled apart)

Repacking always lowers energy, so whichever side you enable gets a discount, and
that sets the sign. `energy.py` enables `pack_separated` and leaves `pack_input`
at its `False` default. All four combinations, same two files:

| | unrelaxed `REF1.pdb` | relaxed `L_CDR2/REF1_rosetta.pdb` |
| --- | --- | --- |
| pack neither | **−31.27** | **−41.56** |
| **pack separated only** ← DiffAb | **+135.49** | **+35.94** |
| pack input only | −158.72 | −96.99 |
| pack both | +9.01 | −17.29 |

The two **symmetric** rows give sensible binding energies. The two **asymmetric**
rows differ from each other by ~294 REU of pure artefact.

### Why the DiffAb setting is not crazy

Repacking the unbound state is *correct physics* — interface side chains that the
partner was holding in place really do relax when it leaves, and that is a real
cost of binding. `pack_separated=True` is the standard Rosetta recommendation.

The unstated assumption is that **the complex was relaxed thoroughly first**, so
the bound state carries no leftover strain and `pack_input` is unnecessary.
DiffAb relaxes only a ~10 A shell around one CDR, so ~250 residues keep raw
crystal side chains plus Rosetta-added hydrogens — strain the subtraction never
removes.

## 3. The identity that explains everything

```
dG_default  =  dG_interaction  +  R
```

`R >= 0` is the strain the **unbound** state can repack away, and only the
unbound state gets that chance. Per reference:

| reference | `dG_default` | interaction | R (strain) |
| --- | --- | --- | --- |
| unrelaxed `REF1.pdb` | 135.31 | −31.27 | 166.58 |
| relaxed (H_CDR1) | 71.13 | −38.03 | 109.16 |
| relaxed (H_CDR2) | 101.32 | −33.34 | 134.66 |
| relaxed (H_CDR3) | 73.42 | −35.67 | 109.09 |
| relaxed (L_CDR1) | 88.98 | −44.88 | 133.86 |
| relaxed (L_CDR2) | **37.72** | −41.56 | **79.28** |
| relaxed (L_CDR3) | 104.30 | −33.83 | 138.13 |

Across the six references the interaction energy varies by only **11.5 REU**
while `R` varies by **58.8**. And:

```
corr(dG_default, R)            = +0.985
corr(dG_default, interaction)  = +0.488
```

**`dG_gen` is essentially a strain meter.** L_CDR2 "wins" at 37.72 not because it
binds better but because its relax cleaned up the most (`R = 79` vs `138` for
L_CDR3). Everything anomalous in section 1 follows: the 66.6 REU reference
spread is `R` varying, and the 32–99 REU "gain" from relaxing is strain removal,
not improved binding.

## 4. What "repackable strain" is

```
R  =  E(chains apart, as-is)  -  E(chains apart, side chains repacked)
```

The energy that **side-chain rotamer re-selection alone** can remove — no
backbone movement, no minimisation. Decomposed by score term (positive =
repacking lowered it):

| term | unrelaxed ref | relaxed ref H3 | H3 low R | H3 high R |
| --- | --- | --- | --- | --- |
| `fa_dun` | 494 | 426 | 407 | 436 |
| `fa_rep` | 88 | 44 | **46** | **375** |
| `fa_sol` | 51 | 53 | 51 | 65 |
| `fa_atr` | −36 | −28 | −33 | −58 |

So `R` is two things: side chains in improbable torsion angles (`fa_dun`, the
large roughly-constant part) and atoms too close together (`fa_rep`, the part
that **varies**).

> **Caveat.** This decomposition repacks the whole pose, which over-relieves: it
> yields `R = 535–886` where the true values are `90–438`, so
> `InterfaceAnalyzerMover` clearly repacks only interface residues. The absolute
> numbers are indicative; the low-vs-high differential is what matters.

**`R` is not a property of binding at all.** That strain sits in the structure
whether the chains are together or apart — it is identical on both sides of the
subtraction. Only the unbound side is allowed to fix it.

### `fa_rep` is the term that varies, and your validity metrics are blind to it

`fa` = *full-atom*; `rep` = *repulsive*. Rosetta splits the Lennard-Jones 6–12
van der Waals potential into two separately weighted halves: `fa_atr` (attractive
r⁻⁶, always ≤ 0, weight 1.00) and `fa_rep` (repulsive r⁻¹², always ≥ 0, weight
0.55). `fa_rep` penalises every atom pair closer than the sum of their van der
Waals radii — **it is the steric clash term.** It is linearised at very short
range rather than diverging, which is why a badly clashing structure scores 1100
instead of infinity, and why minimisation can still pull atoms apart.

Bound-state `fa_rep` over the 311-residue complex, per CDR (n=1200):

| CDR | median | p90 | max |
| --- | --- | --- | --- |
| H_CDR1 | 532 | 546 | 554 |
| H_CDR2 | 517 | 534 | 582 |
| **H_CDR3** | **612** | **884** | **1100** |
| L_CDR1 | 530 | 540 | 609 |
| L_CDR2 | 517 | 539 | 551 |
| L_CDR3 | 519 | 542 | 567 |

Five CDRs sit in a tight 517–532 band — the shared, never-relaxed bulk of the
structure, identical in every design. **H_CDR3 is the only one that moves**,
reaching 2x baseline. Since `fa_rep` is a whole-complex sum, that entire excess
comes from the designed loop and its surroundings.

Spearman correlations, pooled (n=300) and H_CDR3 alone (n=50):

| | pooled | H_CDR3 |
| --- | --- | --- |
| `dG_gen` vs bound `fa_rep` | 0.18 | **0.72** |
| `R` vs bound `fa_rep` | 0.15 | **0.73** |
| interaction vs bound `fa_rep` | 0.29 | 0.25 |
| `fa_rep` vs `clash_n` | 0.35 | 0.27 |
| `fa_rep` vs `pep_n_break` | 0.28 | 0.32 |

H_CDR3's 350 REU dG spread is steric repulsion. But look at actual designs:

| | dG | interaction | R | `fa_rep` | `clash_n` | `pep_n_break` |
| --- | --- | --- | --- | --- | --- | --- |
| lowest R | 52.5 | −37.1 | 89.6 | **489** | 0 | 5 |
| | 67.6 | −28.7 | 96.3 | **493** | 0 | 4 |
| highest R | 392.2 | −21.3 | 413.5 | **915** | 0 | 4 |
| | 409.5 | −29.0 | 428.5 | **822** | 0 | 3 |

`clash_n` is **0 for all four**, and `pep_n_break` is *higher* in the good ones.
The `validity_per_design.csv` metrics cannot see this: `clash_n` counts only
backbone–backbone pairs under 2.2 A inside the CDR, while `fa_rep` is all-atom
repulsion across all 311 residues — overwhelmingly **side chains**, which DiffAb
never generated and relax only partially fixed.

The two are complementary, not redundant: validity covers backbone geometry,
`fa_rep` covers all-atom packing, and H_CDR3 fails on the second.

## 5. Consequences for `ddG` and IMP%

### The offset is not constant, so nothing can be rescaled

`dG_default - dG_interaction` over 300 designs (25 per CDR per run):

| run/CDR | mean | **sd** | min | max |
| --- | --- | --- | --- | --- |
| base/H_CDR1 | 104.0 | 3.6 | 99.3 | 113.1 |
| base/H_CDR2 | 134.5 | **1.5** | 132.6 | 139.8 |
| base/H_CDR3 | 213.9 | **120.3** | 89.6 | **438.3** |
| base/L_CDR1 | 135.1 | 2.1 | 131.2 | 138.2 |
| base/L_CDR2 | 85.5 | 7.0 | 77.8 | 101.4 |
| base/L_CDR3 | 140.8 | 5.2 | 136.4 | 156.4 |
| ft/H_CDR3 | 184.0 | **119.9** | 96.3 | 428.5 |
| **all** | 132.4 | 60.7 | 70.3 | 438.3 |

Within a geometrically clean CDR the offset is nearly constant (H_CDR2 sd 1.5),
so there `ddG` tracks interaction energy well. **H_CDR3's offset ranges over 350
REU**, so there `ddG` measures strain, not binding.

`ddG` correlation with the interaction-energy `ddG`: **0.99** for base/H_CDR2 but
**0.32** for base/H_CDR3 and **0.06** for ft/H_CDR3 — statistically no
relationship, in the CDR that matters most.

### `ddG` is not comparable across CDRs

Its references differ by 67 REU. Within a CDR it is fine, since `dG_gen` and
`dG_ref` come from structures relaxed with the same movemap.

### IMP% at exactly zero is fragile

For L_CDR3 a 2 REU threshold shift moves IMP% by 20 points, because designs pile
up right where the reference sits. And `dG_ref` for L_CDR3 differs by **+3.52
REU** between the two runs (104.30 → 107.83) — six times the within-run sd — so
each run was graded against a different bar. Rescoring both against a common
per-CDR threshold moved the pooled advantage from **+6.8 to −1.7**, and reversed
L_CDR3 from **+30 to −13**.

## 6. The fix

`diffab/tools/eval/binding.py` records three columns, each with a `ref_` control
and a delta. Written to `binding_per_design.csv` / `binding_summary.csv`;
**`summary.csv` is left untouched.**

| column | protocol | property |
| --- | --- | --- |
| `dG_int` | pack neither | interaction energy; deterministic; strain cancels **exactly** |
| `dG_bind` | pack both | closer to a real ΔG; stochastic, ~0.25 REU sd |
| `fa_rep_bound` | no packing | all-atom steric strain of the complex |

The reference is scored **once per CDR** and cached, not recomputed per design.

### `dG_int`'s cancellation is exact, not approximate

Scoring the H_CDR3 relaxed reference bound vs rigidly separated:

| term | complex | separated | difference |
| --- | --- | --- | --- |
| `fa_atr` | −2319.159 | −2242.640 | **−76.519** |
| `fa_sol` | 1343.664 | 1298.388 | **+45.276** |
| `fa_rep` | 473.166 | 454.252 | **+18.914** |
| `fa_elec` | −526.189 | −509.506 | **−16.682** |
| `lk_ball_wtd` | −40.868 | −37.445 | −3.423 |
| `hbond_bb_sc` | −45.205 | −43.272 | −1.933 |
| `hbond_lr_bb` | −150.024 | −149.177 | −0.847 |
| `hbond_sc` | −26.529 | −26.073 | −0.455 |
| `fa_dun` | 972.699 | 972.699 | **0.000** |
| `rama_prepro` | 82.037 | 82.037 | 0.000 |
| `p_aa_pp` | −53.738 | −53.738 | 0.000 |
| `omega`, `pro_close`, `ref`, `fa_intra_*`, `dslf_fa13`, `hbond_sr_bb`, `yhh_planarity` | | | 0.000 |
| **total** | 2.036 | 37.707 | **−35.671** |

**11 of 19 terms cancel to exactly zero**, including all 972 REU of `fa_dun`. The
separated pose is a rigid-body translation, so every one-body and intra-chain
term is identical by construction and only the 8 genuine inter-chain terms
survive. Hence `sd = 0.00`: no packer runs, no RNG is consumed.

### What each is and is not

`dG_int` is an **interaction energy, not a free energy**. The unbound state stays
frozen in its bound conformation, so it omits side-chain reorganisation, backbone
relaxation, and conformational entropy on unbinding — all of which make it
**overstate affinity**. It is also not fully relax-independent: the designed
CDR's side chains were built by relax and sit on the interface. What it removes
is the ~250-residue bulk strain that has nothing to do with binding.

`dG_bind` adds the unbound relaxation that `pack_separated` was designed to
capture, at the cost of stochasticity and of letting the packer partially repair
design flaws.

`fa_rep_bound` is not a binding term at all; it is recorded because it is what
actually drives the spread in `dG_gen`, and because `clash_n` cannot see it.

## 7. Corrected baseline vs fine-tuned comparison

`7DK2_AB_C.pdb_2026_03_03__09_37_49` (baseline) vs
`7DK2_AB_C.pdb_2026_09_02__16_28_52` (fine-tuned), 600 designs each.

IMP% = share of designs scoring better than the native:

```
        |    ddG (current)     |ddG_int (interaction) | ddG_bind (both packed)
cdr     |   base     ft   delta|   base     ft   delta|    base      ft   delta
H_CDR1  |     62     49     -13|      6      1      -5|     47     15     -32
H_CDR2  |     35     50     +15|     20     12      -8|     64     84     +20
H_CDR3  |     10     23     +13|     11      9      -2|      1      4      +3
L_CDR1  |      5      6      +1|      1      2      +1|     52     39     -13
L_CDR2  |      8      3      -5|     17      1     -16|     19      8     -11
L_CDR3  |     16     46     +30|     21     15      -6|     27     28      +1
ALL     |   22.7   29.5    +6.8|   12.7    6.7    -6.0|   35.0   29.7    -5.3
```

**The fine-tuned run's apparent advantage does not survive either new metric.**
Pooled: `+6.8` under the current `ddG`, but `−6.0` on interaction energy and
`−5.3` with both states packed. L_CDR3's headline `+30` collapses to `−6` and
`+1`, consistent with the common-threshold correction in section 5. H_CDR2 is
the one CDR that improves under two of three metrics.

Spearman against the current `ddG` runs 0.45–0.62 for `ddG_int`, so this is
genuinely different information rather than a rescaling.

Median `dG_bind` per CDR (lower is better):

| CDR | baseline | fine-tuned | `ref_dG_bind` (base) |
| --- | --- | --- | --- |
| H_CDR1 | −19.2 | −17.7 | −19.57 |
| H_CDR2 | −6.8 | −7.0 | −5.94 |
| **H_CDR3** | **+32.7** | **+6.9** | −20.24 |
| L_CDR1 | −8.0 | −5.8 | −7.94 |
| L_CDR2 | −13.9 | −12.8 | −16.91 |
| L_CDR3 | +1.6 | +3.0 | −0.17 |

Five of six CDRs now sit in a tight band straddling their native, which is what a
binding energy should look like — the ~135 REU strain pedestal is gone.
**H_CDR3 is the sole outlier and the only CDR whose designs mostly fail to bind
at all**; `dG_bind` does not rescue it (median +32.7 / +6.9 against −19 to +3
elsewhere), because the clashes are too severe for repacking to fix.

The honest read: **on interaction energy the fine-tuned checkpoint is not better
than baseline**, and H_CDR3 designs in both runs are dominated by steric strain
rather than by binding.

## 8. Reproducing every number here

```bash
# the new columns (~9 min per 600 designs on 12 cores)
python -m diffab.tools.eval.binding --root <run_dir> --workers 12

# violin comparison on dG_bind
python -m diffab.tools.eval.plot_energy \
  --runs <baseline> <fine-tuned> --labels baseline fine-tuned \
  --out <fine-tuned> --kind violin --metric dG_bind --clip 100
```

`--metric` also accepts `dG_int` and `dG_separated` (the default, reading
`summary.csv`). Output filenames derive from the metric, so nothing overwrites.

The one-off diagnostics — the 2x2 flag table, the term-cancellation table, the
seed test, the `R` decomposition — were run as throwaway scripts and are **not**
part of the repo. Each is a short `InterfaceAnalyzerMover` or `ref2015` call and
the tables above give the exact settings needed to reproduce them.

## 9. Open items

- **Not changed:** `energy.py` still uses `pack_separated=True` with
  `pack_input=False`, and still recomputes `dG_ref` per design. Fixing it would
  redefine `ddG` for every existing row, so it was left as a deliberate decision
  rather than a side effect. The oldest commit (`4db044b`, line 20) already has
  this setting, so published DiffAb `ddG` numbers carry it too.
- **A single relaxed reference** would remove Cause A entirely: relax the native
  once with `subset='all'` instead of once per CDR with `subset='nbrs'`.
- **`-constant_seed`** would make the existing `dG` reproducible, at the cost of
  hiding rather than removing the variance.
- **`7DK2_AB_C.pdb_5_relax`** was not scored with the new columns; the same
  command with a different `--root` covers it.
- **`dG_int` on all 1200 designs** exists; the pack-both/whole-pose diagnostics
  in sections 4–5 were run on a 300-design subset (25 per CDR per run) and could
  be extended if the correlations matter more precisely.
