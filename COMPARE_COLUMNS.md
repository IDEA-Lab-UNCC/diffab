# Best-batch comparison: column reference

Columns of the two tables written by `python -m diffab.tools.eval.compare`:

| File | One row is | Rows on 7DK2 |
| --- | --- | --- |
| `compare_selected_k.csv` | one `(cdr, selector, evaluator, k)` cell | 288 |
| `compare_sequence.csv` | one `(run, cdr, batch)` group | 84 |

Both land in `<--out>/evaluation/`. See `EVALUATION_METRICS.md` §5b for how to run
it, and the [binding column reference](EVALUATION_METRICS.md#column-reference-binding)
for the metrics being compared.

> A third file, `compare_random_k.csv`, reports best-of-k under *random*
> selection. It is a deterministic function of the score distribution, so it
> tells you nothing the distribution itself does not — prefer
> `binding_summary.csv` and the violin plots for that. It is not documented here.

---

## Why this table exists

Every other summary in this repo averages over all 600 designs. Nobody assays
600 designs. They assay the best handful, so the question that decides whether a
checkpoint is worth anything is **"are the designs I would actually pick
better?"** — and a mean cannot answer it. A model that lifts hopeless designs to
merely-bad moves every mean in the table while being worth nothing; a model that
sharpens the top ten can look flat on the same numbers.

---

## `compare_selected_k.csv`

The mental model is one sentence: **rank all 100 designs by `selector`, keep the
top `k`, and report `evaluator` over that batch.**

The two metrics are deliberately different roles. A model fine-tuned on metric A
will win any comparison that ranks by A and reports A — that measures the
training objective, not the designs. Selecting by A and scoring on B asks whether
the designs you *would have picked* are better by a yardstick the model never saw.

### Identifier columns

| Column | Meaning |
| --- | --- |
| `cdr` | which CDR was designed (`H_CDR1`…`L_CDR3`); each has its own 100 designs |
| `selector` | the metric the top-k was **ranked by** — your computational filter |
| `evaluator` | the metric the selected batch is then **judged on** |
| `k` | batch size: 1, 5, 10, 25 |
| `circular` | `True` when `selector == evaluator`. **Not evidence** — see below |
| `better_when` | `lower` or `higher`, the direction of the **evaluator** metric |

### Value columns

Named from `--labels` (default `baseline`, `finetuned`):

| Column | Meaning |
| --- | --- |
| `<label>_mean` | the evaluator averaged over the k selected designs |
| `<label>_best` | the single best evaluator value within that batch |

Both are reported because a model can move one without the other. `_mean` says
the batch as a whole is better; `_best` says its best member is. If you will
synthesise all k, `_mean` is your expected outcome; if you only need one binder
to work, `_best` is closer to what matters.

Values are in the metric's own units and sign — `sc_value` reads as `sc_value`,
not as its negation.

### Difference and significance

| Column | Meaning |
| --- | --- |
| `delta_mean`, `delta_best` | second arm minus first, **sign-normalised** |
| `ci_lo_*`, `ci_hi_*` | 95% bootstrap CI on that difference (2000 resamples) |
| `sig_*` | `True` when the CI excludes zero |

**The delta convention is the thing to remember: negative always means the second
arm is better**, whichever way the metric itself reads. Without that you cannot
scan a table mixing `ddG_int` (lower better) with `sc_value` (higher better).

The bootstrap resamples **designs, then re-ranks them**. That is the point:
"the top 10 of 100" is itself a noisy thing to be, and resampling the
already-chosen batch would hold the selection fixed and badly understate the
spread. The draws therefore carry both sources of uncertainty — which designs the
run happened to produce, and which of them the selector happened to rank top.

### Two traps

**`circular == True` rows are not evidence.** They are kept so the diagonal is
visible, not so it can be quoted. On a checkpoint RL-trained on `dG_separated`,
every `ddG_*`-selected / `ddG_*`-evaluated cell is measuring the reward.

**`sig_*` is a CI test only — it does not know about measurement noise.**
`ddG_bind` is stochastic: re-scoring the same structures moves it by a median
0.60 REU and a p99 of 8.7. Cross-check any `ddG_bind` result against
`compare_random_k.csv`'s `noise_floor`, or against a replicate of your own. The
deterministic columns (`ddG_int` and every interface descriptor) have a measured
floor of exactly 0 and need no such check.

### What the evaluators mean

Six metrics are compared by default. They are not six views of one thing — each
answers a different question, and a design can pass one while failing another.
Native 7DK2 values are given as the yardstick, since an absolute number here
means little on its own.

| Evaluator | Question it answers | Native 7DK2 |
| --- | --- | --- |
| `ddG_int` | Do the two surfaces attract, in the pose as given? | −33 to −45 REU |
| `ddG_bind` | What survives once both sides are allowed to relax? | −0.7 to −21 REU |
| `sc_value` | Do the two surfaces physically *fit*? | 0.60 – 0.66 |
| `dSASA_int` | How much surface does the interface bury? | 2460 – 2540 Å² |
| `unsat_hbonds` | How many buried polar groups get no partner? | 19 – 27 |
| `dG_per_dSASA` | Is the interface *efficient* per unit area? | −1.33 to −1.81 |

**`ddG_int` — interaction energy.** Both partners frozen in their bound
conformation, so every one-body and intra-chain term cancels exactly and only
the genuine inter-chain terms survive. Strain-free and deterministic. It
*overstates* affinity, because a real antibody pays to hold itself in the bound
conformation and this never charges for it.

**`ddG_bind` — closer to a free energy.** Both states repacked, so the unbound
state gets the side-chain reorganisation that really does accompany unbinding.
The better physical quantity, and the noisier one — see the selector table below.

**`sc_value` — shape complementarity.** Purely geometric: how well the two
surfaces interdigitate, 0 (flat against knobbly) to 1 (perfect cast). **The key
property is that it is not an energy**, so a design cannot raise it by shrinking
its side chains until `fa_rep` falls — the trick that lowers every Rosetta
energy. Real antibody–antigen interfaces sit at 0.65–0.75; below ~0.55 the
surfaces are not really in contact.

**`dSASA_int` — buried interface area.** Total surface occluded on forming the
complex, across both partners. Bigger interfaces generally bind harder, so this
is read as *more is better* — but only alongside `dG_per_dSASA`, because burying
more surface badly is not an improvement. A drop here across a whole batch means
the model is designing physically smaller interfaces.

**`unsat_hbonds` — buried unsatisfied hydrogen bonds.** Polar atoms driven out
of water and into the interface without finding a partner. Each one is a real
energetic penalty (roughly 1–2 kcal/mol) and a standard filter in designed
binders. **Read it next to `dSASA_int`**: fewer unsatisfied H-bonds in a smaller
interface may be nothing more than less polar burial, not a better-satisfied one.
Dividing by `dSASA_int` separates the two.

**`dG_per_dSASA` — energy density.** `dG_separated` per 100 Å² buried. It exists
to close the loophole in raw energy: a design can improve `ddG_int` simply by
burying more surface, which this normalises away. A design that gets better on
`ddG_int` while getting worse here bought its energy with area, not with a better
interface.

### Choosing a selector

`ddG_int` and `ddG_bind` are both defaults, but they are not interchangeable as
*selectors*. Measured by re-scoring the same structures and re-ranking:

| Selector | top-10 preserved | why |
| --- | --- | --- |
| `ddG_int` | **10/10**, all 12 cases | deterministic — no packer runs |
| `ddG_bind` | 0/10 – 9/10, median ~6/10 | stochastic repacking |

Selecting takes an *extreme*, which amplifies noise; evaluating takes a *mean*,
which suppresses it. So rank on `ddG_int` and evaluate on whatever you like —
`ddG_bind` is the better physical quantity and is fine as an evaluator. To rank
on `ddG_bind` honestly, raise `--design-repeats` first; noise falls as √n.

---

## `compare_sequence.csv`

Properties of a **batch** of sequences rather than of a design. A top-10 of
near-identical sequences is one experiment, not ten — that cannot be expressed
per design, so it only exists in this table.

Sequences come from `similarity.extract_reslist`, the same residue slice `rmsd`
and `seqid` use, so they are exactly the designed positions.

### Identifier columns

| Column | Meaning |
| --- | --- |
| `run` | arm label |
| `cdr` | the designed CDR |
| `batch` | which subset: `all` (the 100), or `top{5,10,25}_by_{selector}` |
| `n` | sequences in the batch |

There is no `top1` batch — diversity of one sequence is meaningless.

### Diversity

| Column | Meaning |
| --- | --- |
| `n_unique` | distinct sequences |
| `frac_unique` | `n_unique / n` |
| `mean_hamming` | mean pairwise Hamming distance, in residues |
| `mean_entropy` | Shannon entropy per position, bits, averaged over positions |
| `n_clusters` | single-linkage clusters at Hamming ≤ 2 |

`n_clusters` is the honest count of *distinct designs*: `n_unique` counts a
one-residue variant as a separate sequence, which is not a separate experiment.
A batch whose `n_unique` is 100 but whose `n_clusters` is 1 has explored one idea.

**Only `mean_hamming` is comparable across batch sizes.** `n_unique` and
`n_clusters` are bounded by `n`, and `mean_entropy` falls mechanically as `n`
shrinks — on this data, baseline H_CDR3 goes 3.13 bits over all 100 to 1.61 over
the top 5, which is a sample-size artefact and not a diversity collapse. Compare
entropy only between arms **at the same `k`**.

### Composition

| Column | Meaning |
| --- | --- |
| `frac_aromatic` | share of residues in F/W/Y |
| `frac_small` | share in G/A/S |
| `net_charge` | (K+R) − (D+E), per design |
| `gravy` | mean Kyte–Doolittle hydrophobicity |

`frac_aromatic` earns its place: aromatics are the workhorses of real paratopes,
and an optimiser rewarded for shedding steric strain will design them out — a
drop here alongside a rise in `frac_small` is the signature of a model buying
score by shrinking the CDR rather than by binding better.

### Developability liabilities

Each motif appears twice:

| Suffix | Meaning |
| --- | --- |
| `<motif>_mean` | motifs per design |
| `<motif>_frac` | fraction of the batch carrying at least one |

| Motif | Pattern | Why it matters |
| --- | --- | --- |
| `n_glyc` | `N[^P][ST]` | N-glycosylation sequon — adds a glycan, can abolish binding |
| `deamidation` | `N[GSNTH]` | Asn deamidation; `NG` is the fast one |
| `isomerisation` | `D[GSDT]` | Asp isomerisation |
| `free_cys` | `C` | unpaired cysteine — misfolding and dimerisation |
| `oxidation` | `[MW]` | Met/Trp oxidation |

`_frac` is usually the number to read: one glycosylation sequon is as
disqualifying as three.

**Counted in framework context.** Every motif spans 2–3 residues, so it can
straddle the CDR boundary — an Asn at the last designed position forms a sequon
with the two framework residues after it. Two framework residues each side are
included, and matches lying *entirely* within the framework are subtracted, so
only motifs involving a designed residue are counted. This is not hypothetical:
on 7DK2 it takes H_CDR2's `n_glyc_frac` from 0.0 to 0.7 (baseline) and 0.9
(fine-tuned), sequons that a CDR-only scan misses completely.

### Native reference

| Column | Meaning |
| --- | --- |
| `native_seqid` | mean fraction identity to the native CDR |
| `native_<motif>` | the native CDR's own count for that motif |

`native_<motif>` is the baseline a design should be judged against: the natives
here carry 0–1 liabilities, so a batch at 0.7 is introducing them, not inheriting
them. `native_seqid` is the same quantity as `seqid` in `summary.csv`, recomputed
here so a batch can be read without joining tables.

---

## Reading the two together

The intended workflow:

1. **Pick the shortlist the way you actually would** — rank by `ddG_int`, take
   the top 10.
2. **In `compare_selected_k.csv`, read that batch on evaluators you did not
   select by.** `sc_value` for geometric fit, `unsat_hbonds` for satisfiability,
   `dG_per_dSASA` for energy density. Ignore `circular` rows.
3. **In `compare_sequence.csv`, check the same batch is a real batch** — that
   `n_clusters` is not 1, and that the liabilities have not gone up.

A batch that improves on the energy it was selected by, while its clusters
collapse and its sequons multiply, has not improved.

A worked reading of a real pair of runs — baseline vs DDPO fine-tuned on 7DK2,
top-5 by `ddG_bind` — lives next to the CSVs it describes, in
`results/codesign_single/7DK2_AB_C.pdb_2026_09_02__16_28_52/evaluation/BEST_BATCH_COMPARISON.md`.
