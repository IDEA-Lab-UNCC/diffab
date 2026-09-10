# ipSAE evaluation — handoff

Written 2026-09-10 for whoever picks this up on the workstation. Assumes no
prior context.

## TL;DR

`python -m diffab.tools.eval.ipsae` is built, tested and working end to end. It
refolds designed antibody sequences with ESMFold2 and scores the
antibody–antigen interface with ipSAE, as an independent check on designs that
every other metric in this repo grades using DiffAb's own coordinates.

**It is blocked on a scientific question, not a bug.** On 7DK2, ESMFold2 folds
the antibody and antigen well and pairs heavy–light confidently, but does not
dock the antigen — including for the **native crystal complex**. ipSAE is
therefore 0.000 for every arm and nothing is interpretable.

**The next action is one command** (§5): a native-only sweep over all 22 runs,
to find out whether this is 7DK2-specific or systematic. Everything after that
depends on the answer.

---

## 1. What the tool does

Three complexes are folded per design, so a score can be read against controls:

| Arm | Folded | Column prefix |
|---|---|---|
| design | designed H + L + antigen sequences | *(none)* |
| native | the crystal antibody, same antigen — the ceiling | `ref_` |
| negative | designed chains with CDR residues shuffled in place — the floor | `neg_` |

Outputs `ipsae_per_design.csv` and `ipsae_summary.csv` into each run's
`evaluation/`, plus a fold cache in `evaluation/esmfold2/`.

Key columns: `ipsae_max` (headline, best Fv-vs-antigen ipSAE), `ipsae_H_Ag`,
`ipsae_L_Ag`, `iptm_H_Ag`, `iptm_L_Ag`, `pdockq`, `ptm`, `iptm`,
`pae_min_AbAg` (diagnostic — see §4), `sc_rmsd` (CDR CA-RMSD between
ESMFold2's prediction and DiffAb's design, after superposing on the framework).

Full documentation: `EVALUATION_METRICS.md` §"Column reference: ipSAE" and
run section 7.

## 2. Files

| Path | Role |
|---|---|
| `diffab/tools/eval/ipsae.py` | the eval module and CLI |
| `diffab/tools/fold/base.py` | `FoldTask`/`FoldResult` + the ipSAE input invariants |
| `diffab/tools/fold/esmfold2.py` | ESMFold2 wrapper, confidence extraction, mmCIF→PDB |
| `third_party/ipsae/` | pinned `ipsae.py` (v4, commit `6174cf9`, MIT) + provenance |

Commits: `dc5ccff`, `8e2cb90`, `c9d0970`, `c592311`, `5610549` on `new-eval`.

## 3. Environment

Runs in a dedicated conda env, **not** `diffab` and **not** `base`:

```
conda activate ipsae          # python 3.12
```

Why a separate env: `esm` requires Python ≥3.12 while the `diffab` env is 3.8,
and the eval chain imports no torch at all — `grep -rn "import torch"
diffab/tools/{eval,relax,fold}/` returns nothing. The two halves talk through
files on disk, never in one process:

```
diffab @3.8  → relax, binding.py → *_rosetta.pdb, binding_per_design.csv
ipsae  @3.12 → reads those, folds, → ipsae_per_design.csv
```

The workstation driver is 570.207 / **CUDA 12.8**, but `pip install esm` pulls a
CUDA 13 stack. The ipsae env was corrected with:

```
pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip uninstall -y cuequivariance-ops-torch-cu13
pip install cuequivariance-ops-torch-cu12==0.11.1
```

**Any reinstall of `esm` will drag `cuequivariance-ops-torch-cu13` back in.**
Use `pip install --no-deps esm`.

Hardware: 2× RTX 6000 Ada, 48 GB each. The ESMC stem is ~24 GB in fp32, so one
model per card. `--workers 2` needs `pip install ray`. `--dtype bf16` roughly
halves the footprint if anything OOMs.

## 4. What has been measured

Native arm on `7DK2_AB_C.pdb_2026_09_02__16_28_52`, `H_CDR3`, top-1 by `ddG_int`:

| | ESMFold2-Fast, 3/50 | ESMFold2 (full), 20/100 |
|---|---|---|
| `ref_pae_min_AbAg` | 13.8 Å | **11.6 Å** |
| `ref_iptm_H_Ag` | 0.082 | **0.169** |
| `ref_iptm_L_Ag` | 0.084 | **0.159** |
| `ref_ptm` | 0.571 | 0.590 |
| `ref_iptm` | 0.430 | 0.454 |
| `ref_ipsae_max` | 0.000 | 0.000 |

And from the Fast run's cached PAE matrices:

```
                pLDDT Ab   pLDDT Ag   H–L PAE min   Ab–Ag PAE min / median
design            87.2       82.8         0.6            13.7 / 25.3
native            86.5       83.4         0.6            13.8 / 25.3
negative          87.9       82.3         0.5            12.3 / 23.4
```

Interpretation, and the reason this is not a plumbing bug:

- Both chains fold well individually (pLDDT 83–88).
- **Heavy–light docks perfectly** (interchain PAE 0.6 Å). H–L pairing is itself
  a protein–protein docking problem, so the multi-chain API is being used
  correctly — we are not accidentally folding chains independently.
- The antigen is not docked: median Ab–Ag PAE ~25 Å against a ~31 Å ceiling.
- **Native, design and negative are indistinguishable.** There is no signal.

Raising the cutoff does not rescue it — on the native's PAE matrix:

```
pae_cutoff 10 →      0 of 43166 Ab–Ag pairs pass
pae_cutoff 12 →      0
pae_cutoff 15 →    157   (0.36%)
pae_cutoff 20 →   5702   (13.2%)
pae_cutoff 25 →  20811   (48.2%)
```

Scoring at 20–25 Å would be scoring noise. Going from Fast/3/50 to full/20/100
roughly doubled the interface confidence and is the right direction, but is not
close to sufficient.

## 5. The next action

Fold the native of **every** run and count how many dock. This distinguishes
"ESMFold2 cannot do antibody–antigen from single sequence" from "7DK2 is one of
the >50% of Ab–Ag targets that fail" — and those lead to completely different
next steps. It needs no PyRosetta and no design ranking.

```
python -m diffab.tools.eval.ipsae --root results/codesign_single --arms native --cdr H_CDR3 --workers 2 --layout both --out results
```

22 folds: 19 testset runs + 3 × 7DK2, spanning ~11 distinct PDB entries. Roughly
1.5–2 h on two cards if a fold is ~10 min. The three 7DK2 runs share one native
sequence but have separate cache directories, so those three folds double as a
free read on run-to-run diffusion variance.

Then read `ref_pae_min_AbAg` and `ref_ipsae_max` per structure:

```
python -c "import pandas as pd; t=pd.read_csv('results/evaluation/ipsae_per_design.csv'); print(t[['structure','ref_pae_min_AbAg','ref_ipsae_max','ref_iptm_H_Ag','ref_iptm']].sort_values('ref_pae_min_AbAg').to_string(index=False))"
```

### Decision tree

**Several natives dock** (`ref_pae_min_AbAg` < ~10 Å and `ref_ipsae_max` > ~0.3):
the backend works and 7DK2 is a hard target. Adopt a per-target gate — run the
full three-arm pipeline only where the native control passes, and report ipSAE
only for those targets. State the gate explicitly in the paper; it is a
defensible protocol, not a fudge.

**No natives dock**: single-sequence ESMFold2 cannot do this on your data, and
no further tuning on 7DK2 will help. Options, in order of expected value:

1. **`--num-diffusion-samples 5`** — cheapest untried lever, already exposed.
   The best sample by ipTM is kept. Try this on 2–3 targets before anything
   larger; it costs GPU time linearly.
2. **MSA conditioning** — the full checkpoint is MSA-capable and the strongest
   reported ESMFold2 results use MSAs. `mmseqs2` is already in `env.yaml`.
   **Check the API first**, it is not in the README:
   ```
   python -c "import inspect;from esm.models.esmfold2 import ProteinInput,ESMFold2InputBuilder,StructurePredictionInput as S;print('ProteinInput',inspect.signature(ProteinInput));print('SPI',inspect.signature(S));print('fold',inspect.signature(ESMFold2InputBuilder.fold))"
   ```
   Temper expectations: MSAs mainly improve each chain's own fold, which is
   already good, and antibody–antigen pairs have no inter-chain coevolution.
3. **A different backend** — Boltz-2 or AF3. `third_party/ipsae/ipsae.py`
   reads Boltz output natively (`pae_*.npz` + `.cif`), so only
   `diffab/tools/fold/` needs a new module; `diffab/tools/eval/ipsae.py` is
   backend-agnostic apart from the import.
4. **Drop the ipSAE line.** Report that no available predictor docks these
   complexes confidently enough for interface confidence to discriminate. That
   is a real finding, and it is consistent with published Ab–Ag failure rates
   above 50%.

## 6. Gotchas that already cost time

- **`results/` is gitignored** (`/results*`). Code syncs over git; structures do
  not. Decide which machine owns `results/` — splitting it also splits the
  `esmfold2/` fold cache.
- **`ipsae.py` mangles paths containing `.pdb`.** It builds output filenames
  with `pdb_path.replace(".pdb","")`, a *global* replace, and every DiffAb run
  directory is named `<target>.pdb_<timestamp>`. `run_ipsae` therefore scores a
  copy in a scratch directory and copies the reports back. Do not "simplify" it
  to run in place.
- **The fold cache key includes the folding config**, not just the sequences
  (`config_tag` in `ipsae.py`). Changing checkpoint or sampling and re-running
  used to silently reuse the old structures — which would have made the §4
  comparison impossible. Keep it that way.
- **Design selection is keyed by `(cdr, filename)`.** The same `0006_rosetta.pdb`
  exists in all six CDR directories; matching on basename alone selects every
  CDR's copy.
- **`ddG_int` and `ddG_bind` disagree at the top.** Rank correlation is only
  0.52–0.78 per CDR and the top-5 shortlists barely overlap. `--rank-by
  ddG_int,ddG_bind` folds the union. `ddG_bind` is the better physical quantity
  (see `binding.py`'s docstring); `ddG_int` is deterministic but overstates
  affinity.
- **H_CDR3 is the worst CDR to judge biology from** — median `ddG_bind` +26,
  only 4 of 100 designs negative. Fine for plumbing tests, not for signal.
- **Multi-line shell commands break in this user's terminal.** Trailing
  whitespace after `\` kills the continuation. Give single-line commands.

## 7. Unverified

- `--num-diffusion-samples > 1` has never been run. The best-of-N selection
  logic is unit-tested against a mock but not against real multi-sample output;
  confirm the leading sample axis appears where expected.
- `--negative mismatch` (swap in another target's antigen) has never been run.
- `--dtype bf16` has never been run.
- `sc_rmsd` produced 7.07 Å on the one real design scored. Plausible, but it has
  not been checked against an independently computed value.
- ESMFold2's training cutoff is September 2021, which covers 7DK2 and most of
  the SAbDab test set. The `ref_` arm is therefore partly memorised and is an
  optimistic ceiling. This matters for how the native control is described in
  writing, not for whether the gate in §5 is valid.
