"""ipSAE: does an independent predictor rebuild this complex from sequence alone?

Every other metric in this suite reads the coordinates DiffAb produced.
`similarity.py` compares them to the native, `binding.py` scores their interface
with Rosetta, `interface.py` counts their contacts. All of them take the pose as
given, so none can tell a design that folds and binds from one that only scores
well under the generator's own assumptions.

This module asks the orthogonal question. It throws the generated coordinates
away, hands the designed *sequences* to ESMFold2, and scores the interface of
the structure that comes back with ipSAE (Dunbrack 2025). ipSAE is used rather
than ipTM because ipTM averages over whole chains: an Fv plus antigen is mostly
non-interface residues, so a genuinely confident epitope contact gets diluted by
several hundred residues that were never going to touch anything. ipSAE
restricts the sum to residue pairs under a PAE cutoff and rescales `d0`
accordingly.

Three arms run per design, because an ipSAE value in isolation means nothing:

    design    the designed H + L + antigen sequences        (no prefix)
    native    the crystal antibody, same antigen            `ref_`
    negative  the designed chains with CDR residues         `neg_`
              shuffled in place, same antigen

`ref_` is the ceiling and `neg_` the floor. If they do not separate, the metric
is blind on that target and nothing can be read from the designs -- check that
before reading anything into a method comparison.

`sc_rmsd` comes along for free: ESMFold2 folds from sequence, so comparing its
CDR to DiffAb's CDR (after superposing on the framework) is the structural half
of self-consistency, which ipSAE alone does not cover.

Two caveats worth keeping in view. ESMFold2's training data ends September 2021,
which covers 7DK2 and most of the SAbDab test set, so `ref_` is contaminated by
memorisation and is an optimistic ceiling rather than a neutral reference. And
folding is expensive, so only the best `--top-n` designs per CDR are scored,
ranked by `binding.py`'s `ddG_int`.

    python -m diffab.tools.eval.ipsae --probe                    # first, on the GPU box
    python -m diffab.tools.eval.ipsae --root ./results --top-n 5
"""

# pyright: reportMissingImports=false
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from diffab.tools.eval.base import (
    EvalTask, TaskScanner, add_selection_args, evaluation_dir, filter_tasks,
    run_dir,
)
from diffab.tools.eval.interface import cdr_ranges, split_chains
from diffab.tools.eval.similarity import entity_to_seq

IPSAE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    'third_party', 'ipsae', 'ipsae.py',
)

# Folded structures live here, under the run's evaluation dir, keyed by a hash
# of the sequences folded. Folding dominates the runtime and interruptions are
# likely, so a rerun must not redo completed work.
FOLD_SUBDIR = 'esmfold2'

ARMS = ('designs', 'native', 'negative')
PREFIX = {'designs': '', 'native': 'ref_', 'negative': 'neg_'}

# Written per arm. `ipsae_max` is the headline: ipSAE is a per-ordered-pair
# score, but the meaningful unit is Fv-versus-antigen, and the heavy chain
# usually carries the interface.
REPORT_ORDER = ('ipsae_max', 'ipsae_H_Ag', 'ipsae_L_Ag', 'pdockq',
                'iptm_H_Ag', 'iptm_L_Ag', 'iptm', 'ptm', 'pae_min_AbAg')


def _cutoff_string(value):
    """ipsae.py's own filename convention: int, zero-padded below 10."""
    text = str(int(value))
    return text if int(value) >= 10 else '0' + text


# ---------------------------------------------------------------- sequences

def chain_sequences(model, ab_chains):
    """Ordered `{chain: sequence}` plus `{chain: [residue id]}` for a complex.

    Antibody chains come first, sorted, then antigen -- the order the sequences
    are submitted in, and therefore the order the PAE matrix is indexed in.
    `entity_to_seq` skips anything without a three-letter code, so waters,
    ions and glycans drop out here rather than corrupting the alignment later.
    """
    antibody, antigen = split_chains(model, ab_chains)
    seqs, mappings = {}, {}
    for chain_id in sorted(antibody) + sorted(antigen):
        seq, mapping = entity_to_seq(model[chain_id])
        if not seq:
            continue
        seqs[chain_id] = seq
        mappings[chain_id] = mapping
    return seqs, mappings


def heavy_light_chains(directory):
    """`{'H': chain_id, 'L': chain_id}` from a run's metadata."""
    roles = {}
    for tag, (first, _) in cdr_ranges(directory).items():
        roles.setdefault(tag.split('_')[0], first[0])
    return roles


def cdr_indices(mapping, residue_first, residue_last):
    """0-based positions of the CDR within a chain's sequence.

    Uses the same `(resseq, icode)` comparison as
    `similarity.extract_reslist`, so the residue set is identical to the one
    every other metric slices.
    """
    lo, hi = tuple(residue_first[1:]), tuple(residue_last[1:])
    return [i for i, rid in enumerate(mapping) if lo <= (rid[1], rid[2]) <= hi]


def shuffle_cdr(seq, indices, seed):
    """Permute the CDR residues in place, leaving the rest of the chain alone.

    Composition and length are preserved, so the negative control differs from
    the design only in the arrangement of the residues that were designed --
    which is exactly the variable under test.
    """
    if len(indices) < 2:
        return seq
    chars = list(seq)
    picked = [chars[i] for i in indices]
    rng = random.Random(seed)
    for _ in range(20):
        rng.shuffle(picked)
        if picked != [chars[i] for i in indices]:
            break
    else:
        return seq             # homopolymer CDR; no distinct permutation exists
    for i, aa in zip(indices, picked):
        chars[i] = aa
    return ''.join(chars)


# ------------------------------------------------------------------- ipSAE

def run_ipsae(json_path, pdb_path, pae_cutoff, dist_cutoff):
    """Invoke the vendored script and read back its chain-pair table.

    Runs on copies in a scratch directory rather than in place, for a reason
    that is not obvious: ipsae.py builds its output paths with
    `pdb_path.replace(".pdb", "")` -- a *global* replace, not a suffix strip.
    Every DiffAb run directory is named `<target>.pdb_<timestamp>`, so scoring
    in place sends the output to `<target>_<timestamp>/`, a directory that does
    not exist, and the script dies on the open(). Copying to a clean name
    sidesteps it without patching the vendored file.

    The `<stem>_<pae>_<dist>.txt` and `_byres.txt` reports are copied back
    beside the structure afterwards, since they carry the per-residue detail
    the CSV summarises away.
    """
    cut = f'{_cutoff_string(pae_cutoff)}_{_cutoff_string(dist_cutoff)}'
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(pdb_path, os.path.join(tmp, 'model.pdb'))
        shutil.copyfile(json_path, os.path.join(tmp, 'model.json'))
        proc = subprocess.run(
            [sys.executable, IPSAE_SCRIPT, 'model.json', 'model.pdb',
             str(pae_cutoff), str(dist_cutoff)],
            cwd=tmp, capture_output=True, text=True,
        )
        out_path = os.path.join(tmp, f'model_{cut}.txt')
        if proc.returncode != 0 or not os.path.exists(out_path):
            detail = (proc.stderr or proc.stdout or '').strip()[-600:]
            raise RuntimeError(
                f'ipsae.py failed on {os.path.basename(pdb_path)} '
                f'(exit {proc.returncode}): {detail}'
            )
        table = pd.read_csv(out_path, sep=r'\s+', skip_blank_lines=True)

        stem = pdb_path[:-4] if pdb_path.endswith('.pdb') else pdb_path
        for suffix in ('.txt', '_byres.txt'):
            produced = os.path.join(tmp, f'model_{cut}{suffix}')
            if os.path.exists(produced):
                shutil.copyfile(produced, f'{stem}_{cut}{suffix}')
    return table


def interchain_pae_min(pae, chains, ab_chains, antigen_chains):
    """Smallest antibody-to-antigen PAE in the prediction, in angstroms.

    ipSAE collapses to exactly 0.0 whenever no residue pair clears the cutoff,
    which makes "the interface is weak" and "the antigen was never docked"
    indistinguishable in the output. This separates them: a value near the ~31 A
    PAE ceiling means the predictor had no idea where the antigen goes, and no
    choice of cutoff will help.
    """
    bounds, start = {}, 0
    for cid, seq in chains.items():
        bounds[cid] = (start, start + len(seq))
        start += len(seq)
    def idx(ids):
        spans = [np.arange(*bounds[c]) for c in ids if c in bounds]
        return np.concatenate(spans) if spans else np.array([], dtype=int)
    ab, ag = idx(ab_chains), idx(antigen_chains)
    if not len(ab) or not len(ag):
        return np.nan
    return float(np.asarray(pae)[np.ix_(ab, ag)].min())


def pair_iptm_scores(pair_iptm, chain_order, roles, antigen_chains):
    """Per-chain-pair ipTM for the Fv-versus-antigen pairs.

    ESMFold2 reports a (n_chains, n_chains) ipTM matrix indexed in chain
    submission order. ipsae.py's AF2 path can only carry one scalar ipTM for the
    whole complex, so its `ipTM_af` column is the same number on every row --
    useless for comparing H-vs-antigen against L-vs-antigen. This reads the
    matrix directly, giving an independent per-interface confidence to set
    beside ipSAE.
    """
    if pair_iptm is None:
        return {}
    matrix = np.asarray(pair_iptm)
    index = {c: i for i, c in enumerate(chain_order)}
    antigen = [c for c in antigen_chains if c in index]

    def best(ab_chain):
        if ab_chain is None or ab_chain not in index or not antigen:
            return np.nan
        i = index[ab_chain]
        return float(max(matrix[i, index[a]] for a in antigen))

    return {'iptm_H_Ag': best(roles.get('H')), 'iptm_L_Ag': best(roles.get('L'))}


def aggregate_pairs(table, roles, antigen_chains):
    """Chain-pair rows -> the Fv-versus-antigen columns.

    ipsae.py emits an `asym` row per ordered pair and a `max` row per unordered
    pair; the `max` rows are the symmetric score the paper reports, so those are
    what get read. Where an antigen has several chains, the best-scoring one
    wins -- a design only has to bind somewhere on the target.
    """
    rows = table[table['Type'] == 'max'] if 'Type' in table.columns else table
    antigen = set(antigen_chains)

    def best(ab_chain):
        if ab_chain is None:
            return np.nan
        hits = rows[
            ((rows['Chn1'] == ab_chain) & (rows['Chn2'].isin(antigen)))
            | ((rows['Chn2'] == ab_chain) & (rows['Chn1'].isin(antigen)))
        ]
        return float(hits['ipSAE'].max()) if len(hits) else np.nan

    scores = {
        'ipsae_H_Ag': best(roles.get('H')),
        'ipsae_L_Ag': best(roles.get('L')),
    }
    scores['ipsae_max'] = float(np.nanmax([
        v for v in scores.values() if not np.isnan(v)
    ] or [np.nan]))
    if len(rows):
        scores['pdockq'] = float(rows['pDockQ'].max())
    return scores


# ------------------------------------------------------------------ scRMSD

def sc_rmsd(design_pdb, pred_pdb, mappings, ab_chains, cdr_chain, indices):
    """CDR CA-RMSD between DiffAb's design and ESMFold2's re-prediction.

    The predicted complex sits in an arbitrary frame, so the two are first
    superposed on the antibody framework -- every antibody CA *except* the CDR
    under test. Correspondence is exact rather than aligned: predicted residue
    `i+1` of a chain is the `i`-th entry of that chain's `entity_to_seq`
    mapping, because the structure was written renumbered from the same list.
    """
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)
    design = parser.get_structure('d', design_pdb)[0]
    pred = parser.get_structure('p', pred_pdb)[0]
    cdr_positions = set(indices)

    fixed, moving, cdr_fixed, cdr_moving = [], [], [], []
    for chain_id in sorted(set(ab_chains) & set(mappings)):
        for i, res_id in enumerate(mappings[chain_id]):
            try:
                d_atom = design[chain_id][res_id]['CA']
                p_atom = pred[chain_id][(' ', i + 1, ' ')]['CA']
            except KeyError:
                continue
            if chain_id == cdr_chain and i in cdr_positions:
                cdr_fixed.append(d_atom)
                cdr_moving.append(p_atom)
            else:
                fixed.append(d_atom)
                moving.append(p_atom)

    if len(fixed) < 3 or not cdr_fixed:
        return np.nan

    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    sup.apply([a for a in pred.get_atoms()])

    diff = np.array([f.get_coord() for f in cdr_fixed]) - \
        np.array([m.get_coord() for m in cdr_moving])
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


# -------------------------------------------------------------------- jobs

def _digest(chains):
    payload = '|'.join(f'{k}:{v}' for k, v in chains.items())
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def config_tag(opts):
    """Short hash of everything that changes what a fold produces.

    Part of the cache filename, because the sequence hash alone is not a
    sufficient key: switching checkpoint or sampling settings and re-running
    would silently reuse the previous model's structures and report them as new
    results -- the exact way you would wrongly conclude that a setting change
    made no difference.
    """
    payload = '|'.join(str(opts.get(k)) for k in
                       ('checkpoint', 'num_loops', 'num_sampling_steps', 'dtype',
                        'num_diffusion_samples'))
    return hashlib.sha1(payload.encode()).hexdigest()[:6]


def _fold_one(job, opts):
    """Fold one complex (or reuse a cached fold) and score its interface."""
    from diffab.tools.fold.base import FoldResult, FoldTask, check_af2_inputs
    from diffab.tools.fold.esmfold2 import ESMFold2Engine

    stem = os.path.join(job['out_dir'], job['name'])
    pdb_path, json_path = stem + '.pdb', stem + '.json'

    if os.path.exists(pdb_path) and os.path.exists(json_path):
        with open(json_path) as f:
            payload = json.load(f)
        result = FoldResult(
            pdb_path=pdb_path, cif_path=None,
            pae=np.array(payload['pae']), plddt=np.array(payload['plddt']),
            ptm=payload['ptm'], iptm=payload['iptm'],
            chain_order=list(job['chains'].keys()),
            pair_iptm=(np.array(payload['pair_iptm'])
                       if payload.get('pair_iptm') is not None else None),
        )
        check_af2_inputs(pdb_path, result)
    else:
        engine = ESMFold2Engine(
            checkpoint=opts['checkpoint'], device=opts['device'],
            num_loops=opts['num_loops'], num_sampling_steps=opts['num_sampling_steps'],
            dtype=opts.get('dtype'),
            num_diffusion_samples=opts.get('num_diffusion_samples', 1),
        )
        with engine:
            task = FoldTask(name=job['name'], chains=job['chains'],
                            out_dir=job['out_dir'], seed=job['seed'])
            result = engine.fold(task)
        with open(json_path, 'w') as f:
            json.dump(result.json_payload(), f)

    table = run_ipsae(json_path, result.pdb_path,
                      opts['pae_cutoff'], opts['dist_cutoff'])
    scores = aggregate_pairs(table, job['roles'], job['antigen'])
    scores.update(pair_iptm_scores(result.pair_iptm, result.chain_order,
                                   job['roles'], job['antigen']))
    scores['pae_min_AbAg'] = interchain_pae_min(
        result.pae, job['chains'], job['ab_chains'], job['antigen'])
    scores['ptm'] = result.ptm
    scores['iptm'] = result.iptm

    if job.get('design_pdb'):
        scores['sc_rmsd'] = sc_rmsd(
            job['design_pdb'], result.pdb_path, job['mappings'],
            job['ab_chains'], job['cdr_chain'], job['cdr_indices'],
        )
    return scores


def _fold_many(jobs, opts):
    out = []
    for job in jobs:
        try:
            out.append((job['key'], _fold_one(job, opts)))
        except Exception as e:                                  # noqa: BLE001
            print(f'[WARNING] {job["name"]}: {type(e).__name__}: {e}', flush=True)
            out.append((job['key'], {}))
    return out


def run_jobs(jobs, workers, opts, label):
    """Fold and score `jobs`, one process per GPU, or serially when workers <= 1."""
    if not jobs:
        return {}
    if workers <= 1:
        out = {}
        for i, job in enumerate(jobs, 1):
            for key, scores in _fold_many([job], opts):
                out[key] = scores
            print(f'  {label} {i}/{len(jobs)}', flush=True)
        return out

    import ray
    if not ray.is_initialized():
        ray.init(num_gpus=workers, log_to_driver=False)
    # One GPU per worker: ESMFold2 weights are loaded once per process and the
    # cards are not shared, unlike relax's fractional-GPU OpenMM tasks.
    remote = ray.remote(num_gpus=1, num_cpus=1)(_fold_many)
    # Small chunks, as in binding.py. Progress only prints when a chunk
    # finishes, and one chunk per worker would mean silence for hours on a run
    # this slow. Ray reuses worker processes across tasks and `_MODEL` is
    # memoised per process, so extra chunks cost no extra weight loading.
    size = max(1, len(jobs) // (workers * 4))
    chunks = [jobs[i:i + size] for i in range(0, len(jobs), size)]
    futures = [remote.remote(c, opts) for c in chunks]
    out = {}
    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        for key, scores in ray.get(ready[0]):
            out[key] = scores
        print(f'  {label} {len(out)}/{len(jobs)}', flush=True)
    return out


# --------------------------------------------------------------- selection

def select_top(tasks, top_n, rank_by):
    """The best `top_n` designs per (run, CDR), ranked by an existing metric.

    Folding is far too expensive for every design, and the question is about the
    best few anyway. Ranking reuses `binding_per_design.csv` rather than
    inventing a criterion -- and a run without one is skipped loudly, because
    silently folding an arbitrary subset would be worse than folding none.
    """
    selected, by_run = [], {}
    for task in tasks:
        by_run.setdefault(run_dir(task), []).append(task)

    for directory, group in sorted(by_run.items()):
        csv_path = os.path.join(evaluation_dir(directory, create=False),
                                'binding_per_design.csv')
        if not os.path.exists(csv_path):
            print(f'[WARNING] {directory}: no binding_per_design.csv, skipping. '
                  f'Run `python -m diffab.tools.eval.binding --root <root>` first.')
            continue
        table = pd.read_csv(csv_path)
        columns = [c.strip() for c in rank_by.split(',') if c.strip()]
        missing = [c for c in columns if c not in table.columns]
        if missing:
            print(f'[WARNING] {directory}: no column(s) {missing} in '
                  f'binding_per_design.csv, skipping.')
            continue

        # Keyed by (cdr, filename), not filename alone: the same `0006_rosetta.pdb`
        # exists in all six CDR directories, so matching on the basename would
        # select every CDR's copy of each winner.
        #
        # Several columns take the *union* of each one's top-n. ddG_int and
        # ddG_bind rank-correlate only ~0.6-0.7 and disagree sharply at the top,
        # so `--rank-by ddG_int,ddG_bind` folds both shortlists and lets ipSAE
        # arbitrate between them instead of inheriting one metric's blind spots.
        keep = set()
        for column in columns:
            ranked = table.dropna(subset=[column]).sort_values(column)
            for cdr, cdr_rows in ranked.groupby('cdr'):
                keep.update((cdr, f) for f in cdr_rows.head(top_n)['filename'])
        chosen = [t for t in group if (t.cdr, os.path.basename(t.in_path)) in keep]
        print(f'[INFO] {os.path.basename(directory)}: {len(chosen)} of {len(group)} '
              f'designs selected (top {top_n} per CDR by '
              f'{" u ".join(columns)})')
        selected.extend(chosen)
    return selected


def build_jobs(tasks, arms, negative_seed, mismatch, tag=''):
    """One fold job per (arm, design), with natives deduplicated per run."""
    jobs, seen_native = [], set()
    antigen_pool = {}

    for task in tasks:
        model = task.get_gen_biopython_model()
        seqs, mappings = chain_sequences(model, task.ab_chains)
        antibody, antigen = split_chains(model, task.ab_chains)
        antigen = [c for c in antigen if c in seqs]
        antibody = [c for c in antibody if c in seqs]
        roles = heavy_light_chains(run_dir(task))
        out_dir = os.path.join(evaluation_dir(run_dir(task)), FOLD_SUBDIR)
        antigen_pool.setdefault(task.structure, {c: seqs[c] for c in antigen})

        common = {'roles': roles, 'antigen': antigen, 'ab_chains': antibody,
                  'mappings': mappings, 'out_dir': out_dir}

        if 'designs' in arms:
            jobs.append({
                **common, 'key': ('designs', task.in_path),
                'name': f'design_{_digest(seqs)}{tag}', 'chains': dict(seqs), 'seed': 0,
                'design_pdb': task.in_path, 'cdr_chain': task.residue_first[0],
                'cdr_indices': cdr_indices(
                    mappings[task.residue_first[0]],
                    task.residue_first, task.residue_last),
            })

        if 'native' in arms and run_dir(task) not in seen_native:
            # The native sequence does not depend on the CDR being designed, so
            # one fold per run covers all six -- unlike binding.py, which must
            # rescore per CDR because each REF1 is relaxed differently.
            seen_native.add(run_dir(task))
            ref_model = task.get_ref_biopython_model()
            ref_seqs, _ = chain_sequences(ref_model, task.ab_chains)
            jobs.append({
                **common, 'key': ('native', run_dir(task)),
                'name': f'native_{_digest(ref_seqs)}{tag}', 'chains': ref_seqs, 'seed': 0,
                'design_pdb': None,
            })

        if 'negative' in arms:
            neg = dict(seqs)
            if mismatch:
                other = [s for s in sorted(antigen_pool) if s != task.structure]
                if not other:
                    continue
                swap = antigen_pool[other[hash(task.structure) % len(other)]]
                for chain_id in antigen:
                    neg.pop(chain_id, None)
                for i, seq in enumerate(swap.values()):
                    neg[antigen[i] if i < len(antigen) else f'Z{i}'] = seq
            else:
                chain_id = task.residue_first[0]
                neg[chain_id] = shuffle_cdr(
                    seqs[chain_id],
                    cdr_indices(mappings[chain_id],
                                task.residue_first, task.residue_last),
                    negative_seed,
                )
            jobs.append({
                **common, 'key': ('negative', task.in_path),
                'name': f'negative_{_digest(neg)}{tag}', 'chains': neg, 'seed': 0,
                'design_pdb': None,
            })
    return jobs


def report_columns(columns):
    """`ipsae_max`, `ref_ipsae_max`, `neg_ipsae_max`, `ipsae_H_Ag`, ... in order."""
    ordered = []
    for name in REPORT_ORDER:
        for prefix in ('', 'ref_', 'neg_'):
            candidate = prefix + name
            if candidate in columns:
                ordered.append(candidate)
    if 'sc_rmsd' in columns:
        ordered.append('sc_rmsd')
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--layout', choices=('flat', 'per_run', 'both'), default='per_run')
    parser.add_argument('--workers', type=int, default=1,
                        help='folding processes, one GPU each; 1 runs serially')
    parser.add_argument('--top-n', type=int, default=5,
                        help='designs per CDR to fold, best first')
    parser.add_argument('--rank-by', type=str, default='ddG_int',
                        help='column(s) of binding_per_design.csv to rank by, '
                             'ascending; comma-separated takes the union of '
                             'each one\'s top-n')
    parser.add_argument('--arms', type=str, default='designs,native,negative',
                        help=f'comma-separated subset of {ARMS}')
    parser.add_argument('--negative', choices=('shuffle', 'mismatch'), default='shuffle',
                        help='shuffle the CDR in place, or swap in another target')
    parser.add_argument('--negative-seed', type=int, default=2022)
    parser.add_argument('--pae-cutoff', type=float, default=10.0)
    parser.add_argument('--dist-cutoff', type=float, default=10.0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', choices=('bf16', 'fp16', 'fp32'), default=None,
                        help='model precision; bf16 roughly halves the ~24 GB '
                             'fp32 footprint of the ESMC stem')
    parser.add_argument('--num-loops', type=int, default=None)
    parser.add_argument('--num-sampling-steps', type=int, default=None)
    parser.add_argument('--num-diffusion-samples', type=int, default=1,
                        help='diffusion samples per complex; the best by ipTM '
                             'is kept. Costs GPU time linearly')
    parser.add_argument('--probe', action='store_true',
                        help='fold a toy dimer and print the ESMFold2 result fields')
    add_selection_args(parser)
    args = parser.parse_args()

    if args.probe:
        from diffab.tools.fold import esmfold2
        esmfold2.probe(device=args.device, dtype=args.dtype,
                       checkpoint=args.checkpoint or esmfold2.DEFAULT_CHECKPOINT)
        return

    if not os.path.exists(IPSAE_SCRIPT):
        print(f'Vendored ipsae.py not found at {IPSAE_SCRIPT}.')
        return

    from diffab.tools.fold import esmfold2
    opts = {
        'checkpoint': args.checkpoint or esmfold2.DEFAULT_CHECKPOINT,
        'device': args.device,
        'dtype': args.dtype,
        'num_loops': args.num_loops or esmfold2.DEFAULT_NUM_LOOPS,
        'num_sampling_steps': args.num_sampling_steps or esmfold2.DEFAULT_SAMPLING_STEPS,
        'num_diffusion_samples': args.num_diffusion_samples,
        'pae_cutoff': args.pae_cutoff,
        'dist_cutoff': args.dist_cutoff,
    }

    arms = tuple(a.strip() for a in args.arms.split(',') if a.strip())
    unknown = set(arms) - set(ARMS)
    if unknown:
        print(f'Unknown arm(s) {sorted(unknown)}; choose from {ARMS}.')
        return

    tasks = TaskScanner(root=args.root, postfix=args.pfx).scan()
    if not tasks:
        print(f'No structures matching *_{args.pfx}.pdb found under {args.root}.')
        return
    selected = filter_tasks(tasks, args)
    if not selected:
        print(f'{len(tasks)} structures found, none matched the selection filters.')
        return

    if {'designs', 'negative'} & set(arms):
        tasks = select_top(selected, args.top_n, args.rank_by)
    else:
        # A native-only sweep has nothing to rank: the native is one fold per
        # run regardless of which design is picked. Requiring
        # binding_per_design.csv here would block exactly the runs a control
        # sweep is for -- the ones not yet scored with PyRosetta.
        seen, tasks = set(), []
        for task in selected:
            if run_dir(task) not in seen:
                seen.add(run_dir(task))
                tasks.append(task)
        print(f'[INFO] native-only: {len(tasks)} run(s), no ranking needed.')
    if not tasks:
        print('Nothing to score.')
        return

    jobs = build_jobs(tasks, arms, args.negative_seed,
                      args.negative == 'mismatch', '_' + config_tag(opts))
    print(f'Folding {len(jobs)} complexes for {len(tasks)} designs '
          f'({", ".join(arms)}), {args.workers} worker(s).')

    results = run_jobs(jobs, args.workers, opts, 'folds')

    rows = []
    for task in tasks:
        scores = {}
        for arm in arms:
            key = ('native', run_dir(task)) if arm == 'native' else (arm, task.in_path)
            for name, value in (results.get(key) or {}).items():
                scores[PREFIX[arm] + name if name != 'sc_rmsd' else name] = value
        if not scores:
            continue
        task.scores.update(scores)
        row = task.to_report_dict()
        row['run_dir'] = run_dir(task)
        row['pae_cutoff'] = args.pae_cutoff
        row['dist_cutoff'] = args.dist_cutoff
        rows.append(row)

    if not rows:
        print('Every fold failed; nothing written.')
        return

    table = pd.DataFrame(rows)
    print()

    def write(subset, base):
        out_dir = evaluation_dir(base)
        cols = report_columns(subset.columns) + ['pae_cutoff', 'dist_cutoff']
        keys = ['method', 'structure', 'cdr']

        per_design = os.path.join(out_dir, 'ipsae_per_design.csv')
        subset[keys + ['filename'] + cols].to_csv(
            per_design, index=False, float_format='%.6f')

        summary_path = os.path.join(out_dir, 'ipsae_summary.csv')
        subset.groupby(keys)[cols].mean().to_csv(summary_path, float_format='%.4f')
        print(f'Wrote {per_design}')
        print(f'Wrote {summary_path}')

    if args.layout in ('per_run', 'both'):
        for directory, subset in table.groupby('run_dir'):
            write(subset, directory)
    if args.layout in ('flat', 'both'):
        write(table, args.out or args.root)

    ceiling = table.get('ref_ipsae_max')
    floor = table.get('neg_ipsae_max')
    if ceiling is not None and floor is not None:
        print(f'\nControls: native {ceiling.mean():.3f}, '
              f'design {table["ipsae_max"].mean():.3f}, '
              f'negative {floor.mean():.3f}')
        if ceiling.mean() - floor.mean() < 0.1:
            print('[WARNING] native and negative controls are not separated; '
                  'ipSAE is not discriminating on this target, so differences '
                  'between designs should not be read as meaningful.')


if __name__ == '__main__':
    main()
