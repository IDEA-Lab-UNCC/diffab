"""Best-batch comparison of two generation runs.

The other runners in this package summarise a run. This one answers the question
that actually decides whether a fine-tuned checkpoint is worth anything:
**are the designs you would pick for the wet lab better?**

Nobody assays 600 designs; they assay the best handful. So a model that lifts
hopeless designs to merely-bad can move every mean and median in the table while
being worth nothing, and a model that sharpens the top ten can look flat on the
same summaries. Means cannot separate those two. Best-of-k can.

Three things this does that a mean cannot:

`selected-k` -- **select by metric A, score on metric B, A != B.** When a model was
    RL-trained on A, ranking by A and reporting A measures the training objective
    and nothing else. Selecting by A and scoring on B asks whether the designs you
    would actually have picked are better by a yardstick the model never saw.

`random-k` -- best-of-k under random selection, drawn **without replacement**,
    which is the literal "if I made k of these, what is my best" question. With
    replacement would bias it optimistically.

`noise floor` -- an effect means nothing against zero; it means something against
    a replicate. Given a second independent scoring of the same structures
    (`--replicates`), every comparison is reported beside the replicate delta, and
    the printed verdict calls an effect real only when it clears it. This is not
    ceremony: `dG_bind` best-of-10 has a replicate floor near 2 REU, so effects
    smaller than that -- several of the ones in this dataset -- are noise.

Sequence metrics cover what energies cannot: a top-10 of near-identical sequences
is one experiment, not ten, and a design carrying a glycosylation sequon or a free
cysteine may not be worth making whatever it scores. Diversity and liability are
properties of a *batch*, so they only exist at this level.

Reads the CSVs the other runners wrote plus the relaxed PDBs. No PyRosetta.

    python -m diffab.tools.eval.compare --runs <baseline> <finetuned> \
        --labels baseline finetuned --out <dir>
"""

import os
import re
import argparse
import itertools
import collections

import numpy as np
import pandas as pd

from diffab.tools.eval.base import TaskScanner, evaluation_dir, run_dir
from diffab.tools.eval.binding import DIRECTION
from diffab.tools.eval import similarity

KEYS = ['cdr', 'filename']
DEFAULT_KS = (1, 5, 10, 25)

# Selectors are what you would rank by in practice; evaluators are what the
# selected batch is then judged on. `dG_int` leads because it is the only
# deterministic energy here (see binding.py's docstring).
DEFAULT_SELECTORS = ('ddG_int', 'ddG_bind')
DEFAULT_EVALUATORS = ('ddG_int', 'ddG_bind', 'sc_value', 'unsat_hbonds',
                      'dG_per_dSASA', 'dSASA_int')

BOOTSTRAP = 2000

# Developability liabilities, counted over the CDR in its framework context (see
# `liabilities`). Every one is a 2-3 residue motif, so a match can straddle the
# CDR boundary and would be missed by scanning the designed residues alone.
LIABILITIES = {
    'n_glyc': r'N[^P][ST]',      # N-glycosylation sequon
    'deamidation': r'N[GSNTH]',  # NG worst, then NS/NN/NT
    'isomerisation': r'D[GSDT]',
    'free_cys': r'C',
    'oxidation': r'[MW]',
}

KD = {  # Kyte-Doolittle, for GRAVY
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
AROMATIC, SMALL = set('FWY'), set('GAS')
POSITIVE, NEGATIVE = set('KR'), set('DE')


# ------------------------------------------------------------------ loading

def load_binding(directory, filename='binding_per_design.csv'):
    path = os.path.join(evaluation_dir(directory, create=False), filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


FLANK = 2   # framework residues kept either side; the longest motif here is 3


def _flanks(model, task):
    """The framework residues abutting the CDR, as (left, right) strings.

    Only the CDR is designed, so these are fixed by the run and identical across
    its designs -- but they are needed to catch motifs that straddle the CDR
    boundary.
    """
    chain = model[task.residue_first[0]]
    residues = [r for r in chain if r.id[0] == ' ']
    lo, hi = tuple(task.residue_first[1:]), tuple(task.residue_last[1:])
    inside = [i for i, r in enumerate(residues) if lo <= (r.id[1], r.id[2]) <= hi]
    if not inside:
        return '', ''
    first, last = inside[0], inside[-1]
    left = similarity.entity_to_seq(residues[max(0, first - FLANK):first])[0]
    right = similarity.entity_to_seq(residues[last + 1:last + 1 + FLANK])[0]
    return left, right


def cdr_sequences(directory, pfx='rosetta'):
    """`{(cdr, filename): seq}`, `{cdr: native_seq}`, `{cdr: (left, right) flanks}`.

    Uses `similarity.extract_reslist`, the same slice `rmsd` and `seqid` take, so
    these sequences are the identical residue set every other table reports on.
    """
    tasks = [t for t in TaskScanner(root=directory, postfix=pfx).scan()
             if run_dir(t) == os.path.normpath(directory)]
    designs, natives, flanks = {}, {}, {}
    for task in tasks:
        try:
            model = task.get_gen_biopython_model()
            reslist = similarity.extract_reslist(
                model, task.residue_first, task.residue_last)
            # entity_to_seq returns (sequence, residue-id mapping); only the
            # sequence is wanted here.
            designs[(task.cdr, os.path.basename(task.in_path))] = \
                similarity.entity_to_seq(reslist)[0]
            if task.cdr not in natives:
                ref = task.get_ref_biopython_model()
                natives[task.cdr] = similarity.entity_to_seq(
                    similarity.extract_reslist(
                        ref, task.residue_first, task.residue_last))[0]
                flanks[task.cdr] = _flanks(model, task)
        except Exception as exc:                                  # noqa: BLE001
            print(f'[WARNING] {task.in_path}: {type(exc).__name__}: {exc}')
    return designs, natives, flanks


# ------------------------------------------------------------- best-of-k

def _oriented(values, metric):
    """Sign-flipped so that *smaller is always better*, whatever the metric."""
    return -np.asarray(values, float) if DIRECTION.get(metric, -1) > 0 \
        else np.asarray(values, float)


def random_k(values, k, rng, n=BOOTSTRAP):
    """Best-of-k under random selection, drawn without replacement.

    Without replacement because the question is "I generated 100 and can afford
    to test k of them" -- you cannot test the same design twice. Sampling with
    replacement makes k effectively smaller and biases the best-of upward.
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.full(n, np.nan)
    k = min(k, len(values))
    idx = np.argsort(rng.random((n, len(values))), axis=1)[:, :k]
    return values[idx].min(axis=1)


def selected_k(frame, selector, evaluator, k):
    """Rank by `selector`, then report `evaluator` over that batch.

    Returns (mean over the batch, best in the batch). Both matter: the mean says
    whether the batch as a whole is better, the best says whether its single best
    member is -- and a model can move one without the other.
    """
    sub = frame.dropna(subset=[selector, evaluator])
    if sub.empty:
        return np.nan, np.nan
    order = np.argsort(_oriented(sub[selector], selector), kind='stable')
    chosen = sub.iloc[order[:min(k, len(sub))]]
    evaluated = _oriented(chosen[evaluator], evaluator)
    # Reported in the metric's own units and sign, not the oriented ones.
    flip = -1.0 if DIRECTION.get(evaluator, -1) > 0 else 1.0
    return float(evaluated.mean() * flip), float(evaluated.min() * flip)


def selected_k_boot(frame, selector, evaluator, ks, rng, n=BOOTSTRAP):
    """Bootstrap `selected_k` over designs: `{k: (mean draws, best draws)}`.

    Resampling happens *before* ranking, so the draws carry both sources of
    uncertainty -- which designs the run happened to produce, and which of them
    the selector happened to rank top. Ranking inside the bootstrap is the whole
    point; resampling the already-chosen batch would hold the selection fixed and
    badly understate the spread, because "the top 10 of 100" is itself a noisy
    thing to be.
    """
    sub = frame.dropna(subset=[selector, evaluator])
    if sub.empty:
        return {k: (np.full(n, np.nan), np.full(n, np.nan)) for k in ks}
    sel = _oriented(sub[selector], selector)
    ev = _oriented(sub[evaluator], evaluator)
    size = len(sel)
    idx = rng.integers(0, size, (n, size))
    # One sort per draw serves every k.
    order = np.argsort(sel[idx], axis=1, kind='stable')
    picked = np.take_along_axis(idx, order, axis=1)
    flip = -1.0 if DIRECTION.get(evaluator, -1) > 0 else 1.0
    out = {}
    for k in ks:
        top = ev[picked[:, :min(k, size)]]
        out[k] = (top.mean(axis=1) * flip, top.min(axis=1) * flip)
    return out


# ---------------------------------------------------------------- sequences

def diversity(seqs):
    """How many genuinely different designs a batch contains."""
    seqs = [s for s in seqs if s]
    if not seqs:
        return {}
    n = len(seqs)
    pairs = list(itertools.combinations(range(n), 2))
    hamming = [sum(a != b for a, b in zip(seqs[i], seqs[j])) for i, j in pairs]
    columns = list(zip(*seqs)) if len(set(map(len, seqs))) == 1 else []
    entropies = []
    for col in columns:
        counts = np.array(list(collections.Counter(col).values()), float)
        p = counts / counts.sum()
        entropies.append(float(-(p * np.log2(p)).sum()))
    # Single-linkage clusters at Hamming <= 2: near-duplicates collapse, so this
    # counts distinct designs rather than distinct strings.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), d in zip(pairs, hamming):
        if d <= 2:
            parent[find(i)] = find(j)
    return {
        'n': n,
        'n_unique': len(set(seqs)),
        'frac_unique': len(set(seqs)) / n,
        'mean_hamming': float(np.mean(hamming)) if hamming else 0.0,
        'mean_entropy': float(np.mean(entropies)) if entropies else np.nan,
        'n_clusters': len({find(i) for i in range(n)}),
    }


def liabilities(seq, flanks=('', '')):
    """Developability motif counts over the CDR in its framework context.

    `flanks` is the (left, right) framework residues abutting the CDR. They are
    included because every motif here spans 2-3 residues and can straddle the
    boundary -- an `N` at the last designed position forms a glycosylation sequon
    with the two framework residues after it, and scanning the CDR alone would
    miss it. Only motifs *involving* a designed residue are counted: matches
    lying entirely inside the framework are subtracted, since they are identical
    for every design and are not the CDR's doing.
    """
    left, right = flanks
    context = left + seq + right
    out = {}
    for name, pattern in LIABILITIES.items():
        total = len(re.findall(pattern, context))
        # Framework-only matches: those wholly inside `left` or wholly inside
        # `right`. Regex findall is non-overlapping, which is what we want here.
        framework = len(re.findall(pattern, left)) + len(re.findall(pattern, right))
        out[name] = total - framework
    return out


def composition(seqs):
    joined = ''.join(seqs)
    if not joined:
        return {}
    total = len(joined)
    return {
        'frac_aromatic': sum(c in AROMATIC for c in joined) / total,
        'frac_small': sum(c in SMALL for c in joined) / total,
        'net_charge': (sum(c in POSITIVE for c in joined)
                       - sum(c in NEGATIVE for c in joined)) / len(seqs),
        'gravy': float(np.mean([KD.get(c, 0.0) for c in joined])),
    }


# -------------------------------------------------------------------- report

def build_random_k(arms, replicates, metrics, ks, rng):
    rows = []
    for cdr in sorted(set().union(*(set(f.cdr) for f in arms.values()))):
        for metric in metrics:
            per_arm, per_rep = {}, {}
            for label, frame in arms.items():
                sub = frame[frame.cdr == cdr]
                if metric in sub.columns:
                    per_arm[label] = random_k(_oriented(sub[metric], metric),
                                              max(ks), rng)
            for label, frame in (replicates or {}).items():
                sub = frame[frame.cdr == cdr]
                if metric in sub.columns:
                    per_rep[label] = sub
            for k in ks:
                vals, noise = {}, {}
                for label, frame in arms.items():
                    sub = frame[frame.cdr == cdr]
                    if metric not in sub.columns:
                        continue
                    draws = random_k(_oriented(sub[metric], metric), k, rng)
                    vals[label] = draws
                    rep = per_rep.get(label)
                    if rep is not None:
                        r = random_k(_oriented(rep[metric], metric), k, rng)
                        noise[label] = float(r.mean() - draws.mean())
                if len(vals) != 2:
                    continue
                (la, a), (lb, b) = vals.items()
                delta = float(b.mean() - a.mean())
                lo, hi = np.percentile(b - a, [2.5, 97.5])
                floor = max(abs(v) for v in noise.values()) if noise else np.nan
                # The verdict is decided on oriented values (smaller = better for
                # every metric); the row is written in the metric's own units and
                # sign, so `sc_value` reads as `sc_value` and not as its negation.
                call = verdict(delta, lo, hi, floor, metric)
                flip = -1.0 if DIRECTION.get(metric, -1) > 0 else 1.0
                lo_n, hi_n = sorted((float(lo) * flip, float(hi) * flip))
                rows.append({
                    'cdr': cdr, 'metric': metric, 'k': k,
                    'better_when': 'higher' if flip < 0 else 'lower',
                    f'{la}': float(a.mean()) * flip, f'{lb}': float(b.mean()) * flip,
                    'delta': delta * flip, 'ci_lo': lo_n, 'ci_hi': hi_n,
                    'noise_floor': floor * flip if floor == floor else np.nan,
                    'verdict': call,
                })
    return pd.DataFrame(rows)


def verdict(delta, lo, hi, floor, metric):
    """`better` / `worse` / `ns` -- and never `better` on a descriptive column.

    Two hurdles, because either alone is misleading: the CI must exclude zero
    (the effect is not sampling noise) *and* the effect must clear the replicate
    floor (it is not the metric's own measurement noise).
    """
    if DIRECTION.get(metric, -1) == 0:
        return 'descriptive'
    if lo <= 0 <= hi:
        return 'ns'
    if floor == floor and abs(delta) <= abs(floor):      # not NaN and inside
        return 'within-noise'
    return 'better' if delta < 0 else 'worse'


def build_selected_k(arms, selectors, evaluators, ks, rng):
    """Point estimates plus bootstrap CIs for every selector/evaluator/k cell.

    `delta_*` follows one convention throughout, whatever the metric: **negative
    means the second arm is better.** Without that, a table mixing `ddG_int`
    (lower better) with `sc_value` (higher better) cannot be scanned.
    """
    labels = list(arms)
    rows = []
    for cdr in sorted(set().union(*(set(f.cdr) for f in arms.values()))):
        for selector in selectors:
            for evaluator in evaluators:
                subs, ok = {}, True
                for label, frame in arms.items():
                    sub = frame[frame.cdr == cdr]
                    if selector not in sub.columns or evaluator not in sub.columns:
                        ok = False
                        break
                    subs[label] = sub
                if not ok:
                    continue
                boots = {label: selected_k_boot(sub, selector, evaluator, ks, rng)
                         for label, sub in subs.items()}
                sign = -1.0 if DIRECTION.get(evaluator, -1) > 0 else 1.0
                for k in ks:
                    row = {'cdr': cdr, 'selector': selector,
                           'evaluator': evaluator, 'k': k,
                           'circular': selector == evaluator,
                           'better_when': 'higher' if sign < 0 else 'lower'}
                    for label in labels:
                        mean, best = selected_k(subs[label], selector, evaluator, k)
                        row[f'{label}_mean'] = mean
                        row[f'{label}_best'] = best
                    for i, stat in enumerate(('mean', 'best')):
                        a_d = boots[labels[0]][k][i]
                        b_d = boots[labels[1]][k][i]
                        delta = (row[f'{labels[1]}_{stat}']
                                 - row[f'{labels[0]}_{stat}']) * sign
                        lo, hi = np.percentile((b_d - a_d) * sign, [2.5, 97.5])
                        row[f'delta_{stat}'] = delta
                        row[f'ci_lo_{stat}'] = float(lo)
                        row[f'ci_hi_{stat}'] = float(hi)
                        row[f'sig_{stat}'] = bool(lo > 0) or bool(hi < 0)
                    rows.append(row)
    return pd.DataFrame(rows)


def build_sequence(arms_seqs, natives, arms_frames, selectors, ks):
    rows = []
    for label, (designs, _, flanks) in arms_seqs.items():
        frame = arms_frames[label]
        for cdr in sorted({c for c, _ in designs}):
            batches = {'all': [s for (c, _), s in designs.items() if c == cdr]}
            sub = frame[frame.cdr == cdr]
            for selector in selectors:
                if selector not in sub.columns:
                    continue
                for k in ks:
                    if k < 5:
                        continue
                    order = np.argsort(_oriented(sub[selector], selector),
                                       kind='stable')
                    picked = sub.iloc[order[:k]]
                    batches[f'top{k}_by_{selector}'] = [
                        designs[(cdr, f)] for f in picked.filename
                        if (cdr, f) in designs]
            native = natives.get(cdr, '')
            for batch, seqs in batches.items():
                if not seqs:
                    continue
                row = {'run': label, 'cdr': cdr, 'batch': batch}
                row.update(diversity(seqs))
                row.update(composition(seqs))
                flank = flanks.get(cdr, ('', ''))
                counts = [liabilities(s, flank) for s in seqs]
                for name in LIABILITIES:
                    per = np.array([c[name] for c in counts], float)
                    row[f'{name}_mean'] = float(per.mean())
                    row[f'{name}_frac'] = float((per > 0).mean())
                if native:
                    nat = liabilities(native, flank)
                    row['native_seqid'] = float(np.mean([
                        sum(a == b for a, b in zip(s, native)) / len(native)
                        for s in seqs if len(s) == len(native)] or [np.nan]))
                    for name in LIABILITIES:
                        row[f'native_{name}'] = nat[name]
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--runs', nargs=2, required=True,
                        help='two run directories: baseline first')
    parser.add_argument('--labels', nargs=2, default=['baseline', 'finetuned'])
    parser.add_argument('--replicates', nargs=2, default=None,
                        help='a second independent binding_per_design CSV per run, '
                             'for the noise floor. Pass "-" to skip one. Defaults '
                             'to binding_per_design_replicate.csv where present.')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--ks', type=str, default=','.join(map(str, DEFAULT_KS)))
    parser.add_argument('--selectors', type=str, default=','.join(DEFAULT_SELECTORS))
    parser.add_argument('--evaluators', type=str, default=','.join(DEFAULT_EVALUATORS))
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--no-sequence', action='store_true',
                        help='skip the sequence metrics (they parse every PDB)')
    args = parser.parse_args()

    ks = tuple(int(x) for x in args.ks.split(',') if x.strip())
    selectors = tuple(x.strip() for x in args.selectors.split(',') if x.strip())
    evaluators = tuple(x.strip() for x in args.evaluators.split(',') if x.strip())
    rng = np.random.default_rng(args.seed)

    arms = {}
    for label, directory in zip(args.labels, args.runs):
        frame = load_binding(directory)
        if frame is None:
            print(f'{directory}: no binding_per_design.csv. Run '
                  f'`python -m diffab.tools.eval.binding --root {directory}` first.')
            return
        arms[label] = frame
        print(f'{label:12s} {len(frame):4d} designs  {directory}')

    replicates = {}
    for i, (label, directory) in enumerate(zip(args.labels, args.runs)):
        if args.replicates:
            path = args.replicates[i]
            if path == '-':
                continue
            frame = pd.read_csv(path)
        else:
            frame = load_binding(directory, 'binding_per_design_replicate.csv')
        if frame is not None:
            replicates[label] = frame
            print(f'{label:12s} replicate found -> noise floor enabled')
    if not replicates:
        print('[WARNING] no replicate scoring found; effects are compared against '
              'zero rather than against measurement noise. For dG_bind that is '
              'not safe -- its best-of-10 floor is ~2 REU. Rescore one run a '
              'second time and pass --replicates.')

    metrics = sorted(set(selectors) | set(evaluators))
    out_dir = evaluation_dir(args.out or args.runs[1])

    random_tbl = build_random_k(arms, replicates, metrics, ks, rng)
    selected_tbl = build_selected_k(arms, selectors, evaluators, ks, rng)

    random_path = os.path.join(out_dir, 'compare_random_k.csv')
    selected_path = os.path.join(out_dir, 'compare_selected_k.csv')
    random_tbl.to_csv(random_path, index=False, float_format='%.4f')
    selected_tbl.to_csv(selected_path, index=False, float_format='%.4f')
    print(f'\nWrote {random_path}')
    print(f'Wrote {selected_path}')

    if not args.no_sequence:
        arms_seqs, natives = {}, {}
        for label, directory in zip(args.labels, args.runs):
            designs, nat, flanks = cdr_sequences(directory, args.pfx)
            arms_seqs[label] = (designs, nat, flanks)
            natives.update({k: v for k, v in nat.items() if k not in natives})
        seq_tbl = build_sequence(arms_seqs, natives, arms, selectors, ks)
        seq_path = os.path.join(out_dir, 'compare_sequence.csv')
        seq_tbl.to_csv(seq_path, index=False, float_format='%.4f')
        print(f'Wrote {seq_path}')

    calls = random_tbl[random_tbl.verdict.isin(('better', 'worse'))]
    print(f'\n{len(calls)} of {len(random_tbl)} random-k comparisons clear both '
          f'the CI and the noise floor:')
    for _, r in calls.sort_values(['verdict', 'metric', 'cdr']).iterrows():
        print(f"  {r.verdict:6s} {r.cdr:7s} {r.metric:14s} k={int(r.k):<3d} "
              f"delta {r.delta:+8.3f}  noise {r.noise_floor:+.3f}")
    if random_tbl.noise_floor.isna().all():
        print('  (no replicate: "within-noise" was never testable)')


if __name__ == '__main__':
    main()
