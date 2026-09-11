"""Interface energies that separate binding from packing strain.

`summary.csv`'s `dG_gen` / `dG_ref` / `ddG` come from `InterfaceAnalyzerMover`
with `pack_separated=True` and `pack_input` left at its `False` default. That is
asymmetric: the *unbound* state is allowed to repack its side chains before
being scored, the *bound* state is not. So the quantity reported is

    dG_separated = interaction energy + R

where `R` is whatever strain side-chain repacking can remove -- mostly `fa_dun`
(improbable rotamers) and `fa_rep` (overlapping atoms). Because DiffAb relaxes
only a ~10 A shell around one CDR, the other ~250 residues keep raw input side
chains, and their strain never cancels. Measured on this tree, `R` runs 79-438
REU while the interaction energy itself spans only -21 to -45, so `dG_gen` is
dominated by relaxation quality rather than by binding.

This module records two complementary numbers instead.

`dG_int` -- packing off on both sides. The separated pose is a rigid-body
translation of the bound one, so every one-body and intra-chain term is bit
identical and cancels to *exactly* zero (verified: 11 of 19 `ref2015` terms,
including all 972 REU of `fa_dun`). Only the eight genuine inter-chain terms
survive. Deterministic -- no packer runs, so no RNG is consumed -- and immune to
the `R` artefact. It is an interaction energy, not a free energy: the unbound
state stays frozen in its bound conformation, so it overstates affinity.

`dG_bind` -- packing on both sides. Both states get the same side-chain
optimisation, so strain cancels approximately rather than exactly, but the
unbound state is now allowed the reorganisation that really does accompany
unbinding. Closer to a binding free energy, but **stochastic, and much noisier
than one sample per design can hide**. Measured by scoring the same 600 relaxed
7DK2 structures twice: median run-to-run |difference| 0.60 REU, p90 3.0, p99 8.7,
with rare blow-ups past 100 REU on high-`fa_rep` structures where repacking is
unstable. (An earlier note here claimed sd ~0.25 REU; that is wrong.)

This matters most for *best-of-k* statistics, which take a minimum and so
preferentially select whichever design got a lucky repack: the replicate noise
floor on best-of-10 `ddG_bind` reaches 1.9 REU, and 12 REU at k=1. Raise
`--design-repeats` to average it down, and always compare an effect against a
replicate rather than against zero. `dG_int` has no such problem -- no packer
runs, so it is deterministic (replicate difference exactly 0.00).

`fa_rep_bound` -- weighted all-atom repulsion of the bound complex. Not a
binding term at all; it is recorded because it is what actually drives the
spread in `dG_gen` (Spearman 0.72 within H_CDR3), and because the backbone-only
`clash_n` in `validity_per_design.csv` is blind to it -- side-chain overlap is
invisible to a backbone clash cutoff.

Alongside those, the interface descriptors `InterfaceAnalyzerMover` computes on
the very same call and this module used to discard: `sc_value` (shape
complementarity), `dSASA_int` / `dSASA_hphobic` / `dSASA_polar` (buried area),
`unsat_hbonds` (Rosetta's `delta_unsatHbonds`, renamed to free the `delta_`
prefix), `hbonds_int`, `hbond_E_fraction`, `nres_int`, and `dG_per_dSASA`.

They are worth having because they are **not energies**. `sc_value` grades the
geometric fit of the two surfaces, so a design that lowers its score by shedding
bulk -- shrinking side chains until `fa_rep` falls -- cannot raise it; real
antibody-antigen interfaces sit at 0.65-0.75. `dG_per_dSASA` normalises the
energy by the area it buries, so burying more surface stops being a free win.
`unsat_hbonds` is the standard "is this interface actually satisfiable" filter.
`DIRECTION` records which way each reads, including the three that are
descriptive rather than graded.

Nothing here touches `summary.csv`; output goes to `binding_per_design.csv` and
`binding_summary.csv` beside the other evaluation artefacts.

    python -m diffab.tools.eval.binding --root ./results --pfx rosetta
"""

# pyright: reportMissingImports=false
import os
import argparse
import pandas as pd

from diffab.tools.eval.base import (
    EvalTask, TaskScanner, add_selection_args, evaluation_dir, filter_tasks,
    run_dir,
)

# `dG_bind` repacks, so it varies run to run. Designs get one sample; the
# reference is shared by every design of its CDR and is cheap to average, so it
# gets several.
REF_REPEATS = 3

# Columns written, each design value immediately followed by its native control
# and then the difference. Negative deltas mean the design scores better than
# the native -- lower is better for all three.
REPORT_ORDER = (
    'dG_int',           # interaction energy, deterministic, strain-free
    'dG_bind',          # both states repacked; closer to a real dG
    'fa_rep_bound',     # all-atom steric strain of the complex
    'sc_value',         # shape complementarity -- geometry, not energy
    'dSASA_int',        # buried interface area, A^2
    'dSASA_hphobic',
    'dSASA_polar',
    'unsat_hbonds',     # buried unsatisfied H-bonds
    'hbonds_int',       # H-bonds across the interface
    'hbond_E_fraction',
    'nres_int',         # residues in the interface
    'dG_per_dSASA',     # energy per unit buried area
)

DELTA_NAMES = {
    'dG_int': 'ddG_int',
    'dG_bind': 'ddG_bind',
    'fa_rep_bound': 'delta_fa_rep_bound',
    **{name: 'delta_' + name for name in REPORT_ORDER[3:]},
}

# `InterfaceAnalyzerMover` field -> column name. The mover computes all of these
# on the same call that yields `dG_int`, so they were already being paid for and
# thrown away; taking them from the no-repack call also makes them deterministic,
# unlike anything read off a `pack_*` call.
#
# Renamed: Rosetta's `delta_unsatHbonds` is a delta against the separated state,
# not against the native, so keeping the name would collide with this module's
# `delta_` prefix (which always means design minus native).
DESCRIPTORS = {
    'sc_value': 'sc_value',
    'dSASA_int': 'dSASA_int',
    'dSASA_hphobic': 'dSASA_hphobic',
    'dSASA_polar': 'dSASA_polar',
    'delta_unsatHbonds': 'unsat_hbonds',
    'hbonds_int': 'hbonds_int',
    'hbond_E_fraction': 'hbond_E_fraction',
    'nres_int': 'nres_int',
    'dG_separated/dSASAx100': 'dG_per_dSASA',
}

# Which way each column reads: -1 lower is better, +1 higher is better, 0 no
# preferred direction. `compare.py` orients every table by this and refuses to
# call a 0 column better or worse -- `dSASA_polar` and `nres_int` describe the
# interface's composition and size, they do not grade it.
DIRECTION = {
    'dG_int': -1, 'dG_bind': -1, 'fa_rep_bound': -1,
    'sc_value': +1, 'dSASA_int': +1, 'hbonds_int': +1, 'hbond_E_fraction': +1,
    'unsat_hbonds': -1, 'dG_per_dSASA': -1,
    'dSASA_hphobic': 0, 'dSASA_polar': 0, 'nres_int': 0,
}

_INITIALISED = False
_SCOREFXN = None


def _pyrosetta():
    """Import and initialise PyRosetta on first use.

    Deferred so that importing this module does not require PyRosetta -- and so
    that each Ray worker initialises in its own process.
    """
    global _INITIALISED, _SCOREFXN
    import pyrosetta
    if not _INITIALISED:
        pyrosetta.init(' '.join([
            '-mute', 'all',
            '-use_input_sc',
            '-ignore_unrecognized_res',
            '-ignore_zero_occupancy', 'false',
            '-load_PDB_components', 'false',
            '-no_fconfig',
        ]))
        _SCOREFXN = pyrosetta.create_score_function('ref2015')
        _INITIALISED = True
    return pyrosetta, _SCOREFXN


def _interface_analyzer(interface):
    """InterfaceAnalyzerMover for a chain-pair spec such as 'AB_C'.

    Older PyRosetta took the spec in the constructor; 2025+ builds dropped that
    overload and expose it only through set_interface().
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    try:
        return InterfaceAnalyzerMover(interface)
    except TypeError:
        mover = InterfaceAnalyzerMover()
        mover.set_interface(interface)
        return mover


def _pose_scores(pose):
    """Every mover output, from whichever score container this build exposes.

    `Pose.scores` is deprecated in favour of `Pose.cache` in 2026 builds, so
    `cache` is tried first and `scores` kept as the fallback.
    """
    try:
        return dict(pose.cache)
    except AttributeError:
        return dict(pose.scores)


def _apply(pdb_path, interface, pack_separated, pack_input):
    """All `InterfaceAnalyzerMover` outputs under one choice of the two flags.

    The pose is reloaded per call: `pack_input=True` repacks in place, so a pose
    cannot be reused between settings.
    """
    pyrosetta, _ = _pyrosetta()
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = _interface_analyzer(interface)
    mover.set_pack_separated(pack_separated)
    mover.set_pack_input(pack_input)
    mover.apply(pose)
    return _pose_scores(pose)


def fa_rep_bound(pdb_path):
    """Weighted `fa_rep` of the complex as given, no repacking."""
    pyrosetta, scorefxn = _pyrosetta()
    from pyrosetta.rosetta.core.scoring import ScoreType
    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxn(pose)
    total = pose.energies().total_energies()[ScoreType.fa_rep]
    return float(total * scorefxn.get_weight(ScoreType.fa_rep))


def score_structure(pdb_path, interface, repeats=1):
    """The recorded quantities for one PDB: energies plus interface descriptors."""
    unpacked = _apply(pdb_path, interface, False, False)
    binds = [_apply(pdb_path, interface, True, True)['dG_separated']
             for _ in range(repeats)]
    out = {
        'dG_int': float(unpacked['dG_separated']),
        'dG_bind': sum(float(b) for b in binds) / len(binds),
        'fa_rep_bound': fa_rep_bound(pdb_path),
    }
    # Descriptors come from the unpacked call so they describe the pose as
    # relaxed, and inherit `dG_int`'s determinism. A field missing from an older
    # PyRosetta build is skipped rather than written as NaN.
    for field, column in DESCRIPTORS.items():
        if field in unpacked:
            out[column] = float(unpacked[field])
    return out


def interface_spec(task: EvalTask):
    """Chain-pair spec such as 'AB_C', antibody chains first.

    Chain ids are sorted so the spec is reproducible; `ab_chains` arrives from a
    set, whose iteration order is arbitrary.
    """
    model = task.get_gen_biopython_model()
    antibody = sorted(task.ab_chains)
    antigen = sorted(c.id for c in model if c.id not in task.ab_chains)
    return f"{''.join(antibody)}_{''.join(antigen)}"


def eval_binding(task: EvalTask, interface):
    task.scores.update(score_structure(task.in_path, interface))
    return task


def paired_columns(columns):
    """`dG_int`, `ref_dG_int`, `ddG_int`, `dG_bind`, ... in REPORT_ORDER."""
    ordered = []
    for name in REPORT_ORDER:
        for candidate in (name, f'ref_{name}', DELTA_NAMES[name]):
            if candidate in columns:
                ordered.append(candidate)
    return ordered


def _chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _score_many(jobs):
    """[(key, path, interface, repeats)] -> [(key, scores)], one process."""
    return [(key, score_structure(path, iface, reps)) for key, path, iface, reps in jobs]


def run_jobs(jobs, workers, label):
    """Score `jobs` across `workers` processes, or serially when workers <= 1."""
    if not jobs:
        return {}
    if workers <= 1:
        out = {}
        for i, (key, path, iface, reps) in enumerate(jobs, 1):
            out[key] = score_structure(path, iface, reps)
            if i % 25 == 0 or i == len(jobs):
                print(f'  {label} {i}/{len(jobs)}', flush=True)
        return out

    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=workers, log_to_driver=False)
    remote = ray.remote(num_cpus=1)(_score_many)
    # Small chunks keep every worker fed and make progress visible; large enough
    # that PyRosetta's one-off init per worker is amortised.
    size = max(1, len(jobs) // (workers * 4))
    futures = [remote.remote(c) for c in _chunks(jobs, size)]
    out, done_n = {}, 0
    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        for key, scores in ray.get(ready[0]):
            out[key] = scores
        done_n = len(out)
        print(f'  {label} {done_n}/{len(jobs)}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--layout', choices=('flat', 'per_run', 'both'), default='per_run')
    parser.add_argument('--design-repeats', type=int, default=1,
                        help='repacking samples averaged per design for dG_bind. '
                             '1 leaves a noise floor of ~2 REU on best-of-10 '
                             'statistics; raise it when comparing runs on dG_bind. '
                             'Costs one InterfaceAnalyzer pass per extra repeat '
                             'and does not affect the deterministic dG_int.')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                        help='processes to score with; 1 runs serially')
    add_selection_args(parser)
    args = parser.parse_args()

    tasks = TaskScanner(root=args.root, postfix=args.pfx).scan()
    if not tasks:
        print(f'No structures matching *_{args.pfx}.pdb found under {args.root}.')
        return
    selected = filter_tasks(tasks, args)
    if not selected:
        print(f'{len(tasks)} structures found, none matched the selection filters.')
        return
    tasks = selected

    # One interface spec per run: the chain composition is a property of the
    # complex, and reading it costs a PDB parse.
    specs = {}
    for task in tasks:
        directory = run_dir(task)
        if directory not in specs:
            specs[directory] = interface_spec(task)

    # The native is shared by every design of its CDR. `summary.csv` rescores it
    # once per design (1200 times here); once per CDR is enough.
    ref_jobs, seen = [], set()
    for task in tasks:
        key = (run_dir(task), task.cdr)
        if key in seen:
            continue
        seen.add(key)
        ref_jobs.append((key, task.ref_path, specs[run_dir(task)], REF_REPEATS))

    design_jobs = [
        (task.in_path, task.in_path, specs[run_dir(task)], args.design_repeats)
        for task in tasks
    ]

    print(f'Found {len(tasks)} structures, {len(design_jobs)} designs and '
          f'{len(ref_jobs)} references to score, {args.workers} worker(s).')

    refs = run_jobs(ref_jobs, args.workers, 'references')
    designs = run_jobs(design_jobs, args.workers, 'designs')

    rows = []
    for task in tasks:
        scores = designs.get(task.in_path)
        if scores is None:
            continue
        task.scores.update(scores)
        native = refs.get((run_dir(task), task.cdr), {})
        task.scores.update({'ref_' + k: v for k, v in native.items()})
        for name in REPORT_ORDER:
            ref = task.scores.get(f'ref_{name}')
            if ref is not None:
                task.scores[DELTA_NAMES[name]] = task.scores[name] - ref
        row = task.to_report_dict()
        row['run_dir'] = run_dir(task)
        rows.append(row)

    table = pd.DataFrame(rows)
    print()

    def write(subset, base):
        out_dir = evaluation_dir(base)
        cols = paired_columns(subset.columns)
        keys = ['method', 'structure', 'cdr']

        per_design = os.path.join(out_dir, 'binding_per_design.csv')
        subset[keys + ['filename'] + cols].to_csv(
            per_design, index=False, float_format='%.6f')

        summary_path = os.path.join(out_dir, 'binding_summary.csv')
        subset.groupby(keys)[cols].mean().to_csv(summary_path, float_format='%.4f')
        print(f'Wrote {per_design}')
        print(f'Wrote {summary_path}')

    if args.layout in ('per_run', 'both'):
        for directory, subset in table.groupby('run_dir'):
            write(subset, directory)
    if args.layout in ('flat', 'both'):
        write(table, args.out or args.root)


if __name__ == '__main__':
    main()
