"""Standalone runner for the reference-free validity metrics in `geometry`.

Kept separate from `run` because it needs no PyRosetta and no shelve database:
it rescans the result tree every time, so it can be re-run after adding a
metric without clearing `evaluation_db`.

    python -m diffab.tools.eval.validate --root ./results --pfx rosetta
"""

import os
import argparse
import pandas as pd

from diffab.tools.eval.base import (
    TaskScanner, add_selection_args, evaluation_dir, filter_tasks, run_dir,
)
from diffab.tools.eval.geometry import (
    eval_backbone_validity,
    eval_prerelax_validity,
    eval_relax_shift,
    native_relax_shift,
    native_validity,
)

# Summary layout: (per-design column, aggregations, matching reference column).
# The reference is a single value repeated across every design in a group, so it
# is taken as-is and named without an aggregation suffix -- it sits immediately
# after the generated column it is the control for.
SUMMARY_SPEC = [
    ('pep_rmsd_ideal',     ['mean'],                'ref_pep_rmsd_ideal'),
    ('pep_frac_viol',      ['mean'],                'ref_pep_frac_viol'),
    ('pep_n_break',        ['mean', 'max'],         None),
    ('pep_min',            ['mean', 'min'],         'ref_pep_min'),
    ('omega_dev_mean',     ['mean'],                'ref_omega_dev_mean'),
    ('omega_n_viol',       ['mean'],                None),
    ('clash_n',            ['mean', 'max'],         'ref_clash_n'),
    ('pre_clash_n',        ['mean', 'max'],         None),
    ('pre_clash_n_severe', ['mean', 'max'],         None),
    ('shift_ca_rmsd',      ['mean', 'std', 'max'],  'ref_shift_ca_rmsd'),
    ('shift_bb_rmsd',      ['mean', 'std', 'max'],  'ref_shift_bb_rmsd'),
    ('shift_bb_max',       ['max'],                 None),
]


def summarise(table):
    grouped = table.groupby(['method', 'structure', 'cdr'])
    columns = {}
    for column, aggs, reference in SUMMARY_SPEC:
        if column not in table.columns:
            continue
        for agg in aggs:
            columns[f'{column}_{agg}'] = grouped[column].agg(agg)
        if reference is not None and reference in table.columns:
            columns[reference] = grouped[reference].first()
    return pd.DataFrame(columns)


def write_tables(table, base, stem='validity'):
    out_dir = evaluation_dir(base)
    per_design = os.path.join(out_dir, f'{stem}_per_design.csv')
    summary_path = os.path.join(out_dir, f'{stem}_summary.csv')
    table.drop(columns=['run_dir']).to_csv(per_design, index=False, float_format='%.6f')
    summary = summarise(table)
    summary.to_csv(summary_path, float_format='%.4f')
    return per_design, summary_path, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--layout', choices=('flat', 'per_run', 'both'), default='per_run',
                        help="where to write: 'per_run' (default) = into each run's "
                             "evaluation/ directory; 'flat' = one combined roll-up under "
                             "--out / --root; 'both' = per-run plus the roll-up")
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
    if len(selected) != len(tasks):
        print(f'Found {len(tasks)} structures, {len(selected)} selected.')
    else:
        print(f'Found {len(selected)} structures.')
    tasks = selected

    native_cache = {}
    rows = []
    for i, task in enumerate(tasks, 1):
        task = eval_backbone_validity(task)
        task = eval_prerelax_validity(task, postfix=args.pfx)
        task = eval_relax_shift(task, postfix=args.pfx)

        # The native CDR through the same relax protocol: the null for shift_*.
        key = (task.structure, task.cdr)
        if key not in native_cache:
            native = native_validity(task)
            native.update(native_relax_shift(task, postfix=args.pfx))
            native_cache[key] = native
        task.scores.update(native_cache[key])

        row = task.to_report_dict()
        row['run_dir'] = run_dir(task)
        rows.append(row)
        if i % 100 == 0:
            print(f'  {i}/{len(tasks)}')

    table = pd.DataFrame(rows)
    print()

    if args.layout in ('per_run', 'both'):
        for directory, subset in table.groupby('run_dir'):
            per_design, summary_path, _ = write_tables(subset, directory)
            print(f'Wrote {per_design}')
            print(f'Wrote {summary_path}')

    if args.layout in ('flat', 'both'):
        out_dir = args.out or args.root
        per_design, summary_path, summary = write_tables(table, out_dir)
        print(f'Wrote {per_design}')
        print(f'Wrote {summary_path}\n')
        with pd.option_context('display.width', 200, 'display.max_columns', 50):
            print(summary.to_string())


if __name__ == '__main__':
    main()
