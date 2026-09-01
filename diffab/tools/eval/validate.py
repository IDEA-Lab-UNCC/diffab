"""Standalone runner for the reference-free validity metrics in `geometry`.

Kept separate from `run` because it needs no PyRosetta and no shelve database:
it rescans the result tree every time, so it can be re-run after adding a
metric without clearing `evaluation_db`.

    python -m diffab.tools.eval.validate --root ./results --pfx rosetta
"""

import os
import argparse
import pandas as pd

from diffab.tools.eval.base import TaskScanner
from diffab.tools.eval.geometry import (
    eval_backbone_validity,
    eval_prerelax_validity,
    eval_relax_shift,
    native_relax_shift,
    native_validity,
)

# Columns worth summarising, and how.
SUMMARY_AGG = {
    'pep_frac_viol': ['mean'],
    'pep_n_break': ['mean', 'max'],
    'pep_min': ['mean', 'min'],
    'pep_rmsd_ideal': ['mean'],
    'omega_dev_mean': ['mean'],
    'omega_n_viol': ['mean'],
    'clash_n': ['mean', 'max'],
    'pre_clash_n': ['mean', 'max'],
    'pre_clash_n_severe': ['mean', 'max'],
    'shift_ca_rmsd': ['mean', 'std', 'max'],
    'shift_bb_rmsd': ['mean', 'std', 'max'],
    'shift_bb_max': ['max'],
    'ref_shift_ca_rmsd': ['first'],
    'ref_shift_bb_rmsd': ['first'],
    'ref_pep_frac_viol': ['first'],
    'ref_pep_min': ['first'],
    'ref_pep_rmsd_ideal': ['first'],
    'ref_omega_dev_mean': ['first'],
    'ref_clash_n': ['first'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    tasks = TaskScanner(root=args.root, postfix=args.pfx).scan()
    if not tasks:
        print(f'No structures matching *_{args.pfx}.pdb found under {args.root}.')
        return
    print(f'Found {len(tasks)} structures.')

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

        rows.append(task.to_report_dict())
        if i % 100 == 0:
            print(f'  {i}/{len(tasks)}')

    table = pd.DataFrame(rows)
    out_dir = args.out or args.root
    per_design = os.path.join(out_dir, 'validity_per_design.csv')
    table.to_csv(per_design, index=False, float_format='%.6f')

    present = {k: v for k, v in SUMMARY_AGG.items() if k in table.columns}
    summary = table.groupby(['method', 'structure', 'cdr']).agg(present)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary_path = os.path.join(out_dir, 'validity_summary.csv')
    summary.to_csv(summary_path, float_format='%.4f')

    print(f'\nWrote {per_design}')
    print(f'Wrote {summary_path}\n')
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(summary.to_string())


if __name__ == '__main__':
    main()
