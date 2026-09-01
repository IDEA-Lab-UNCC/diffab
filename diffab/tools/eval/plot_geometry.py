"""Distribution of designed peptide bond lengths, one panel per CDR.

    python -m diffab.tools.eval.plot_geometry --root ./results --pfx rosetta
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffab.tools.eval.base import (
    TaskScanner, add_selection_args, evaluation_dir, filter_tasks, run_dir,
)
from diffab.tools.eval.geometry import (
    IDEAL_PEP_CN, PEP_TOL, BREAK_TOL, peptide_bond_lengths, extract_reslist,
)

# Engh & Huber report a standard deviation of 0.014 A on the peptide C-N bond;
# 3 sigma is the usual outlier criterion in structure validation.
EH_SIGMA = 0.014

CDR_ORDER = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']

FILENAME = 'peptide_bond_distribution.png'


def collect(tasks):
    """Peptide bond lengths keyed by (run directory, CDR), for designs and native."""
    gen, ref = {}, {}
    for task in tasks:
        key = (run_dir(task), task.cdr)
        reslist = extract_reslist(
            task.get_gen_biopython_model(), task.residue_first, task.residue_last)
        gen.setdefault(key, []).append(peptide_bond_lengths(reslist))
        if key not in ref:
            ref_reslist = extract_reslist(
                task.get_ref_biopython_model(), task.residue_first, task.residue_last)
            ref[key] = peptide_bond_lengths(ref_reslist)
    return {k: np.concatenate(v) for k, v in gen.items()}, ref


def pool(gen, ref, directory=None):
    """Collapse the (run, CDR) keys to CDR, optionally for one run only."""
    g, r = {}, {}
    for (run, cdr), bonds in gen.items():
        if directory is not None and run != directory:
            continue
        g.setdefault(cdr, []).append(bonds)
        r.setdefault(cdr, []).append(ref[(run, cdr)])
    return ({k: np.concatenate(v) for k, v in g.items()},
            {k: np.concatenate(v) for k, v in r.items()})


def print_stats(gen, ref, cdrs):
    print(f"{'cdr':8}{'n':>6}{'mean':>8}{'sd':>8}{'min':>8}{'p1':>8}{'p50':>8}{'p99':>8}"
          f"{'max':>8}{'native mean':>13}{'native sd':>11}")
    for cdr in cdrs:
        d, r = gen[cdr], ref.get(cdr, np.array([np.nan]))
        print(f"{cdr:8}{len(d):6}{d.mean():8.3f}{d.std():8.3f}{d.min():8.3f}"
              f"{np.percentile(d, 1):8.3f}{np.percentile(d, 50):8.3f}"
              f"{np.percentile(d, 99):8.3f}{d.max():8.3f}"
              f"{np.nanmean(r):13.4f}{np.nanstd(r):11.4f}")


def make_figure(gen, ref, cdrs, xmax, title_suffix=''):
    ncol = 3
    nrow = int(np.ceil(len(cdrs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    bins = np.linspace(0.0, xmax, 100)

    for ax, cdr in zip(axes.ravel(), cdrs):
        d = gen[cdr]
        ax.hist(d, bins=bins, color='#4C72B0', alpha=0.85, label=f'designs (n={len(d)})')
        ax.axvspan(IDEAL_PEP_CN - PEP_TOL, IDEAL_PEP_CN + PEP_TOL,
                   color='#55A868', alpha=0.22, label=f'ideal ±{PEP_TOL} Å')
        for sign in (-1, 1):
            ax.axvline(IDEAL_PEP_CN + sign * BREAK_TOL, color='#C44E52', ls=':', lw=1.2)
        ax.axvline(IDEAL_PEP_CN, color='k', ls='--', lw=1.2, label=f'ideal {IDEAL_PEP_CN} Å')
        if cdr in ref:
            ax.axvline(ref[cdr].mean(), color='#DD8452', lw=1.8,
                       label=f'native {ref[cdr].mean():.3f} Å')
        frac = np.mean(np.abs(d - IDEAL_PEP_CN) > PEP_TOL) * 100
        brk = np.mean(np.abs(d - IDEAL_PEP_CN) > BREAK_TOL) * 100
        ax.set_title(f'{cdr}   {frac:.0f}% outside ±{PEP_TOL},  {brk:.0f}% outside ±{BREAK_TOL}',
                     fontsize=10)
        ax.set_xlabel('C(i)–N(i+1) distance (Å)')
        ax.set_ylabel('bonds')
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=7.5, loc='upper left')

    for ax in axes.ravel()[len(cdrs):]:
        ax.axis('off')

    fig.suptitle('Designed CDR peptide bond length distribution%s '
                 '(dotted red = ±%.2f Å "chain broken" threshold)'
                 % (title_suffix, BREAK_TOL), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def order(gen):
    return [c for c in CDR_ORDER if c in gen] + [c for c in gen if c not in CDR_ORDER]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--layout', choices=('flat', 'per_run', 'both'), default='per_run')
    parser.add_argument('--xmax', type=float, default=3.1)
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
    print(f'Found {len(tasks)} structures, {len(selected)} selected.')

    gen, ref = collect(selected)

    if args.layout in ('per_run', 'both'):
        for directory in sorted({run for run, _ in gen}):
            g, r = pool(gen, ref, directory)
            cdrs = order(g)
            fig = make_figure(g, r, cdrs, args.xmax,
                              title_suffix=f'\n{os.path.basename(directory)}')
            out = os.path.join(evaluation_dir(directory), FILENAME)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'Wrote {out}')

    g, r = pool(gen, ref)
    cdrs = order(g)
    print()
    print_stats(g, r, cdrs)

    if args.layout in ('flat', 'both'):
        fig = make_figure(g, r, cdrs, args.xmax)
        out = os.path.join(evaluation_dir(args.out or args.root), FILENAME)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'\nWrote {out}')

    # Sensitivity of the violation rate to the tolerance, pooled over all CDRs.
    allbonds = np.concatenate([g[c] for c in cdrs])
    print(f'\nPooled n={len(allbonds)}; Engh & Huber sigma = {EH_SIGMA} A')
    print(f"{'tolerance':>12}{'= n sigma':>11}{'% violating':>14}")
    for tol in (0.014, 0.028, 0.042, 0.05, 0.10, 0.20, 0.30):
        pct = np.mean(np.abs(allbonds - IDEAL_PEP_CN) > tol) * 100
        print(f'{tol:12.3f}{tol / EH_SIGMA:11.1f}{pct:14.1f}')


if __name__ == '__main__':
    main()
