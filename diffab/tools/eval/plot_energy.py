"""Distribution of interface energy per CDR, against the native reference.

Reads `summary.csv` (written by `diffab.tools.eval.run`) and plots `dG_gen` for
each CDR with the native `dG_ref` alongside it.

`dG_ref` is not a single number: `energy.py` recomputes it for every design, and
`InterfaceAnalyzerMover(pack_separated=True)` repacks the separated state
stochastically. In practice the 100 values per CDR land on only 6-16 distinct
outcomes and are left-skewed -- the mode sits near the maximum with rare low
excursions -- so a mean and a min-max band misrepresent it. It is drawn as its own
distribution instead, on a strip sharing the x-axis.

Single run:

    python -m diffab.tools.eval.plot_energy --root ./results

Overlay several runs in one figure:

    python -m diffab.tools.eval.plot_energy --runs <runA> <runB> --out <runB>

Runs are never pooled. Each design distribution keeps its own reference,
colour-matched: two runs relaxed from the same native can still disagree on
`dG_ref` (L_CDR3 differs by 3.5 REU between the 7DK2 runs, ~6x the within-run
spread), and pooling would hide that inside a wider band.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffab.tools.eval.base import evaluation_dir

CDR_ORDER = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']
FILENAME = 'dG_distribution.png'
COMPARE_FILENAME = 'dG_distribution_compare.png'
VIOLIN_FILENAME = 'dG_violin.png'
VIOLIN_COMPARE_FILENAME = 'dG_violin_compare.png'
COLOURS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

REQUIRED = {'cdr', 'dG_gen', 'dG_ref'}


def find_summaries(root):
    """Every `summary.csv` under `root`, keyed by the run directory."""
    hits = {}
    for path in sorted(glob.glob(os.path.join(root, '**', 'summary.csv'), recursive=True)):
        # <run>/evaluation/summary.csv -> <run>
        hits[os.path.dirname(os.path.dirname(path))] = path
    return hits


def load_run(directory):
    """`summary.csv` for one run directory, or None with a reason printed."""
    path = os.path.join(directory, 'evaluation', 'summary.csv')
    if not os.path.exists(path):
        print(f'  {directory}: no evaluation/summary.csv -- run diffab.tools.eval.run first')
        return None
    table = pd.read_csv(path)
    missing = REQUIRED - set(table.columns)
    if missing:
        print(f'  {path}: missing {sorted(missing)} (run with --no_energy?)')
        return None
    return table


def order_cdrs(tables):
    seen = set()
    for t in tables:
        seen |= set(t['cdr'])
    return [c for c in CDR_ORDER if c in seen] + sorted(seen - set(CDR_ORDER))


def print_stats(table, label=None):
    if label:
        print(f'\n--- {label} ---')
    print(f"{'cdr':8}{'n':>5}{'dG_gen mean':>13}{'median':>9}{'min':>9}{'max':>10}"
          f"{'dG_ref':>9}{'ref sd':>8}{'ref uniq':>10}{'% < ref':>9}")
    for cdr in order_cdrs([table]):
        sub = table[table['cdr'] == cdr]
        gen, ref = sub['dG_gen'].to_numpy(float), sub['dG_ref'].to_numpy(float)
        print(f'{cdr:8}{len(gen):5d}{gen.mean():13.2f}{np.median(gen):9.2f}'
              f'{gen.min():9.2f}{gen.max():10.2f}{ref.mean():9.2f}{ref.std():8.2f}'
              f'{len(np.unique(np.round(ref, 3))):10d}'
              f'{100 * np.mean(gen < ref.mean()):9.0f}')


def _spikes(ax, values, colour, span, label=None):
    """Reference values as discrete spikes, height proportional to frequency.

    The values are highly quantised, so binning them would smear the modes
    together; each distinct value gets its own bar instead.
    """
    uniq, counts = np.unique(np.round(values, 3), return_counts=True)
    width = max(span / 200.0, 1e-6)
    ax.bar(uniq, counts, width=width, color=colour, alpha=0.9, label=label)


def make_figure(runs, clip=None, bins=40, title=''):
    """`runs` is a list of (label, table). One cell per CDR: designs above,
    reference strip below, sharing the x-axis."""
    cdrs = order_cdrs([t for _, t in runs])
    ncol = 3
    nblock = int(np.ceil(len(cdrs) / ncol))

    fig = plt.figure(figsize=(5.8 * ncol, 4.1 * nblock))
    # Outer grid is generously spaced so one cell's strip never collides with the
    # next row's title; each cell is then split tightly into main + strip.
    outer = fig.add_gridspec(nblock, ncol, hspace=0.42, wspace=0.24)

    for i, cdr in enumerate(cdrs):
        block, col = divmod(i, ncol)
        inner = outer[block, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        main = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1], sharex=main)

        per_run = []
        for (label, table), colour in zip(runs, COLOURS):
            sub = table[table['cdr'] == cdr]
            if sub.empty:
                continue
            per_run.append((label, colour,
                            sub['dG_gen'].to_numpy(float),
                            sub['dG_ref'].to_numpy(float)))
        if not per_run:
            main.axis('off'); strip.axis('off')
            continue

        # Shared x-range over every run; the clip affects drawing only.
        pooled = np.concatenate([g for _, _, g, _ in per_run])
        hi = np.percentile(pooled, clip) if clip else pooled.max()
        lo = min(pooled.min(), min(r.min() for _, _, _, r in per_run))
        hi = max(hi, max(r.max() for _, _, _, r in per_run))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)

        for label, colour, gen, ref in per_run:
            pct = 100 * np.mean(gen < ref.mean())
            main.hist(gen, bins=edges, histtype='step', lw=1.8, color=colour,
                      label=f'{label}   {pct:.0f}% < ref   med {np.median(gen):.1f}')
            main.axvline(np.median(gen), color=colour, ls='--', lw=1.2)
            _spikes(strip, ref, colour, hi - lo, label=f'ref {ref.mean():.1f}')

        n_hidden = int((pooled > hi).sum())
        if n_hidden:
            main.text(0.985, 0.60, f'{n_hidden} design(s) > {hi:.0f} off-scale',
                      transform=main.transAxes, ha='right', va='top',
                      fontsize=6.5, color='0.35')

        main.set_title(cdr, fontsize=11)
        main.set_ylabel('designs')
        main.legend(fontsize=7, loc='upper right')
        main.tick_params(labelbottom=False)
        main.set_xlim(lo, hi)

        strip.set_ylabel('ref', fontsize=8)
        strip.legend(fontsize=6.5, loc='upper left')
        strip.tick_params(labelsize=8)
        strip.set_xlabel('dG_separated (REU)  —  lower is better', fontsize=8.5)

    fig.suptitle(title, fontsize=12)
    return fig


def make_violin_figure(runs, clip=None, title='', ylim=None):
    """All CDRs on one shared dG axis, one violin per (CDR, run).

    A single axis is the point: every CDR is read against the same energy scale,
    so the panels cannot mislead by each rescaling to its own range.

    The native reference is drawn per violin as a solid line at its median, with
    dots at each distinct observed value sized by how often it occurred -- the
    reference takes only a handful of values, so a mean and a whisker would imply
    a smooth spread it does not have.
    """
    cdrs = order_cdrs([t for _, t in runs])
    nrun = len(runs)
    width = 0.8 / nrun

    fig, ax = plt.subplots(figsize=(max(9.0, 2.1 * len(cdrs)), 6.4))

    pooled = np.concatenate([t['dG_gen'].to_numpy(float) for _, t in runs])
    cap = np.percentile(pooled, clip) if clip else None

    handles, hidden = [], 0
    for j, ((label, table), colour) in enumerate(zip(runs, COLOURS)):
        offset = (j - (nrun - 1) / 2.0) * width
        for i, cdr in enumerate(cdrs):
            sub = table[table['cdr'] == cdr]
            if sub.empty:
                continue
            gen = sub['dG_gen'].to_numpy(float)
            ref = sub['dG_ref'].to_numpy(float)
            shown = gen if cap is None else gen[gen <= cap]
            hidden += len(gen) - len(shown)
            if len(shown) < 2:
                continue

            parts = ax.violinplot([shown], positions=[i + offset],
                                  widths=width * 0.9, showextrema=False,
                                  showmedians=True)
            for body in parts['bodies']:
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(colour)
                body.set_linewidth(1.0)
            parts['cmedians'].set_color(colour)
            parts['cmedians'].set_linewidth(2.0)

            # Native reference for this run/CDR.
            half = width * 0.45
            ax.hlines(np.median(ref), i + offset - half, i + offset + half,
                      color=colour, lw=2.4, zorder=5)
            uniq, counts = np.unique(np.round(ref, 3), return_counts=True)
            ax.scatter(np.full(len(uniq), i + offset), uniq,
                       s=6 + 40 * counts / counts.max(), facecolors='none',
                       edgecolors=colour, linewidths=0.9, zorder=6)

        handles.append(plt.Line2D([], [], color=colour, lw=6, alpha=0.55,
                                  label=f'{label}  (designs)'))
    handles.append(plt.Line2D([], [], color='0.25', lw=2.4,
                              label='native dG_ref (median)'))
    handles.append(plt.Line2D([], [], color='0.25', lw=0, marker='o',
                              markerfacecolor='none',
                              label='dG_ref values (size = frequency)'))

    ax.set_xticks(range(len(cdrs)))
    ax.set_xticklabels(cdrs)
    ax.set_ylabel('dG_separated (REU)   —  lower is more favourable')
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(handles=handles, fontsize=8.5, loc='upper left')

    subtitle = title
    if hidden:
        subtitle += f'    ({hidden} design(s) above {cap:.0f} REU off-scale)'
    ax.set_title(subtitle, fontsize=12)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--runs', type=str, nargs='+', default=None,
                        help='run directories to overlay in a single figure; '
                             'each must contain evaluation/summary.csv')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='display names for --runs (default: directory basename)')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--bins', type=int, default=40)
    parser.add_argument('--kind', choices=('hist', 'violin'), default='hist',
                        help="'hist' (default) draws one panel per CDR; 'violin' puts "
                             'every CDR on one shared dG axis')
    parser.add_argument('--ylim', type=float, nargs=2, default=None,
                        metavar=('LO', 'HI'),
                        help='explicit dG axis limits for --kind violin')
    parser.add_argument('--clip', type=float, default=99.0,
                        help='hide the design tail above this percentile when drawing; '
                             'printed statistics always use every design')
    args = parser.parse_args()

    if args.runs:
        labels = args.labels or [os.path.basename(p.rstrip('/')) for p in args.runs]
        if len(labels) != len(args.runs):
            parser.error('--labels must match the number of --runs')
        runs = []
        for path, label in zip(args.runs, labels):
            table = load_run(path)
            if table is not None:
                runs.append((label, table))
        if not runs:
            return
        for label, table in runs:
            print_stats(table, label)
        title = '  vs  '.join(l for l, _ in runs)
        if args.kind == 'violin':
            fig = make_violin_figure(runs, clip=args.clip, title=title, ylim=args.ylim)
            name = VIOLIN_COMPARE_FILENAME
        else:
            fig = make_figure(runs, clip=args.clip, bins=args.bins, title=title)
            name = COMPARE_FILENAME
        out_base = args.out or args.runs[-1]
        out = os.path.join(evaluation_dir(out_base), name)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'\nWrote {out}')
        return

    summaries = find_summaries(args.root)
    if not summaries:
        print(f'No summary.csv found under {args.root}. Run diffab.tools.eval.run first.')
        return
    for directory, path in summaries.items():
        table = pd.read_csv(path)
        missing = REQUIRED - set(table.columns)
        if missing:
            print(f'Skipping {path}: missing {sorted(missing)}')
            continue
        label = os.path.basename(directory)
        print(f'\n=== {label} ===')
        print_stats(table)
        if args.kind == 'violin':
            fig = make_violin_figure([(label, table)], clip=args.clip,
                                     title=label, ylim=args.ylim)
            name = VIOLIN_FILENAME
        else:
            fig = make_figure([(label, table)], clip=args.clip,
                              bins=args.bins, title=label)
            name = FILENAME
        out = os.path.join(evaluation_dir(args.out or directory), name)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'Wrote {out}')


if __name__ == '__main__':
    main()
