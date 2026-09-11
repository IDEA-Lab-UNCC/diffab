"""One table joining every per-run evaluation file, one row per (structure, CDR).

The four runners each write their own summary and deliberately do not join
them -- they key on `(method, structure, cdr)` but are produced by different
tools with different dependencies. Comparing many runs means reading all of
them, which is what this does: walk `--root` for `evaluation/` directories,
left-merge the four summaries of each on those keys, and concatenate.

Reads only what the other runners have already written; it computes nothing and
needs neither PyRosetta nor Biopython. Re-run it after any of them.

    python -m diffab.tools.eval.aggregate --root ./results/codesign_single
"""

import os
import argparse
import pandas as pd

from diffab.tools.eval.base import EVAL_SUBDIR, _keep, add_selection_args

KEYS = ['method', 'structure', 'cdr']

# summary.csv is per design, so it is the one file that has to be aggregated
# here rather than merely read; the other three are already per (method,
# structure, cdr).
SIMILARITY_COLUMNS = ['rmsd', 'seqid', 'dG_gen', 'dG_ref', 'ddG']

# Selected from validity_summary.csv (31 columns) and terms_summary.csv (17).
# Each `ref_*` control follows the column it controls, as in `SUMMARY_SPEC`
# (validate.py) and `REPORT_ORDER` (rosetta_terms.py).
VALIDITY_COLUMNS = [
    'pep_rmsd_ideal_mean', 'ref_pep_rmsd_ideal',
    'pep_frac_viol_mean', 'ref_pep_frac_viol',
    'pep_n_break_mean',
    'omega_dev_mean_mean', 'ref_omega_dev_mean',
    'clash_n_mean', 'ref_clash_n',
    'pre_clash_n_mean', 'pre_clash_n_severe_mean',
    'shift_ca_rmsd_mean', 'ref_shift_ca_rmsd',
    'shift_bb_rmsd_mean', 'ref_shift_bb_rmsd',
]

TERM_COLUMNS = [
    'term_total_no_ref', 'ref_term_total_no_ref',
    'term_total_per_res', 'ref_term_total_per_res',
    'term_rama_prepro', 'ref_term_rama_prepro',
    'term_fa_rep', 'ref_term_fa_rep',
    'term_p_aa_pp', 'ref_term_p_aa_pp',
    'term_n_res',
]

COLUMN_ORDER = (
    KEYS + ['n_designs'] + SIMILARITY_COLUMNS + VALIDITY_COLUMNS + TERM_COLUMNS
    + ['native_contacts', 'run_dir']
)

FILENAME = 'all_pairs_summary.csv'


def find_evaluation_dirs(root):
    """Every `<run>/evaluation` under `root`, sorted by run directory."""
    found = []
    for parent, dirs, _ in os.walk(root):
        if os.path.basename(parent) == EVAL_SUBDIR:
            dirs[:] = []            # nothing of interest below an evaluation/
            found.append(parent)
    return sorted(found)


def _read(directory, name, warn):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        warn.append(path)
        return None
    table = pd.read_csv(path)
    return table if not table.empty else None


def similarity_rows(directory, warn):
    """summary.csv is one row per design -- average it into one row per CDR."""
    table = _read(directory, 'summary.csv', warn)
    if table is None:
        return None
    columns = [c for c in SIMILARITY_COLUMNS if c in table.columns]
    grouped = table.groupby(KEYS)
    out = grouped[columns].mean().reset_index()
    out.insert(len(KEYS), 'n_designs', grouped.size().reset_index(drop=True))
    return out


def contact_rows(directory, warn):
    """Native contacts per CDR, from the antibody side of interface_summary.csv.

    A CDR absent from that file makes no contact at the 4.5 A cutoff, which the
    merge below turns into 0 rather than a blank.
    """
    table = _read(directory, 'interface_summary.csv', warn)
    if table is None:
        return None
    antibody = table[table['side'] == 'antibody']
    out = antibody[['method', 'structure', 'region', 'n_contact']].copy()
    out = out.rename(columns={'region': 'cdr', 'n_contact': 'native_contacts'})
    # References dedupe by content, so there is normally one row per region;
    # guard against a tree where --ref-pfx produced several.
    return out.groupby(KEYS, as_index=False)['native_contacts'].max()


def aggregate_run(directory, warn):
    """One frame of (method, structure, cdr) rows for a single run."""
    table = similarity_rows(directory, warn)
    if table is None:
        return None

    for name, columns in (('validity_summary.csv', VALIDITY_COLUMNS),
                          ('terms_summary.csv', TERM_COLUMNS)):
        part = _read(directory, name, warn)
        if part is None:
            continue
        keep = KEYS + [c for c in columns if c in part.columns]
        table = table.merge(part[keep], on=KEYS, how='left')

    contacts = contact_rows(directory, warn)
    if contacts is not None:
        table = table.merge(contacts, on=KEYS, how='left')
        table['native_contacts'] = table['native_contacts'].fillna(0).astype(int)

    table['run_dir'] = os.path.dirname(directory)
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--out', type=str, default=None,
                        help=f'output CSV (default: <root>/{FILENAME})')
    add_selection_args(parser)
    args = parser.parse_args()

    directories = find_evaluation_dirs(args.root)
    if not directories:
        print(f'No {EVAL_SUBDIR}/ directories found under {args.root}.')
        return

    warn, tables = [], []
    for directory in directories:
        table = aggregate_run(directory, warn)
        if table is not None:
            tables.append(table)
    if not tables:
        print(f'Found {len(directories)} evaluation directories, none with a summary.csv.')
        return

    table = pd.concat(tables, ignore_index=True)
    # The same substring matching the other runners use, applied to rows rather
    # than to tasks.
    for field in KEYS:
        spec = getattr(args, field, None)
        if spec:
            table = table[table[field].map(lambda v: _keep(v, spec))]
    if table.empty:
        print('Nothing matched the selection filters.')
        return

    columns = [c for c in COLUMN_ORDER if c in table.columns]
    columns += [c for c in table.columns if c not in columns]
    table = table[columns].sort_values(['structure', 'cdr']).reset_index(drop=True)

    out = args.out or os.path.join(args.root, FILENAME)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    table.to_csv(out, index=False, float_format='%.4f')

    for path in warn:
        print(f'Missing (columns left blank): {path}')
    print(f'\n{len(table)} rows from {table["structure"].nunique()} structures '
          f'x {table["cdr"].nunique()} CDRs, {len(columns)} columns.')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
