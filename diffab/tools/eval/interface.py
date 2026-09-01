"""Which residues actually touch the antigen, by distance.

`ddG` is a single scalar over the whole interface, so it cannot say *where* an
antibody binds: a design can score acceptably while gripping a shifted patch of
the antigen, or while the designed CDR has drifted out of contact and the
framework carries the interaction. Chain assignment comes from the metadata, so
the contact geometry is directly measurable from the output PDBs.

A contact is a pair of residues whose closest **heavy atoms** lie within
`CONTACT_CUTOFF` (4.5 A). See EVALUATION_METRICS.md for why that cutoff, and in
particular why the `CA-CA <= 6.0` convention in `utils/transforms/mask.py` is not
the right one here -- it exists to pick a patch anchor, not to define an
interface, and Cα positions ignore side-chain reach.

The reference is read **unrelaxed** by default (`--ref-pfx ''`, i.e. `REF1.pdb`),
independently of `--pfx`. Every CDR directory holds a byte-identical copy of the
native, so this gives one unambiguous answer per structure. The relaxed
`REF1_<pfx>.pdb` copies are *not* six replicates of one process: each had a
single CDR made flexible, which also drags every downstream residue of that chain
through the fold tree and repacks spatial neighbours, while leaving the rest
untouched. Treating them as replicates measures the relax protocol, not the
antibody.

    python -m diffab.tools.eval.interface --root ./results --reference-only
"""

import os
import copy
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import NeighborSearch, Selection, ShrakeRupley

from diffab.tools.eval.base import (
    TaskScanner, add_selection_args, evaluation_dir, filter_tasks, run_dir,
)
from diffab.tools.eval.geometry import is_heavy

CONTACT_CUTOFF = 4.5        # A; minimum heavy-atom distance defining a contact
CUTOFFS = (4.0, 4.5, 5.0, 6.0)
SEARCH_RADIUS = 14.0        # A; neighbour-search radius, well beyond any cutoff
SASA_BURIED = 1.0           # A^2; dSASA above this counts as buried on binding

HIST_BINS = np.arange(2.5, 12.01, 0.5)


def split_chains(model, ab_chains):
    """(antibody, antigen) chain ids, following `energy.eval_interface_energy`.

    Antibody chains come from the metadata; every other chain in the model is
    treated as antigen, so nothing is hardcoded to this particular complex.
    """
    antibody = [c.id for c in model if c.id in set(ab_chains)]
    antigen = [c.id for c in model if c.id not in set(ab_chains)]
    return antibody, antigen


def _residues(model, chains):
    return [r for c in model if c.id in set(chains) for r in c if r.id[0] == ' ']


def min_heavy_distances(model, from_chains, to_chains):
    """Per-residue minimum heavy-atom distance from `from_chains` to `to_chains`.

    Returns one record per residue; the distance is inf for residues with no
    partner atom inside SEARCH_RADIUS.
    """
    targets = [a for r in _residues(model, to_chains) for a in r if is_heavy(a)]
    if not targets:
        return []
    search = NeighborSearch(targets)

    records = []
    for res in _residues(model, from_chains):
        best = np.inf
        for atom in res:
            if not is_heavy(atom):
                continue
            for other in search.search(atom.get_coord(), SEARCH_RADIUS, level='A'):
                best = min(best, float(np.linalg.norm(atom.get_coord() - other.get_coord())))
        records.append({
            'chain': res.get_parent().id,
            'resseq': res.id[1],
            'icode': res.id[2].strip(),
            'resname': res.get_resname(),
            'min_dist': best,
        })
    return records


def residue_key(record):
    return (record['chain'], record['resseq'], record['icode'])


def interface_residues(records, cutoff=CONTACT_CUTOFF):
    return {residue_key(r) for r in records if r['min_dist'] <= cutoff}


def file_digest(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def cdr_ranges(directory):
    """{tag: (residue_first, residue_last)} for every CDR of a run."""
    with open(os.path.join(directory, 'metadata.json')) as f:
        metadata = json.load(f)
    return {
        item['tag']: (item['residue_first'], item['residue_last'])
        for item in metadata['items']
        if item.get('residue_first') and item.get('residue_last')
    }


def region_of(chain, resseq, icode, ranges):
    """CDR tag containing this residue, else `framework-<chain>`."""
    for tag, (first, last) in ranges.items():
        if first[0] != chain:
            continue
        if (first[1], first[2].strip()) <= (resseq, icode) <= (last[1], last[2].strip()):
            return tag
    return f'framework-{chain}'


def buried_sasa(model, keep_chains, remove_chains):
    """dSASA per residue: solvent exposure lost by `keep_chains` on binding.

    Computed as SASA(keep_chains alone) - SASA(keep_chains in the complex), so a
    positive value means the residue is buried by the partner.
    """
    shrake = ShrakeRupley()

    bound = copy.deepcopy(model)
    shrake.compute(bound, level='R')
    in_complex = {(r.get_parent().id, r.id[1], r.id[2].strip()): r.sasa
                  for c in bound if c.id in set(keep_chains)
                  for r in c if r.id[0] == ' '}

    free = copy.deepcopy(model)
    for chain_id in remove_chains:
        free.detach_child(chain_id)
    shrake.compute(free, level='R')
    alone = {(r.get_parent().id, r.id[1], r.id[2].strip()): r.sasa
             for c in free for r in c if r.id[0] == ' '}

    return {k: alone[k] - in_complex.get(k, 0.0) for k in alone}


def _histogram(distances):
    finite = np.array([d for d in distances if np.isfinite(d)])
    counts, edges = np.histogram(finite[finite < HIST_BINS[-1]], bins=HIST_BINS)
    return counts, edges


def reference_report(task, out_dir):
    """Contact analysis of one reference structure. Returns the per-residue table."""
    model = task.get_ref_biopython_model()
    directory = run_dir(task)
    ranges = cdr_ranges(directory)
    antibody, antigen = split_chains(model, task.ab_chains)

    ab_records = min_heavy_distances(model, antibody, antigen)
    ag_records = min_heavy_distances(model, antigen, antibody)

    ab_burial = buried_sasa(model, antibody, antigen)
    ag_burial = buried_sasa(model, antigen, antibody)

    for record in ab_records:
        record['side'] = 'antibody'
        record['region'] = region_of(record['chain'], record['resseq'],
                                     record['icode'], ranges)
        record['dsasa'] = ab_burial.get(residue_key(record), np.nan)
    for record in ag_records:
        record['side'] = 'antigen'
        record['region'] = f"antigen-{record['chain']}"
        record['dsasa'] = ag_burial.get(residue_key(record), np.nan)

    print(f'\n=== {task.structure}  {task.cdr}  reference ===')
    print(f'  file            : {task.ref_path}')
    print(f'  antibody chains : {"".join(antibody)}  ({len(ab_records)} residues)')
    print(f'  antigen chains  : {"".join(antigen)}  ({len(ag_records)} residues)')

    ab_dist = np.array([r['min_dist'] for r in ab_records])
    print(f'\n  paratope / epitope size by cutoff')
    print(f"    {'cutoff':>8}{'antibody':>10}{'antigen':>9}")
    for cutoff in CUTOFFS:
        n_ab = int((ab_dist <= cutoff).sum())
        n_ag = sum(1 for r in ag_records if r['min_dist'] <= cutoff)
        mark = '  <- primary' if cutoff == CONTACT_CUTOFF else ''
        print(f'    {cutoff:8.1f}{n_ab:10d}{n_ag:9d}{mark}')
    print(f'    closest contact: {ab_dist.min():.2f} A')

    print(f'\n  paratope composition by region')
    print(f"    {'region':16}" + ''.join(f'{c:>7.1f}' for c in CUTOFFS) + f"{'min':>9}")
    regions = sorted(ranges) + sorted({f'framework-{c}' for c in antibody})
    for region in regions:
        subset = np.array([r['min_dist'] for r in ab_records if r['region'] == region])
        if subset.size == 0:
            continue
        counts = ''.join(f'{int((subset <= c).sum()):7d}' for c in CUTOFFS)
        print(f'    {region:16}{counts}{subset.min():9.2f}')

    print(f'\n  contacting antibody residues at <= {CONTACT_CUTOFF} A')
    for r in sorted([r for r in ab_records if r['min_dist'] <= CONTACT_CUTOFF],
                    key=lambda x: x['min_dist']):
        icode = r['icode'] or ' '
        print(f"    {r['region']:16} {r['chain']}{r['resseq']}{icode:<2} "
              f"{r['resname']}  {r['min_dist']:.2f}")

    epitope = sorted([r for r in ag_records if r['min_dist'] <= CONTACT_CUTOFF],
                     key=lambda x: x['min_dist'])
    print(f'\n  epitope: {len(epitope)} antigen residues at <= {CONTACT_CUTOFF} A')
    print('    ' + ', '.join(f"{r['chain']}{r['resseq']}{r['icode']} {r['resname']}"
                             for r in epitope))

    counts, edges = _histogram(ab_dist)
    print(f'\n  min-distance histogram, antibody side (0.5 A bins)')
    for count, low in zip(counts, edges[:-1]):
        trough = '  <-- cutoff sits here' if low == CONTACT_CUTOFF - 0.5 else ''
        print(f'    {low:4.1f}-{low + 0.5:4.1f} {"#" * count} ({count}){trough}')

    buried = sum(1 for r in ab_records
                 if np.isfinite(r['dsasa']) and r['dsasa'] > SASA_BURIED)
    agree = sum(1 for r in ab_records
                if (r['min_dist'] <= CONTACT_CUTOFF)
                == (np.isfinite(r['dsasa']) and r['dsasa'] > SASA_BURIED))
    print(f'\n  SASA cross-check (independent of distance)')
    print(f'    residues with dSASA > {SASA_BURIED} A^2 : {buried}')
    print(f'    agreement with the {CONTACT_CUTOFF} A cutoff : '
          f'{agree}/{len(ab_records)} ({100 * agree / len(ab_records):.0f}%)')

    table = pd.DataFrame(ab_records + ag_records)
    table.insert(0, 'reference', os.path.relpath(task.ref_path, directory))
    table.insert(0, 'structure', task.structure)
    table.insert(0, 'method', task.method)
    table['contact'] = table['min_dist'] <= CONTACT_CUTOFF
    return table[['method', 'structure', 'reference', 'side', 'region', 'chain',
                  'resseq', 'icode', 'resname', 'min_dist', 'dsasa', 'contact']]


REGION_ORDER = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']


def _residue_label(row):
    return f"{row['chain']}{row['resseq']}{row['icode']}"


def _sort_key(label):
    digits = ''.join(c for c in label[1:] if c.isdigit())
    return (label[0], int(digits) if digits else 0, label)


def summarise(table):
    """One row per region: how many residues contact, and which.

    Grouped by reference file. Contact sets from different references are never
    merged: with `--ref-pfx rosetta` each reference had a different single CDR
    relaxed -- which also drags downstream residues of that chain through the
    fold tree and repacks spatial neighbours, while leaving the rest untouched --
    so they are not replicates of one measurement and intersecting them would
    mean nothing.
    """
    contacts = table[table['contact']].copy()
    if contacts.empty:
        return pd.DataFrame()
    contacts['residue'] = contacts.apply(_residue_label, axis=1)

    rows = []
    for (method, structure, reference, side, region), group in contacts.groupby(
            ['method', 'structure', 'reference', 'side', 'region'], sort=False):
        names = group.drop_duplicates('residue').set_index('residue')['resname']
        labels = [f'{r} {names[r]}' for r in sorted(names.index, key=_sort_key)]
        rows.append({
            'method': method,
            'structure': structure,
            'reference': reference,
            'side': side,
            'region': region,
            'n_contact': len(labels),
            'residues': '; '.join(labels),
        })

    summary = pd.DataFrame(rows)
    rank = {r: i for i, r in enumerate(REGION_ORDER)}
    summary['_o'] = summary['region'].map(lambda r: rank.get(r, len(rank)))
    return summary.sort_values(['reference', 'side', '_o', 'region']).drop(columns='_o')


def print_summary(summary):
    for side in ('antibody', 'antigen'):
        part = summary[summary['side'] == side]
        if part.empty:
            continue
        print(f'\n=== {side} contacts by region ===')
        print(f"{'region':14}{'n':>4}  residues")
        for _, row in part.iterrows():
            print(f"{row['region']:14}{row['n_contact']:>4}  {row['residues']}")
        print(f"{'TOTAL':14}{part['n_contact'].sum():>4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--layout', choices=('flat', 'per_run', 'both'), default='per_run')
    parser.add_argument('--ref-pfx', type=str, default='',
                        help="postfix of the reference file to analyse; '' (default) "
                             "reads the unrelaxed REF1.pdb, 'rosetta' reads "
                             "REF1_rosetta.pdb")
    parser.add_argument('--reference-only', action='store_true', default=True,
                        help='analyse the reference structures (currently the only mode)')
    add_selection_args(parser)
    args = parser.parse_args()

    tasks = TaskScanner(root=args.root, postfix=args.pfx).scan()
    if not tasks:
        print(f'No structures matching *_{args.pfx}.pdb found under {args.root}.')
        return
    # The reference file is chosen independently of the designs' postfix.
    ref_name = f'REF1_{args.ref_pfx}.pdb' if args.ref_pfx else 'REF1.pdb'
    for task in tasks:
        task.ref_path = os.path.join(os.path.dirname(task.in_path), ref_name)
    selected = filter_tasks(tasks, args)
    if not selected:
        print(f'{len(tasks)} structures found, none matched the selection filters.')
        return

    # Dedupe by file content, not path: every CDR directory holds its own copy of
    # the same native, so six paths collapse to one structure.
    seen, references = set(), []
    for task in selected:
        if not os.path.exists(task.ref_path):
            continue
        key = (task.structure, file_digest(task.ref_path))
        if key not in seen:
            seen.add(key)
            references.append(task)
    print(f'Found {len(tasks)} structures, '
          f'{len(references)} distinct reference structure(s).')

    tables = []
    for task in references:
        tables.append(reference_report(task, run_dir(task)))
    table = pd.concat(tables, ignore_index=True)

    def write(subset, base):
        out_dir = evaluation_dir(base)
        path = os.path.join(out_dir, 'interface_reference.csv')
        subset.to_csv(path, index=False, float_format='%.4f')
        summary_path = os.path.join(out_dir, 'interface_summary.csv')
        summarise(subset).to_csv(summary_path, index=False)
        print(f'\nWrote {path}')
        print(f'Wrote {summary_path}')

    if args.layout in ('per_run', 'both'):
        by_run = {}
        for task, part in zip(references, tables):
            by_run.setdefault(run_dir(task), []).append(part)
        for directory, parts in by_run.items():
            write(pd.concat(parts, ignore_index=True), directory)
    if args.layout in ('flat', 'both'):
        write(table, args.out or args.root)

    print_summary(summarise(table))


if __name__ == '__main__':
    main()
