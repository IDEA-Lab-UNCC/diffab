"""Per-residue `ref2015` term breakdown over the designed CDR residues.

`total_score` alone is a poor validity signal here: it carries no bond-length or
bond-angle term (so it is blind to the connectivity defects that `geometry`
measures) and its `ref` term shifts with amino-acid composition, which differs
between designs. The individual terms are the useful part -- in particular
`p_aa_pp` (is the designed residue type compatible with the backbone it was
given) and `rama_prepro` (are the backbone dihedrals in an allowed region).

    python -m diffab.tools.eval.rosetta_terms --root ./results --pfx rosetta
"""

# pyright: reportMissingImports=false
import os
import argparse
import pandas as pd

from diffab.tools.eval.base import EvalTask, TaskScanner

# Terms reported, in the order they are written to the CSV.
TERMS = (
    'fa_rep',           # steric clashes
    'fa_atr',           # packing
    'fa_sol',           # desolvation
    'fa_elec',
    'fa_dun',           # side-chain rotamer strain
    'p_aa_pp',          # P(amino acid | phi, psi) -- sequence/backbone compatibility
    'rama_prepro',      # backbone dihedral legality
    'omega',            # peptide planarity
    'hbond_sr_bb',      # backbone H-bonds, short range
    'hbond_lr_bb',      # backbone H-bonds, long range
    'hbond_bb_sc',
    'hbond_sc',
    'ref',              # composition-dependent reference energy
)

_INITIALISED = False
_SCOREFXN = None


def _pyrosetta():
    """Import and initialise PyRosetta on first use.

    Deferred so that importing this module (or anything that re-exports it) does
    not require PyRosetta to be installed.
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


def _pose_index(pose, residue):
    """Pose index for a metadata residue identifier like ['A', 95, ' ']."""
    if residue[-1] == ' ':
        residue = residue[:-1]
    return pose.pdb_info().pdb2pose(*residue)


def cdr_term_scores(pdb_path, residue_first, residue_last):
    """Weighted `ref2015` terms summed over the CDR residues.

    Two-body energies are split between the participating residues by Rosetta,
    so these are the CDR's share of each term rather than the energy of every
    interaction it takes part in.
    """
    pyrosetta, scorefxn = _pyrosetta()
    from pyrosetta.rosetta.core.scoring import score_type_from_name

    pose = pyrosetta.pose_from_pdb(pdb_path)
    scorefxn(pose)
    energies = pose.energies()
    weights = scorefxn.weights()

    first, last = _pose_index(pose, residue_first), _pose_index(pose, residue_last)
    if first == 0 or last == 0 or last < first:
        return {}
    n_res = last - first + 1

    scores = {}
    for name in TERMS:
        st = score_type_from_name(name)
        weight = weights[st]
        total = sum(energies.residue_total_energies(i)[st] for i in range(first, last + 1))
        scores[f'term_{name}'] = float(total * weight)
    total_score = sum(energies.residue_total_energy(i) for i in range(first, last + 1))
    scores['term_total'] = float(total_score)
    scores['term_total_per_res'] = float(total_score / n_res)
    # `ref` depends only on composition; subtracting it makes totals comparable
    # across designs with different CDR sequences.
    scores['term_total_no_ref'] = float(total_score - scores['term_ref'])
    scores['term_n_res'] = n_res
    return scores


def eval_rosetta_terms(task: EvalTask):
    task.scores.update(cdr_term_scores(task.in_path, task.residue_first, task.residue_last))
    return task


def native_rosetta_terms(task: EvalTask):
    scores = cdr_term_scores(task.ref_path, task.residue_first, task.residue_last)
    return {'ref_' + k: v for k, v in scores.items()}


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
        task = eval_rosetta_terms(task)
        key = (task.structure, task.cdr)
        if key not in native_cache:
            native_cache[key] = native_rosetta_terms(task)
        task.scores.update(native_cache[key])
        rows.append(task.to_report_dict())
        if i % 50 == 0:
            print(f'  {i}/{len(tasks)}')

    table = pd.DataFrame(rows)
    out_dir = args.out or args.root
    per_design = os.path.join(out_dir, 'terms_per_design.csv')
    table.to_csv(per_design, index=False, float_format='%.6f')

    cols = [c for c in table.columns if c.startswith('term_') or c.startswith('ref_term_')]
    summary = table.groupby(['method', 'structure', 'cdr'])[cols].mean()
    summary_path = os.path.join(out_dir, 'terms_summary.csv')
    summary.to_csv(summary_path, float_format='%.4f')

    print(f'\nWrote {per_design}')
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
