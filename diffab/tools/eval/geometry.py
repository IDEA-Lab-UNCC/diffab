"""Reference-free structural validity metrics for generated CDRs.

Unlike `similarity` (which compares against the reference) and `energy` (which
scores the antibody-antigen interface), everything here asks whether the
generated backbone is a chemically legal protein on its own terms.

This module deliberately has no PyRosetta dependency: `ref2015` carries no
bond-length or bond-angle term, and FastRelax samples torsions only, so neither
the relax stage nor the energy stage can see the defects measured here.
"""

import os
import numpy as np
from Bio import PDB
from Bio.PDB import NeighborSearch, Selection

from diffab.tools.eval.base import EvalTask
from diffab.tools.eval.similarity import extract_reslist


# Engh & Huber (1991) ideal backbone geometry, as used by Rosetta and by PDB validation.
IDEAL_PEP_CN = 1.329    # C(i) -- N(i+1), the peptide bond
IDEAL_N_CA = 1.458
IDEAL_CA_C = 1.525
IDEAL_C_O = 1.231
IDEAL_N_CA_C = 111.2    # degrees

PEP_TOL = 0.05          # A; ~10x the spread measured in framework/crystal backbones
BREAK_TOL = 0.30        # A; beyond this the chain is effectively severed
OMEGA_TOL = 30.0        # degrees from planarity
CLASH_DIST = 2.2        # A; non-bonded heavy-atom contact counted as a clash
CLASH_SEVERE = 1.5      # A

BACKBONE = ('N', 'CA', 'C', 'O')


def _is_heavy(atom):
    element = (atom.element or '').strip().upper()
    if element:
        return element != 'H'
    # Fall back to the atom name when the element column is blank (e.g. '1HB').
    return not atom.get_id().lstrip('0123456789').upper().startswith('H')


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _backbone_coords(reslist):
    """Backbone coordinates per residue, plus a flag for chain continuity.

    Residues missing a backbone atom are dropped, and the gap is recorded so
    that a peptide bond is never measured across a residue that was skipped.
    """
    coords, contiguous, n_incomplete = [], [], 0
    for res in reslist:
        if res.id[0] != ' ':
            continue
        try:
            coords.append({a: np.array(res[a].get_coord(), dtype=np.float64) for a in BACKBONE})
            contiguous.append(True)
        except KeyError:
            n_incomplete += 1
            contiguous.append(False)
    return coords, n_incomplete


def backbone_geometry(reslist):
    """Bond lengths, bond angles and peptide planarity over `reslist`."""
    res, n_incomplete = _backbone_coords(reslist)
    if len(res) == 0:
        return {}

    pep = np.array([np.linalg.norm(res[i + 1]['N'] - res[i]['C']) for i in range(len(res) - 1)])
    n_ca = np.array([np.linalg.norm(r['CA'] - r['N']) for r in res])
    ca_c = np.array([np.linalg.norm(r['C'] - r['CA']) for r in res])
    c_o = np.array([np.linalg.norm(r['O'] - r['C']) for r in res])
    ang = np.array([_angle(r['N'], r['CA'], r['C']) for r in res])

    # omega: CA(i) - C(i) - N(i+1) - CA(i+1). Planar trans is +/-180, cis is 0.
    omega = np.array([
        _dihedral(res[i]['CA'], res[i]['C'], res[i + 1]['N'], res[i + 1]['CA'])
        for i in range(len(res) - 1)
    ])
    trans_dev = 180.0 - np.abs(omega)      # 0 for ideal trans
    cis_dev = np.abs(omega)                # 0 for ideal cis (proline)
    omega_dev = np.minimum(trans_dev, cis_dev)

    scores = {
        'n_res': len(res),
        'n_incomplete_res': n_incomplete,
        'nca_c_dev_max': float(np.abs(ang - IDEAL_N_CA_C).max()),
        'bond_nca_dev_max': float(np.abs(n_ca - IDEAL_N_CA).max()),
        'bond_cac_dev_max': float(np.abs(ca_c - IDEAL_CA_C).max()),
        'bond_co_dev_max': float(np.abs(c_o - IDEAL_C_O).max()),
    }
    if len(pep) > 0:
        scores.update({
            'pep_mean': float(pep.mean()),
            'pep_min': float(pep.min()),
            'pep_max': float(pep.max()),
            'pep_rmsd_ideal': float(np.sqrt(np.mean((pep - IDEAL_PEP_CN) ** 2))),
            'pep_n_viol': int(np.sum(np.abs(pep - IDEAL_PEP_CN) > PEP_TOL)),
            'pep_frac_viol': float(np.mean(np.abs(pep - IDEAL_PEP_CN) > PEP_TOL)),
            'pep_n_break': int(np.sum(np.abs(pep - IDEAL_PEP_CN) > BREAK_TOL)),
            'omega_dev_mean': float(omega_dev.mean()),
            'omega_dev_max': float(omega_dev.max()),
            'omega_n_viol': int(np.sum(omega_dev > OMEGA_TOL)),
        })
    return scores


def backbone_clashes(model, reslist):
    """Non-bonded heavy-atom clashes involving the generated backbone.

    Restricted to backbone atoms because pre-relax structures carry no side
    chains for the designed residues; this keeps the number comparable across
    the relax stage.
    """
    # Position of each residue within its chain, to exclude bonded neighbours.
    order = {}
    for chain in model:
        idx = 0
        for res in chain:
            if res.id[0] != ' ':
                continue
            order[(chain.id, res.id)] = idx
            idx += 1

    heavy = [a for a in Selection.unfold_entities(model, 'A') if _is_heavy(a)]
    if not heavy:
        return {}
    search = NeighborSearch(heavy)

    focus = []
    for res in reslist:
        if res.id[0] != ' ':
            continue
        for name in BACKBONE:
            if name in res:
                focus.append(res[name])

    n_clash, n_severe, min_dist = 0, 0, np.inf
    seen = set()
    for atom in focus:
        res_a = atom.get_parent()
        key_a = (res_a.get_parent().id, res_a.id)
        for other in search.search(atom.get_coord(), CLASH_DIST, level='A'):
            if other is atom:
                continue
            res_b = other.get_parent()
            key_b = (res_b.get_parent().id, res_b.id)
            if key_a == key_b:
                continue
            # Skip covalently bonded neighbours along the chain.
            if key_a[0] == key_b[0] and key_a in order and key_b in order:
                if abs(order[key_a] - order[key_b]) == 1:
                    continue
            pair = tuple(sorted((id(atom), id(other))))
            if pair in seen:
                continue
            seen.add(pair)
            dist = float(np.linalg.norm(atom.get_coord() - other.get_coord()))
            n_clash += 1
            if dist < CLASH_SEVERE:
                n_severe += 1
            min_dist = min(min_dist, dist)

    return {
        'clash_n': n_clash,
        'clash_n_severe': n_severe,
        'clash_min_dist': float(min_dist) if np.isfinite(min_dist) else float('nan'),
    }


def reslist_shift(res_list1, res_list2):
    """Per-atom displacement between two copies of the same residue list.

    No superposition: DiffAb holds the framework and antigen fixed, so both
    structures already share a frame. Backbone only, since the pre-relax
    structure has no side chains for the designed residues.
    """
    r1, _ = _backbone_coords(res_list1)
    r2, _ = _backbone_coords(res_list2)
    if len(r1) == 0 or len(r1) != len(r2):
        return {}
    ca = np.array([np.linalg.norm(a['CA'] - b['CA']) for a, b in zip(r1, r2)])
    bb = np.array([np.linalg.norm(a[n] - b[n]) for a, b in zip(r1, r2) for n in BACKBONE])
    return {
        'shift_ca_rmsd': float(np.sqrt(np.mean(ca ** 2))),
        'shift_bb_rmsd': float(np.sqrt(np.mean(bb ** 2))),
        'shift_bb_max': float(bb.max()),
    }


def raw_path(path, postfix):
    """Path to the pre-relax structure that `path` was produced from."""
    if not postfix:
        return path
    directory, fname = os.path.split(path)
    stem, ext = os.path.splitext(fname)
    suffix = '_' + postfix
    if not stem.endswith(suffix):
        return path
    return os.path.join(directory, stem[:-len(suffix)] + ext)


def _load(path):
    return PDB.PDBParser(QUIET=True).get_structure(path, path)[0]


def _cdr_scores(model, task, prefix):
    reslist = extract_reslist(model, task.residue_first, task.residue_last)
    scores = backbone_geometry(reslist)
    scores.update(backbone_clashes(model, reslist))
    return {prefix + k: v for k, v in scores.items()}


def eval_backbone_validity(task: EvalTask):
    """Geometry and clashes of the generated CDR, as evaluated (post-relax)."""
    task.scores.update(_cdr_scores(task.get_gen_biopython_model(), task, ''))
    return task


def eval_prerelax_validity(task: EvalTask, postfix=None):
    """Same metrics on the raw model output, before any relaxation.

    Bond lengths and angles are unchanged by torsion-space relax, but clashes
    are not: this is the only place they can be measured as generated.
    """
    path = raw_path(task.in_path, postfix)
    if path == task.in_path or not os.path.exists(path):
        return task
    task.scores.update(_cdr_scores(_load(path), task, 'pre_'))
    return task


def eval_relax_shift(task: EvalTask, postfix=None):
    """How far relaxation had to move the generated backbone."""
    path = raw_path(task.in_path, postfix)
    if path == task.in_path or not os.path.exists(path):
        return task
    model_pre, model_post = _load(path), task.get_gen_biopython_model()
    task.scores.update(reslist_shift(
        extract_reslist(model_pre, task.residue_first, task.residue_last),
        extract_reslist(model_post, task.residue_first, task.residue_last),
    ))
    return task


def native_validity(task: EvalTask):
    """Geometry and clashes of the *native* CDR, as the null for the generated one.

    The framework and antigen are shared between the two structures, so any
    difference here is attributable to the designed residues. Constant per
    (structure, CDR), so callers should cache it.
    """
    return _cdr_scores(task.get_ref_biopython_model(), task, 'ref_')


def native_relax_shift(task: EvalTask, postfix=None):
    """The same shift measured on the native CDR: the null for `eval_relax_shift`.

    Constant per (structure, CDR), so callers should cache it.
    """
    path = raw_path(task.ref_path, postfix)
    if path == task.ref_path or not os.path.exists(path):
        return {}
    shift = reslist_shift(
        extract_reslist(_load(path), task.residue_first, task.residue_last),
        extract_reslist(task.get_ref_biopython_model(), task.residue_first, task.residue_last),
    )
    return {'ref_' + k: v for k, v in shift.items()}
