"""Shared types for structure-prediction backends, and the ipSAE file contract.

A folding backend takes sequences and returns coordinates plus confidence. What
it must hand back is fixed by what `third_party/ipsae/ipsae.py` reads, not by
what any particular model happens to expose:

  * a PDB file, from which ipsae takes **CA atoms in file order** as the PAE
    index and **CB atoms** (CA for glycine) as the distance matrix, and
  * a JSON file with a `pae` matrix and optional `plddt`, `ptm`, `iptm`.

Two invariants follow, and neither one fails loudly on its own -- a violation
produces a plausible-looking ipSAE that is simply wrong. `check_af2_inputs`
enforces both by re-parsing the written PDB with ipsae's own column rules:

  1. `len(CA atoms) == pae.shape[0]`, with the residue order identical. ipsae
     does no sorting or matching; index i of the PAE is whatever the i-th CA in
     the file happens to be.
  2. `len(CB atoms) == len(CA atoms)`. ipsae builds its distance matrix from CB
     and indexes it with CA-derived indices, so a single residue missing its CB
     silently shifts every distance after it.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

FilePath = str

# Residues ipsae.py counts as polymer tokens (`residue_set` in that script).
# Anything else in the file is skipped, so it must be skipped here too.
_STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
}


@dataclass
class FoldTask:
    """One complex to predict.

    `chains` is ordered: the backend is asked to emit chains in this order, and
    the PAE matrix is interpreted in it. Python dicts preserve insertion order,
    which is what makes that promise meaningful.
    """
    name: str
    chains: Dict[str, str]              # chain id -> one-letter sequence
    out_dir: FilePath
    seed: int = 0

    def total_length(self):
        return sum(len(s) for s in self.chains.values())


@dataclass
class FoldResult:
    """A prediction, in the form ipsae.py consumes."""
    pdb_path: FilePath
    cif_path: Optional[FilePath]
    pae: np.ndarray                     # (N, N), angstroms
    plddt: np.ndarray                   # (N,), 0-100
    ptm: float
    iptm: float
    chain_order: List[str]
    residues: List[Tuple[str, int]] = field(default_factory=list)  # (chain, resseq)

    def json_payload(self):
        """The AF2-style dict ipsae.py's `af2` branch reads."""
        return {
            'pae': np.asarray(self.pae).tolist(),
            'plddt': np.asarray(self.plddt).tolist(),
            'ptm': float(self.ptm),
            'iptm': float(self.iptm),
        }


class FoldingEngine(abc.ABC):
    """Context-managed structure predictor, mirroring `dock/base.py`."""

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, typ, value, traceback):
        pass

    @abc.abstractmethod
    def fold(self, task: FoldTask) -> FoldResult:
        pass


def scan_pdb_tokens(pdb_path):
    """CA and CB atoms of a PDB, read exactly the way ipsae.py reads them.

    Deliberately duplicates ipsae's fixed-column parsing instead of using
    BioPython: the point is to check what *that* script will see, and BioPython
    is more forgiving about malformed columns than it is.
    """
    ca, cb = [], []
    with open(pdb_path, 'r') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain_id = line[21].strip()
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if atom_name == 'CA' or 'C1' in atom_name:
                ca.append((chain_id, resseq, resname))
            if atom_name == 'CB' or 'C3' in atom_name or (resname == 'GLY' and atom_name == 'CA'):
                cb.append((chain_id, resseq, resname))
    return ca, cb


def check_af2_inputs(pdb_path, result: FoldResult):
    """Raise unless the PDB/PAE pair satisfies both ipSAE invariants."""
    ca, cb = scan_pdb_tokens(pdb_path)
    n_pae = int(np.asarray(result.pae).shape[0])

    if len(ca) != n_pae:
        raise ValueError(
            f'{pdb_path}: {len(ca)} CA atoms but the PAE matrix is {n_pae}x{n_pae}. '
            f'ipsae.py indexes the PAE by CA order, so these must match exactly.'
        )
    if len(cb) != len(ca):
        missing = [f'{c}{r} {n}' for (c, r, n) in ca if (c, r, n) not in set(cb)][:5]
        raise ValueError(
            f'{pdb_path}: {len(ca)} CA atoms but {len(cb)} CB atoms. ipsae.py builds '
            f'its distance matrix from CB and indexes it with CA positions, so every '
            f'non-glycine residue needs a CB. First missing: {missing}'
        )
    if np.asarray(result.pae).shape != (n_pae, n_pae):
        raise ValueError(f'PAE matrix is not square: {np.asarray(result.pae).shape}')
    if len(result.plddt) != n_pae:
        raise ValueError(
            f'pLDDT has {len(result.plddt)} entries but the PAE is {n_pae}x{n_pae}.'
        )

    file_chains = [c for c, _, _ in ca]
    seen = list(dict.fromkeys(file_chains))
    if seen != list(result.chain_order):
        raise ValueError(
            f'{pdb_path}: chain order in the file is {seen}, but the prediction '
            f'reports {list(result.chain_order)}. The PAE would be misaligned.'
        )

    # ipsae reads columns 22:26 only -- an insertion code in column 27 is
    # dropped, silently merging 100/100A/100B. Predicted structures are numbered
    # sequentially so this should never fire, but a hand-made input could.
    per_chain = {}
    for c, r, _ in ca:
        per_chain.setdefault(c, []).append(r)
    for chain, nums in per_chain.items():
        if len(set(nums)) != len(nums):
            raise ValueError(
                f'{pdb_path}: chain {chain} has duplicate residue numbers. ipsae.py '
                f'ignores insertion codes, so the structure must be renumbered '
                f'sequentially before scoring.'
            )
        if max(nums) > 9999:
            raise ValueError(
                f'{pdb_path}: chain {chain} residue number {max(nums)} overflows the '
                f'PDB 4-column field. Use a shorter construct or the mmCIF path.'
            )
    return len(ca)
