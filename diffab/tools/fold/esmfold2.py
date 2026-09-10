"""ESMFold2 backend (CZI Biohub, 2026) -- sequence in, complex + PAE out.

Chosen over AlphaFold3 and Boltz-2 for one reason: it is reported to beat AF3 on
antibody-antigen binding-pose accuracy, which is the only regime this project
cares about and the one every predictor is worst at. It is also MIT-licensed,
pip-installable, and takes multi-chain input natively, so no glycine-linker hack
is needed the way it was with the original ESMFold.

Two things here are written defensively on purpose.

`_esm_symbols` resolves class names from a candidate list: the upstream README
and the HuggingFace card disagree on capitalisation (`EsmFold2Model` vs
`ESMFold2Model`), and this code is written on a machine that cannot install the
package.

`extract_confidence` does the same for the PAE accessor, which is the one part
of the contract no reachable documentation pins down -- the hosted OpenProtein
wrapper exposes `get_pae()` returning `(diffusion_samples, N, N)`, but the local
package's name is unconfirmed. Run `--probe` on the workstation first; it prints
the real attribute names and shapes, and this list should then be trimmed to
the one that works.

    python -m diffab.tools.eval.ipsae --probe
"""

# pyright: reportMissingImports=false
import io
import os
from typing import Dict, List

import numpy as np

from diffab.tools.fold.base import (
    FoldingEngine, FoldResult, FoldTask, check_af2_inputs,
)

DEFAULT_CHECKPOINT = 'biohub/ESMFold2-Fast'

# Sampling defaults. The Fast checkpoint is single-sequence and inference
# optimised; these follow the values in the upstream quickstart.
DEFAULT_NUM_LOOPS = 3
DEFAULT_SAMPLING_STEPS = 50

# Candidate accessors, most likely first. See the module docstring.
_PAE_ATTRS = ('pae', 'get_pae', 'predicted_aligned_error', 'pde')
_PLDDT_ATTRS = ('plddt', 'get_plddt')
_PTM_ATTRS = ('ptm', 'get_ptm')
_IPTM_ATTRS = ('iptm', 'get_iptm')

_MODEL = None
_MODEL_KEY = None


def _first_attr(obj, names, what):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(
        f'Could not find {what} on {type(obj).__name__}. Tried {list(names)}. '
        f'Available: {sorted(n for n in dir(obj) if not n.startswith("_"))}'
    )


def _esm_symbols():
    """The ESMFold2 classes, tolerant of the two documented spellings."""
    try:
        from esm.models import esmfold2 as mod
    except ImportError as e:
        # Same convention as design_dock.py's missing-HDOCK check: say how to
        # fix it rather than surfacing a traceback.
        raise ImportError(
            'ESMFold2 is not installed. Install it into the environment that '
            'runs this (a recent PyTorch is required):\n'
            '    pip install esm\n'
            'Weights are pulled from HuggingFace on first use '
            f'({DEFAULT_CHECKPOINT}).'
        ) from e
    return {
        'model': _first_attr(mod, ('EsmFold2Model', 'ESMFold2Model'), 'the model class'),
        'builder': _first_attr(mod, ('ESMFold2InputBuilder', 'EsmFold2InputBuilder'),
                               'the input builder'),
        'protein': _first_attr(mod, ('ProteinInput',), 'ProteinInput'),
        'spi': _first_attr(mod, ('StructurePredictionInput',), 'StructurePredictionInput'),
    }


def load_model(checkpoint=DEFAULT_CHECKPOINT, device='cuda'):
    """Load weights once per process.

    Memoised in a module global rather than passed around, so that each Ray
    worker pays the load cost once -- the same reason `eval/binding.py` caches
    its PyRosetta init.
    """
    global _MODEL, _MODEL_KEY
    key = (checkpoint, device)
    if _MODEL is None or _MODEL_KEY != key:
        sym = _esm_symbols()
        print(f'[INFO] Loading {checkpoint} onto {device}', flush=True)
        try:
            model = sym['model'].from_pretrained(checkpoint, device=device)
        except TypeError:   # older signature: .from_pretrained(name).to(device)
            model = sym['model'].from_pretrained(checkpoint).to(device)
        _MODEL = model.eval()
        _MODEL_KEY = key
    return _MODEL


def _as_array(value):
    """Tensor or array -> numpy, detached and on the host."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _fetch(result, names, what, required=True):
    for name in names:
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        if callable(value):
            value = value()
        if value is not None:
            return value
    if not required:
        return None
    raise AttributeError(
        f'Could not read {what} from the ESMFold2 result. Tried {list(names)}. '
        f'Available attributes: {sorted(n for n in dir(result) if not n.startswith("_"))}. '
        f'Run `python -m diffab.tools.eval.ipsae --probe` and update the candidate '
        f'lists in diffab/tools/fold/esmfold2.py.'
    )


def _drop_sample_axis(array, ndim_wanted):
    """Strip the leading diffusion-sample axis, if the array carries one.

    The hosted API returns PAE as `(diffusion_samples, N, N)` and pLDDT as
    `(diffusion_samples, N)`. We always request one sample, so take the first.
    """
    array = _as_array(array)
    while array.ndim > ndim_wanted:
        if array.shape[0] != 1:
            print(f'[WARNING] {array.shape[0]} diffusion samples returned; using the first.',
                  flush=True)
        array = array[0]
    return array


def extract_confidence(result):
    """`(pae (N,N), plddt (N,), ptm, iptm)` from a prediction result."""
    pae = _drop_sample_axis(_fetch(result, _PAE_ATTRS, 'the PAE matrix'), 2)
    plddt = _drop_sample_axis(_fetch(result, _PLDDT_ATTRS, 'pLDDT'), 1)

    # pLDDT is reported on 0-1 by some builds and 0-100 by others; ipsae's
    # pDockQ terms assume 0-100.
    if plddt.size and float(np.nanmax(plddt)) <= 1.0:
        plddt = plddt * 100.0

    ptm = _fetch(result, _PTM_ATTRS, 'pTM', required=False)
    iptm = _fetch(result, _IPTM_ATTRS, 'ipTM', required=False)
    to_float = lambda v: -1.0 if v is None else float(_as_array(v).reshape(-1)[0])
    return pae, plddt, to_float(ptm), to_float(iptm)


def _cif_text(result):
    complex_obj = _fetch(result, ('complex', 'structure', 'atoms'), 'the predicted structure')
    writer = _first_attr(complex_obj, ('to_mmcif', 'to_cif'), 'an mmCIF writer')
    return writer()


def write_pdb(cif_text, out_path, chain_order: List[str], expect: Dict[str, int]):
    """mmCIF text -> a PDB that satisfies ipsae.py's parsing assumptions.

    Chains are emitted in `chain_order` (the order the sequences were given, and
    therefore the PAE's index order), residues renumbered 1..n per chain with no
    insertion codes, and everything without a CA dropped. Renumbering happens by
    building fresh Chain objects rather than mutating ids in place, which would
    collide mid-loop -- the same idiom as `tools/renumber/run.py`.
    """
    from Bio.PDB import MMCIFParser, PDBIO, Model, Chain, Selection

    structure = MMCIFParser(QUIET=True).get_structure('pred', io.StringIO(cif_text))
    source = structure[0]

    found = {c.id for c in source}
    missing = [c for c in chain_order if c not in found]
    if missing:
        raise ValueError(
            f'Prediction is missing chain(s) {missing}; it has {sorted(found)}.'
        )

    model_new = Model.Model(0)
    for chain_id in chain_order:
        chain_new = Chain.Chain(chain_id)
        n = 0
        for residue in Selection.unfold_entities(source[chain_id], 'R'):
            if 'CA' not in residue:
                continue        # water, ion, or an unresolved position
            n += 1
            copied = residue.copy()
            copied.id = (' ', n, ' ')
            chain_new.add(copied)
        if n != expect[chain_id]:
            raise ValueError(
                f'Chain {chain_id}: predicted {n} residues but {expect[chain_id]} were '
                f'submitted. The PAE would not line up with the structure.'
            )
        model_new.add(chain_new)

    pdb_io = PDBIO()
    pdb_io.set_structure(model_new)
    pdb_io.save(out_path)
    return out_path


class ESMFold2Engine(FoldingEngine):

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device='cuda',
                 num_loops=DEFAULT_NUM_LOOPS, num_sampling_steps=DEFAULT_SAMPLING_STEPS,
                 keep_cif=True):
        self.checkpoint = checkpoint
        self.device = device
        self.num_loops = num_loops
        self.num_sampling_steps = num_sampling_steps
        self.keep_cif = keep_cif

    def __enter__(self):
        load_model(self.checkpoint, self.device)
        return self

    def __exit__(self, typ, value, traceback):
        return False

    def fold(self, task: FoldTask) -> FoldResult:
        sym = _esm_symbols()
        model = load_model(self.checkpoint, self.device)

        chain_order = list(task.chains.keys())
        spi = sym['spi'](sequences=[
            sym['protein'](id=cid, sequence=seq) for cid, seq in task.chains.items()
        ])
        result = sym['builder']().fold(
            model, spi,
            num_loops=self.num_loops,
            num_sampling_steps=self.num_sampling_steps,
            num_diffusion_samples=1,
            seed=task.seed,
        )

        pae, plddt, ptm, iptm = extract_confidence(result)

        os.makedirs(task.out_dir, exist_ok=True)
        stem = os.path.join(task.out_dir, task.name)
        cif_text = _cif_text(result)
        cif_path = None
        if self.keep_cif:
            cif_path = stem + '.cif'
            with open(cif_path, 'w') as f:
                f.write(cif_text)

        pdb_path = write_pdb(
            cif_text, stem + '.pdb', chain_order,
            expect={cid: len(seq) for cid, seq in task.chains.items()},
        )

        folded = FoldResult(
            pdb_path=pdb_path, cif_path=cif_path,
            pae=pae, plddt=plddt, ptm=ptm, iptm=iptm,
            chain_order=chain_order,
        )
        check_af2_inputs(pdb_path, folded)
        return folded


def probe(device='cuda', checkpoint=DEFAULT_CHECKPOINT):
    """Fold a two-chain toy peptide and report what the result object exposes.

    The point is to settle the accessor names on a machine that actually has the
    package, before spending GPU-hours on a run that would fail at the end.
    """
    sym = _esm_symbols()
    print(f'[INFO] Resolved classes: '
          f"{ {k: v.__name__ for k, v in sym.items()} }", flush=True)

    model = load_model(checkpoint, device)
    spi = sym['spi'](sequences=[
        sym['protein'](id='A', sequence='GSHMKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK'),
        sym['protein'](id='B', sequence='MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIP'),
    ])
    result = sym['builder']().fold(
        model, spi, num_loops=1, num_sampling_steps=10,
        num_diffusion_samples=1, seed=0,
    )

    print('\n--- result attributes ---')
    for name in sorted(n for n in dir(result) if not n.startswith('_')):
        try:
            value = getattr(result, name)
        except Exception as e:                          # noqa: BLE001
            print(f'  {name}: <error {type(e).__name__}>')
            continue
        if callable(value):
            print(f'  {name}(): callable')
            continue
        shape = getattr(value, 'shape', None)
        print(f'  {name}: {type(value).__name__}'
              + (f' shape={tuple(shape)}' if shape is not None else ''))

    print('\n--- extract_confidence ---')
    pae, plddt, ptm, iptm = extract_confidence(result)
    print(f'  pae   {pae.shape} range {pae.min():.2f}-{pae.max():.2f}')
    print(f'  plddt {plddt.shape} range {plddt.min():.2f}-{plddt.max():.2f}')
    print(f'  ptm={ptm:.4f}  iptm={iptm:.4f}')
    print('\nExpected 74 residues (37 + 37).')
    return result
