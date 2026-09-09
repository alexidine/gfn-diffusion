"""Pre-encoded per-atom molecular embeddings, in TREE ORDER, for conditioning the policy.

WHY PRECOMPUTE. The encoder reads the 2D bond graph -- atom types, bonds, parity -- and
nothing it sees changes during a rollout. Geometry enters through the dynamic state path,
never here (feed the encoder `pos` and you condition on the answer). So calling it inside the
SDE loop recomputes a constant ~T times per rollout, on a path that is already DISPATCH-BOUND,
with an 800k-parameter 4-layer network carrying dense L x L attention. Once per molecule, ever,
is the same number.

PRECOMPUTING IS FREEZING. A cached embedding carries no gradient, so the encoder cannot be
fine-tuned end-to-end while the cache is in use. That is the intended trade for now: the SSL
pretraining existed to earn a representation good enough to freeze, and fine-tuning would
silently invalidate the validation that bought it. It is reversible -- the cache is a fast
path, not a commitment.

⚠ THE ATOM ORDERINGS OF THE TWO PATHS DO NOT MATCH, AND THE MISMATCH IS SILENT.
`graph_from_smiles` builds heavy atoms first and appends every hydrogen (RDKit `AddHs`); the
conformer path reorders to the torsion tree's placement order. Measured on butanol:

    conformer (spec.z):  [6, 1, 1, 6, 6, 1, 1, 6, 1, 1, 8, 1, 1, 1, 1]
    encoder   (enc z) :  [6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Attaching encoder-order embeddings to a conformer batch would condition EVERY ATOM ON THE
WRONG ATOM'S EMBEDDING, with no shape error and entirely plausible numbers. The alignment is
`spec.perm`, verified exact here on every molecule at build time:

    en.mol atom order == graph_from_smiles atom order        (checked)
    spec.z == mol.z[spec.perm]                               (checked)

so embeddings are computed in encoder order and permuted ONCE, at build time, into tree order.
The stored `z_tree` lets the consumer re-check the alignment at load without rebuilding.

THE CACHE IS STAMPED WITH THE ENCODER IT CAME FROM and refuses to load against a different
one -- the same discipline the replay buffers grew after an energy-currency mismatch restored
rows that were silently in the wrong units.

    python -m models.encoder_cache --smiles CCCCO CC(C)CO --out conformer_embeddings.pt
    python -m models.encoder_cache --smiles-file mols.txt --ckpt <path> --out emb.pt
"""
from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from models.encoder_probe import ARMS, Sample, atom_features
from paths import artifact
from models.graph_encoder import MPNNEncoder, dense_spd_batch
from models.graph_encodings import (bond_features_from_smiles, canonical_root,
                                    graph_from_smiles, mol_for_labels, rwse, shortest_paths)

FORMAT = 'encoder_cache_v1'
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'encoder_ckpt', 'mp+attn+spd_n20000_s0.pt')


def _fingerprint(path: str) -> str:
    """sha256 of the checkpoint file. Exact, cheap, and it changes when the weights do."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _features(smiles: str, encoding: str, k: int) -> Sample:
    """`build_sample`'s feature half, without the probe labels.

    Deliberately mirrors it line for line rather than calling it: `build_sample` also
    evaluates every PROBE, and `orbit_size` runs an automorphism enumeration that is the most
    expensive thing in the battery and can raise on symmetric molecules. The cache needs the
    inputs only. Any drift between this and `build_sample` is a feature-parity bug, so they
    must be edited together.
    """
    z, e1, parity = graph_from_smiles(smiles)
    n = len(z)
    root = canonical_root(mol_for_labels(smiles))
    bf = bond_features_from_smiles(smiles)
    spd = shortest_paths(e1, n)
    struct = {'none': lambda: np.zeros((n, k)),
              'rwse': lambda: rwse(e1, n, k=k)}[encoding]()
    return Sample(smiles, n,
                  atom_features(z, e1, parity, n, root), struct,
                  np.concatenate([e1, e1[::-1]], axis=1),
                  np.concatenate([bf, bf], axis=0), spd)


def _batch(samples: Sequence[Sample], device, want_spd: bool,
           dtype=torch.float32) -> Dict:
    """The encoder's inputs only -- `encoder_probe.collate`'s feature half.

    Not a call to `collate` because that also materialises `y_<probe>` for every entry of
    PROBES, so the cache would fail the moment someone adds or renames a probe. The cache
    needs the ENCODER, and the encoder never sees a label.
    """
    offs, xs, sts, eis, eas, batch = 0, [], [], [], [], []
    for gi, s_ in enumerate(samples):
        xs.append(s_.x); sts.append(s_.struct); eis.append(s_.edge_index + offs)
        eas.append(s_.edge_attr); batch.append(np.full(s_.n, gi)); offs += s_.n
    # THE ENCODER'S DTYPE, not the global default. `build_conformer_conditions` runs under
    # float64 on purpose, and the encoder's weights are float32; following the global default
    # here fails with a bare 'mat1 and mat2 must have the same dtype'. The encoder's precision
    # is a property of its checkpoint, so its inputs follow IT.
    t = lambda a, d=dtype: torch.as_tensor(a, dtype=d, device=device)
    b = {'x': t(np.concatenate(xs)), 'struct': t(np.concatenate(sts)),
         'edge_index': t(np.concatenate(eis, axis=1), torch.long),
         'edge_attr': t(np.concatenate(eas)),
         'batch': t(np.concatenate(batch), torch.long),
         'n_graphs': len(samples)}
    if want_spd:
        length = max(s_.n for s_ in samples)
        b['spd'] = dense_spd_batch([s_.spd for s_ in samples], len(samples), length, device)
    return b


def load_encoder(ckpt_path: str = DEFAULT_CKPT, device='cpu') -> Dict:
    """The trained encoder, its input recipe, and the stamp that identifies it.

    Instantiates `MPNNEncoder` directly and takes only the `encoder.*` weights, rather than
    rebuilding the whole `ProbeModel`. The probe heads are an artefact of how the encoder was
    VALIDATED, not of what it IS, and binding to them would make the cache fail whenever the
    probe list changes. `n_heads` and `max_spd` are read off `spd_bias.weight`, whose shape is
    exactly `(max_spd + 2, n_heads)`, so they cannot drift from the checkpoint either.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck['state_dict']
    arm = ck.get('arm', 'mp+attn+spd')
    cfg = ARMS[arm]
    probe = _features('CCO', cfg['encoding'], ck['k'])
    node_dim, edge_dim = int(probe.x.shape[1]), int(probe.edge_attr.shape[1])

    n_heads, max_spd = 4, 8
    bias = sd.get('encoder.attn.0.spd_bias.weight')
    if bias is not None:
        max_spd, n_heads = int(bias.shape[0]) - 2, int(bias.shape[1])

    enc = MPNNEncoder(node_dim + ck['k'], edge_dim, hidden=ck['hidden'],
                      layers=ck['layers'], attention=ck['attention'],
                      n_heads=n_heads, max_spd=max_spd).to(device)
    enc.load_state_dict({kk[len('encoder.'):]: v for kk, v in sd.items()
                         if kk.startswith('encoder.')})
    enc.eval()
    return {'encoder': enc, 'dtype': next(enc.parameters()).dtype,
            'arm': arm, 'cfg': cfg, 'k': ck['k'],
            'hidden': ck['hidden'], 'layers': ck['layers'], 'attention': ck['attention'],
            'n_heads': n_heads, 'max_spd': max_spd,
            'node_dim': node_dim, 'edge_dim': edge_dim, 'device': device,
            'ckpt_path': os.path.abspath(ckpt_path), 'sha256': _fingerprint(ckpt_path)}


def embed(bundle: Dict, smiles: str, perm: Optional[np.ndarray] = None,
          z_tree: Optional[np.ndarray] = None):
    """Per-atom `h` and pooled `g` for one molecule.

    `perm` is `ConformerTorsions.spec.perm` -- the encoder-order -> tree-order map. Pass it
    and the returned `h` is in TREE order, ready to drop onto a conformer batch. Pass
    `z_tree` (= `spec.z`) as well and the alignment is ASSERTED rather than assumed; that
    assertion is the only thing standing between a reordering upstream and a silently
    mis-conditioned policy.
    """
    s = _features(smiles, bundle['cfg']['encoding'], bundle['k'])
    b = _batch([s], bundle['device'], bundle['cfg']['spd'], bundle['dtype'])
    with torch.no_grad():
        x = torch.cat([b['x'], b['struct']], dim=-1)
        h, g = bundle['encoder'](x, b['edge_index'], b['edge_attr'],
                                 b['batch'], b['n_graphs'], spd=b.get('spd'))
    z_enc = np.asarray(graph_from_smiles(smiles)[0]).astype(int)
    if perm is not None:
        perm = np.asarray(perm).astype(int)
        if perm.shape[0] != s.n:
            raise ValueError(
                f'{smiles}: spec.perm has {perm.shape[0]} entries but the encoder built '
                f'{s.n} atoms. The two paths disagree on the molecule, not just its order.')
        if z_tree is not None and not np.array_equal(z_enc[perm], np.asarray(z_tree).astype(int)):
            raise ValueError(
                f'{smiles}: ATOM ORDER MISMATCH. z_encoder[spec.perm] != spec.z, so the '
                f'permutation no longer aligns the encoder to the torsion tree. Storing '
                f'these embeddings would condition every atom on another atom, with no '
                f'shape error to catch it.\n  z_enc[perm] = {z_enc[perm].tolist()}\n'
                f'  spec.z      = {np.asarray(z_tree).astype(int).tolist()}')
        h = h[torch.as_tensor(perm, dtype=torch.long, device=h.device)]
        z_enc = z_enc[perm]
    return h.cpu(), g[0].cpu(), z_enc


def build_cache(smiles: Sequence[str], ckpt_path: str = DEFAULT_CKPT,
                out_path: str = None, device='cpu',
                level: str = 'torsion') -> Dict:
    """Encode every molecule once and write the stamped cache, keyed on canonical SMILES.

    THE KEY IS THE STEREO-SPECIFIC CANONICAL SMILES, not the skeleton. Diastereomers share a
    skeleton and must not share an embedding; keying on the wrong equivalence class is the
    mistake that leaked 67.9% of a held-out set out of the probe battery, and it is silent
    both times.
    """
    if out_path is None:
        out_path = str(artifact('conformer_embeddings.pt'))
    from rdkit import Chem
    from energies.conformer_torsions import ConformerTorsions

    bundle = load_encoder(ckpt_path, device)
    entries, skipped = {}, []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            skipped.append((smi, 'unparseable'))
            continue
        key = Chem.MolToSmiles(m)
        if key in entries:
            continue
        try:
            en = ConformerTorsions(smiles=smi, device='cpu', level=level)
            h, g, z_tree = embed(bundle, smi, perm=en.spec.perm, z_tree=en.spec.z)
        except Exception as exc:                              # noqa: BLE001 - reported below
            skipped.append((smi, f'{type(exc).__name__}: {exc}'))
            continue
        entries[key] = {'h': h, 'g': g, 'z_tree': torch.as_tensor(z_tree, dtype=torch.long),
                        'n': int(h.shape[0]), 'smiles': smi,
                        'n_torsions': int(en.spec.n_dof)}

    blob = {'format': FORMAT,
            'encoder': {kk: bundle[kk] for kk in
                        ('arm', 'k', 'hidden', 'layers', 'attention', 'n_heads',
                         'max_spd', 'node_dim', 'edge_dim', 'ckpt_path', 'sha256')},
            'entries': entries}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    torch.save(blob, out_path)
    print(f'wrote {out_path}: {len(entries)} molecules, '
          f'h dim {bundle["hidden"]}, g dim {2 * bundle["hidden"]}, '
          f'encoder {bundle["arm"]} @ {bundle["sha256"][:12]}')
    for smi, why in skipped:
        print(f'  SKIPPED {smi}: {why}')
    return blob


def load_cache(path: str, ckpt_path: Optional[str] = None) -> Dict:
    """Read a cache and REFUSE it if it came from a different encoder.

    A stale embedding fails exactly the way an unstamped replay row did: plausible numbers,
    wrong content, nothing raises. `ckpt_path` is optional only so a consumer that has no
    opinion can still read the header; pass it whenever you have it.
    """
    blob = torch.load(path, map_location='cpu', weights_only=False)
    fmt = blob.get('format')
    if fmt != FORMAT:
        raise ValueError(f'{path}: format {fmt!r}, expected {FORMAT!r}. Rebuild it with '
                         f'python -m models.encoder_cache.')
    if ckpt_path is not None:
        want = _fingerprint(ckpt_path)
        got = blob['encoder']['sha256']
        if want != got:
            raise ValueError(
                f'{path} was built from a DIFFERENT encoder.\n'
                f'  cache : {got}  ({blob["encoder"]["ckpt_path"]})\n'
                f'  wanted: {want}  ({os.path.abspath(ckpt_path)})\n'
                f'Embeddings from one encoder are meaningless to another. Rebuild:\n'
                f'    python -m models.encoder_cache --ckpt {ckpt_path} --out {path}')
    return blob


def lookup(blob: Dict, smiles: str, z_tree: Optional[np.ndarray] = None):
    """Fetch one molecule's `(h, g)` by SMILES, re-checking the atom order if given."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    key = Chem.MolToSmiles(m) if m is not None else smiles
    e = blob['entries'].get(key)
    if e is None:
        raise KeyError(f'{smiles} (canonical {key!r}) is not in the cache; it holds '
                       f'{len(blob["entries"])} molecules. Rebuild with this molecule '
                       f'included.')
    if z_tree is not None:
        zt = np.asarray(z_tree).astype(int)
        if not np.array_equal(e['z_tree'].numpy(), zt):
            raise ValueError(
                f'{smiles}: cached atom order does not match this batch.\n'
                f'  cache : {e["z_tree"].tolist()}\n  batch : {zt.tolist()}')
    return e['h'], e['g']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smiles', nargs='+', default=None)
    ap.add_argument('--smiles-file', default=None,
                    help='one SMILES per line; blank lines and # comments ignored')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--out', default=str(artifact('conformer_embeddings.pt')))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--level', default='torsion')
    a = ap.parse_args(argv)

    smis = list(a.smiles or [])
    if a.smiles_file:
        with open(a.smiles_file) as f:
            smis += [ln.strip() for ln in f
                     if ln.strip() and not ln.lstrip().startswith('#')]
    if not smis:
        ap.error('give --smiles and/or --smiles-file')
    build_cache(smis, a.ckpt, a.out, a.device, a.level)


if __name__ == '__main__':
    main()
