"""RDF distances from every sample at every optimisation depth to every reference minimum (the benchmark protocol's
coverage score, ``docs/design/generator_benchmark_protocol.md`` step 2.1.1).

Structures: the reference minima (``converge``'s ``converged_<energy>/`` under REF_DIR, ``kind`` sample) and each
source's samples at each depth (``protocol_refine``'s labelled ``depth_<k>/*.extxyz``), all rebuilt as P1 cells of the constant
conformer (``p1_relax.p1_crystal``) and given envwise RDFs in one ``rdf_embed.rdfs`` call, so every structure shares one
channel labelling. Distances are ``crystal_rdf.compute_rdf_distance`` (mean over active channels of the per-channel
earth mover's distance), computed on the device in blocks against the reference only.

usage (licensed CSD Python API; GPU):
    python -m eval.day_bench.protocol_score OUT_NPZ REFCODE CONFORMER REF_DIR LABEL=DIR[+DIR...] [LABEL=DIR ...]
        [--energy exp6] [--device cuda]

Writes OUT_NPZ: ``D`` [structures, minima], ``D_ref`` [minima, minima], per structure ``label``, ``folder``, ``depth``,
``fname``, ``path``; per minimum ``ref_file``, ``ref_energy`` (Day's DFT-corrected), ``ref_e`` (the benchmark energy
after convergence), ``ref_sg``.
"""
import argparse
import csv
import glob
import os
import re
import time
import warnings

import numpy as np
import torch
from ase.io import read

from eval.day_bench.p1_relax import p1_crystal
from eval.day_bench.rdf_embed import rdfs
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance

BLOCK = 64  # structures per distance block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_npz')
    ap.add_argument('refcode')
    ap.add_argument('conformer')
    ap.add_argument('ref_dir')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--energy', default='exp6')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    from ccdc import io as ccdc_io
    from eval.day_bench.day_landscape import load_reference
    torch.set_num_threads(2)
    if str(a.device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.3)
    ref = load_reference(a.refcode, a.conformer, ccdc_io.EntryReader('CSD'))

    first = {r['file']: r for r in csv.DictReader(open(os.path.join(a.ref_dir, f'p1_relax_{a.energy}.csv')))}
    conv = [r for r in csv.DictReader(open(os.path.join(a.ref_dir, f'converge_{a.energy}.csv')))
            if r.get('calls') and r['kind'] == 'sample']
    entries = [dict(kind='ref', path=os.path.join(a.ref_dir, f'converged_{a.energy}', r['file']), file=r['file'],
                    ref_energy=float(first[r['file']]['ref_energy']), ref_e=float(r['e_after']),
                    ref_sg=int(first[r['file']]['day_sg'])) for r in conv]
    for spec in a.sources:
        lab, dirs = spec.split('=', 1)
        for d in dirs.split('+'):
            for p in sorted(glob.glob(os.path.join(d, 'depth_*', '*.extxyz'))):
                entries.append(dict(kind='sample', label=lab, folder=os.path.basename(os.path.normpath(d)),
                                    depth=int(re.search(r'depth_(\d+)', p).group(1)), file=os.path.basename(p),
                                    path=p))
    t0 = time.time()
    crystals = []
    for e in entries:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            c, why = p1_crystal(read(e['path']), ref, identifier=e['file'])
        assert c is not None, f'{e["path"]}: {why}'
        crystals.append(c)
    print(f'{len(entries)} structures rebuilt in {time.time() - t0:.0f} s ({len(conv)} reference minima)', flush=True)
    rdf, bins = rdfs(crystals, device=a.device, chunk=32 if str(a.device).startswith('cuda') else 4)  # 4 on CPU: rdf_embed.CHUNK
    assert torch.isfinite(rdf).all(), 'non-finite RDF'
    print(f'RDFs {tuple(rdf.shape)} in {time.time() - t0:.0f} s', flush=True)
    n_ref = len(conv)
    r_ref = rdf[:n_ref].to(a.device)
    bins = bins.to(a.device)
    D = np.empty((len(entries), n_ref), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(entries), BLOCK):
            blk = rdf[i:i + BLOCK].to(a.device)
            D[i:i + len(blk)] = compute_rdf_distance(blk[:, None], r_ref[None], bins).cpu().numpy()
    print(f'distances in {time.time() - t0:.0f} s', flush=True)
    s = [e for e in entries if e['kind'] == 'sample']
    np.savez_compressed(
        a.out_npz, D=D[n_ref:], D_ref=D[:n_ref], label=np.array([e['label'] for e in s]),
        folder=np.array([e['folder'] for e in s]), depth=np.array([e['depth'] for e in s]),
        fname=np.array([e['file'] for e in s]), path=np.array([e['path'] for e in s]),
        ref_file=np.array([e['file'] for e in entries[:n_ref]]), ref_path=np.array([e['path'] for e in entries[:n_ref]]),
        ref_energy=np.array([e['ref_energy'] for e in entries[:n_ref]]),
        ref_e=np.array([e['ref_e'] for e in entries[:n_ref]]), ref_sg=np.array([e['ref_sg'] for e in entries[:n_ref]]))
    print(f'wrote {a.out_npz}', flush=True)


if __name__ == '__main__':
    main()
