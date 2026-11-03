"""RDF distances from every sample at every optimisation depth to every reference minimum (the benchmark protocol's
coverage score, ``docs/design/generator_benchmark_protocol.md`` step 2.1.1).

Structures: the reference minima (``day_landscape``'s reference folder REF_DIR after ``protocol_refine`` wrote its
deepest depth there: ``reference.csv``, ``refine_<energy>.csv``, ``depth_<max>/*.extxyz``) and each source's samples at
each depth (``protocol_refine``'s labelled ``depth_<k>/*.extxyz``), all rebuilt as P1 cells of the constant conformer
(``p1_relax.p1_crystal``) and given envwise RDFs in one ``rdf_embed.rdfs`` call, so every structure shares one channel
labelling. Distances are ``crystal_rdf.compute_rdf_distance`` (mean over active channels of the per-channel earth
mover's distance), computed on the device in blocks against the reference only. CONFORMER defaults to REF_DIR's
``conformer.txt``.

usage (licensed CSD Python API; GPU):
    python -m eval.day_bench.protocol_score OUT_NPZ REFCODE REF_DIR LABEL=DIR[+DIR...] [LABEL=DIR ...]
        [--conformer SPEC] [--energy exp6] [--device cuda]

Writes OUT_NPZ: ``D`` [structures, minima], ``D_ref`` [minima, minima], per structure ``label``, ``folder``, ``depth``,
``fname``, ``path``; per minimum ``ref_file``, ``ref_fitdma_rel`` and ``ref_corrected_rel`` (kJ/mol above the
landscape's lowest under Day's two rankings; NaN where the correction does not cover it), ``ref_e`` (the benchmark
energy at the deepest depth), ``ref_f`` (largest generalised force there), ``ref_sg`` and ``ref_zprim`` (molecules per
primitive cell).
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
    ap.add_argument('ref_dir')
    ap.add_argument('sources', nargs='+')
    ap.add_argument('--conformer', default=None)
    ap.add_argument('--energy', default='exp6')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    conformer = a.conformer or open(os.path.join(a.ref_dir, 'conformer.txt')).read().strip()
    from ccdc import io as ccdc_io
    from eval.day_bench.day_landscape import load_reference
    torch.set_num_threads(2)
    if str(a.device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(0.3)
    ref = load_reference(a.refcode, conformer, ccdc_io.EntryReader('CSD'))

    day = {r['file']: r for r in csv.DictReader(open(os.path.join(a.ref_dir, 'reference.csv')))}
    refine = [r for r in csv.DictReader(open(os.path.join(a.ref_dir, f'refine_{a.energy}.csv')))
              if r['rebuilt'] == 'True']
    deep = max(int(re.search(r'depth_(\d+)', p).group(1)) for p in glob.glob(os.path.join(a.ref_dir, 'depth_*')))
    num = lambda v: float(v) if v != '' else np.nan
    conv = [r for r in refine if r['file'] in day]
    entries = [dict(kind='ref', path=os.path.join(a.ref_dir, f'depth_{deep}', r['file']), file=r['file'],
                    fitdma_rel=num(day[r['file']]['fitdma_rel']), corrected_rel=num(day[r['file']]['corrected_rel']),
                    ref_e=float(r[f'e_{deep}']), ref_f=float(r[f'f_{deep}']), ref_sg=int(day[r['file']]['day_sg']),
                    ref_zprim=int(day[r['file']]['z_prim'])) for r in conv]
    print(f'reference: {len(conv)} of {len(day)} minima rebuilt and relaxed, read at depth {deep}', flush=True)
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
        **{f'ref_{k}': np.array([e[k] for e in entries[:n_ref]]) for k in ('file', 'path')},
        **{f'ref_{k}': np.array([e[k] for e in entries[:n_ref]]) for k in ('fitdma_rel', 'corrected_rel')},
        **{k: np.array([e[k] for e in entries[:n_ref]]) for k in ('ref_e', 'ref_f', 'ref_sg', 'ref_zprim')})
    print(f'wrote {a.out_npz}', flush=True)


if __name__ == '__main__':
    main()
