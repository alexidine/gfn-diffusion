"""``build_conformer_set.py --workers N`` writes what the serial build writes.

What it would catch, at level `full`, MMFF94, the stereo lock of configs/conformer_mk.yaml, CPU,
float64 builds, with the frozen encoder baked: a parallel walk that orders molecules or
isomers differently, stops at another molecule (the rung ends before the slice does, so
molecules already started past the end must be discarded), loses or duplicates a refusal,
or returns a member that is not bit-identical after crossing a process boundary. Every artifact
is compared byte for byte, and the manifest field by field without its argv, time stamp,
duration and git status.

The encoder checkpoint is untracked: point GFN_ENCODER_CKPT at it from a worktree.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/conformer/test_build_conformer_set_workers.py
"""
import csv
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parents[2]            # energy_sampling/
# QM9 rows by file index: ordinary molecules, cages, stereocentres, a nitrile and an alkyne,
# and refusals (water: lt4_atoms; methane and methanol)
ROWS = [(25407, 'CCCO'), (25604, 'C1OC2C3CC1C2O3'), (33732, 'CC1COC2=C1OC=C2'),
        (37992, 'CC(=O)N1C2CC(=O)C12'), (39946, 'CCC1OCOCC=C1'), (45037, 'OC1CC2C1OCC2O'),
        (52508, 'N#CC1CCC11CCC1'), (57258, 'CC1(C#C)C(O)C1CO'), (60175, 'CO'),
        (62851, 'CCC(C)O'), (65128, 'CC1COC1(C)CC#N'), (78979, 'CCC1C2CC1(C)O2'),
        (91408, 'O=C1CNCCOCO1'), (95994, 'OCC1(CC1O)C1CC1'), (108315, 'C'),
        (114806, 'COC1CC2(CO2)C1C'), (116340, 'O'), (125600, 'CCOCCOC1CC1')]
VOLATILE = ('argv', 'created_utc', 'seconds', 'git')


@pytest.fixture(autouse=True)
def _restore_torch_state():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def _encoder_ckpt():
    from models import encoder_cache
    p = Path(os.environ.get('GFN_ENCODER_CKPT', encoder_cache.DEFAULT_CKPT))
    if not p.exists():
        pytest.fail(f'encoder checkpoint not found at {p}: set GFN_ENCODER_CKPT')
    return p


def _build(tmp_path, workers):
    import build_conformer_set as bcs
    src = tmp_path / 'slice.csv'
    if not src.exists():
        with open(src, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset_index', 'smiles'])
            w.writerows(ROWS)
    out = tmp_path / f'w{workers}'
    bcs.main(['--out-dir', str(out), '--source', str(src), '--n-train', '6',
              '--n-heldout', '2', '--heldout-permille', '250',
              '--encoder-ckpt', str(_encoder_ckpt()), '--workers', str(workers)])
    with open(out / 'manifest.json', encoding='utf-8') as f:
        return out, json.load(f)


def test_workers_write_the_serial_build(tmp_path):
    serial, m1 = _build(tmp_path, 1)
    pooled, m2 = _build(tmp_path, 2)
    # the rung ends inside the slice on the training side, so the pooled walk had molecules
    # in flight past its end
    assert m1['split']['attempted']['train'] < m1['split']['keys']['train']
    names = sorted(p.name for p in serial.iterdir() if p.name != 'manifest.json')
    assert names == sorted(p.name for p in pooled.iterdir() if p.name != 'manifest.json')
    assert 'conditions_train.pt' in names and 'rejections.tsv' in names
    for name in names:
        assert (serial / name).read_bytes() == (pooled / name).read_bytes(), name
    for m in (m1, m2):
        for k in VOLATILE:
            m.pop(k)
    assert m1 == m2
    assert m1['reasons']['molecule'], 'the slice should refuse some molecule'
    assert m1['references']['reference'] == {'embedded': sum(m1['rungs'][s]['conditions']
                                                             for s in m1['rungs'])}
