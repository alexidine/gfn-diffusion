"""configs/conformer_cond/make.py: a dry run renders the 3-rung ladder, the output is a pure
function of its inputs, an arm's name changes with anything that changes the run, every leak
and drift it exists to catch aborts generation by name, and the rendered sbatch -- the shared
crystal template with four substitutions -- runs each leg the way it says against a fake
cluster tree.

The generator reads the COMMITTED canonical configs; here the working-tree files stand in for
them, handed to `build` directly, so the test checks the files it ships beside.
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ES = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location('conformer_cond_make',
                                               ES / 'configs' / 'conformer_cond' / 'make.py')
make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make)

RUNGS = ['r0', 'r1', 'r2']
RDKIT = '2025.03.5'
PRIOR = 'conformer_prior_v2.pt'


def _load(rel):
    return yaml.safe_load((ES / rel).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def inputs():
    contract = make.energy_contract(
        (ES / 'energies' / 'conformer_torsions.py').read_text(encoding='utf-8'),
        (ES / 'conformer_modeller.py').read_text(encoding='utf-8'))
    return _load('configs/conformer_mk.yaml'), _load('configs/mk_dev.yaml'), contract


def _write_set(d, payload, inputs, n=6, conditions=None, **manifest):
    """A set builder output directory: the training conditions and a manifest that lists
    them, counts n train molecules in `conditions` conditions (default n), and was built
    under the base's energy kwargs."""
    d.mkdir(parents=True, exist_ok=True)
    cond = d / make.CONDITIONS_FILE
    cond.write_bytes(payload)
    params = inputs[2][0]
    kwargs = {k: v for k, v in inputs[0]['energy_config'].items()
              if k in params and k not in make.RUN_SURFACE_KEYS}
    man = {'format': make.MANIFEST_FORMAT, 'versions': {'rdkit': RDKIT},
           'artifacts': {make.CONDITIONS_FILE: {'sha256': make.sha256_of(cond),
                                                'bytes': cond.stat().st_size}},
           'energy': {'kwargs': kwargs},
           'rungs': {'train': {'molecules': n, 'conditions': n if conditions is None else conditions}}}
    man.update(manifest)
    (d / make.MANIFEST_FILE).write_text(json.dumps(man), encoding='utf-8')
    return man


@pytest.fixture
def local(inputs, tmp_path):
    """The local artifacts the index takes its sizes and hashes from."""
    sets, es = tmp_path / 'sets', tmp_path / 'es'
    for i, rung in enumerate(RUNGS):
        _write_set(sets / make.RUNGS[rung]['set_id'], b'c' * (100 + i), inputs,
                   n=make.RUNGS[rung]['n'])
    es.mkdir()
    (es / PRIOR).write_bytes(b'p' * 321)
    return sets, es


def _build(inputs, local, rungs=RUNGS, base=None, mk_dev=None, committed=lambda p: False):
    b, m, contract = inputs
    return make.build(copy.deepcopy(base if base is not None else b),
                      copy.deepcopy(mk_dev if mk_dev is not None else m), contract, rungs,
                      local_sets=local[0], local_es=local[1], committed=committed)


def _at_head_without(inputs, key):
    """`inputs` with the energy contract as a HEAD lacking `key`'s handler would parse it:
    the working-tree signature minus that parameter and its default."""
    b, m, (params, non_energy, defaults) = inputs
    assert key in params, f'{key} is not a working-tree ConformerTorsions parameter'
    return b, m, (params - {key}, non_energy, {k: v for k, v in defaults.items() if k != key})


def _refused(fn, *names):
    with pytest.raises(SystemExit) as e:
        fn()
    msg = str(e.value)
    for n in names:
        assert n in msg, f'{n!r} not named in: {msg}'
    return msg


def _set(cfg, dotted, value):
    node = cfg
    *head, last = dotted.split('.')
    for k in head:
        node = node[k]
    node[last] = value


# ----------------------------------------------------------------------------- dry run

def test_a_dry_run_renders_the_three_rung_ladder(inputs, local, tmp_path):
    arms = _build(inputs, local)
    assert [n.rsplit('_', 1)[0] for n in arms] == ['cc_r0_n6', 'cc_r1_n50', 'cc_r2_n500']
    out = tmp_path / 'out'
    out.mkdir()
    make.write(out, arms)
    for name, arm in arms.items():
        cfg = arm['cfg']
        written = yaml.safe_load((out / f'{name}.yaml').read_text(encoding='utf-8'))
        assert written == cfg and cfg['run_name'] == name
        assert cfg['molecules_path'] == (f"{make.CLUSTER_SETS}/{make.RUNGS[arm['rung']]['set_id']}"
                                         f"/{make.CONDITIONS_FILE}")
        assert cfg['energy_config']['internal_prior_path'] == f'{make.CLUSTER_SETS}/{PRIOR}'
        assert cfg['checkpoint_name'] == make.PLACEHOLDER and cfg['load_weights_only'] is False
        assert cfg['prior_path'] is None and cfg['test_molecules_path'] is None
        assert written['energy_config']['stereo_coeff'] == make.STEREO_COEFF   # the lock is on
    for path in out.rglob('*'):
        if path.is_file():
            text = path.read_text(encoding='utf-8')
            assert 'D:\\' not in text and 'C:\\' not in text and 'C:/' not in text, path
    rows = [r.split('\t') for r in (out / 'INDEX_a.tsv').read_text(encoding='utf-8').splitlines()]
    assert rows[0][:5] == ['arm', 'rung', 'start', 'warm_src', 'rdkit']
    assert [r[0] for r in rows[1:]] == list(arms)
    cond = local[0] / 'qm9_r0' / make.CONDITIONS_FILE
    assert rows[1][1:] == ['r0', 'fresh', '-', RDKIT,
                           f'qm9_r0/{make.CONDITIONS_FILE}', '100', make.sha256_of(cond),
                           PRIOR, '321', make.sha256_of(local[1] / PRIOR)]
    sbatch = (out / 'submit_conformer_cond_a.sbatch').read_text(encoding='utf-8')
    assert 'python -u conformer_modeller.py --config' in sbatch
    assert 'python -u train.py' not in sbatch and 'MONO_CLASS' not in sbatch
    assert 'PRIOR_BYTES' not in sbatch
    assert '#SBATCH --array=0-2' in sbatch


def test_generation_is_deterministic(inputs, local, tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f'out{k}'
        out.mkdir()
        make.write(out, _build(inputs, local))
        outs.append({p.relative_to(out).as_posix(): p.read_bytes()
                     for p in sorted(out.rglob('*')) if p.is_file()})
    assert outs[0] == outs[1]


# ----------------------------------------------------------------------------- arm identity

def _r0_name(inputs, local, **kw):
    return next(iter(_build(inputs, local, rungs=['r0'], **kw)))


def test_an_arm_is_renamed_by_any_change_to_its_run(inputs, local, monkeypatch):
    """The sbatch finds checkpoints by arm name, so a regenerated arm that runs differently
    must not share one with the arm it replaces."""
    name = _r0_name(inputs, local)
    assert _r0_name(inputs, local) == name                     # unchanged -> same arm
    base = copy.deepcopy(inputs[0])
    base['eval_period'] = 250
    assert _r0_name(inputs, local, base=base) != name           # a config value
    rungs = copy.deepcopy(make.RUNGS)
    rungs['r0']['epochs'] = 50_000
    monkeypatch.setattr(make, 'RUNGS', rungs)
    assert _r0_name(inputs, local) != name                      # a rung setting
    monkeypatch.undo()
    _write_set(local[0] / 'qm9_r0', b'C' * 100, inputs)          # same size, other bytes
    assert _r0_name(inputs, local) != name                      # the data


# ----------------------------------------------------------------------------- refusals

@pytest.mark.parametrize('key, value, named', [
    ('buffers.anchor_buffer.shape_path', 'D:\\crystal_datasets\\tiles.pt', 'shape_path'),
    ('buffers.anchor_buffer.shape_path', 'E:/data/tiles.pt', 'shape_path'),
    ('mlip_path', 'C:/models/esen.pt', 'mlip_path'),
    ('profiling.trace.outdir', 'E:/profiles', 'profiling.trace.outdir'),   # not a data key
])
def test_a_local_drive_path_aborts_generation(inputs, local, key, value, named):
    base = copy.deepcopy(inputs[0])
    _set(base, key, value)
    _refused(lambda: _build(inputs, local, base=base), named)


def test_a_relative_untracked_data_path_aborts_generation(inputs, local):
    base = copy.deepcopy(inputs[0])
    base['buffers']['anchor_buffer']['shape_path'] = 'tiles/anchor_tile.pt'
    _refused(lambda: _build(inputs, local, base=base),
             'shape_path', 'not committed at HEAD')
    # the same relative path to a COMMITTED file is what a cluster pull delivers
    _build(inputs, local, base=base, committed=lambda p: True)


def test_an_uncommitted_energy_config_key_aborts_generation(inputs, local):
    """The contract is parsed from HEAD, so a key the base carries whose handler is only in
    the working tree is outside it. This test used to add stereo_coeff to the base, when the
    working-tree ConformerTorsions did not take it; the stereo lock is now in the working-tree
    signature and the canonical declares the key, so the uncommitted handler is reproduced by
    parsing the contract without it (as HEAD reads until the lock is committed), with the set
    built by a builder on that code."""
    head = _at_head_without(inputs, 'stereo_coeff')
    assert inputs[0]['energy_config']['stereo_coeff'] == make.STEREO_COEFF
    _write_set(local[0] / 'qm9_r0', b'c' * 100, head)
    _refused(lambda: _build(head, local, rungs=['r0']), 'stereo_coeff', 'COMMITTED')
    # a misspelt key is outside the contract whatever is committed (the set rebuilt under the
    # full contract, so the energy-kwargs comparison passes)
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs)
    base = copy.deepcopy(inputs[0])
    base['energy_config']['stereo_coef'] = make.STEREO_COEFF
    _refused(lambda: _build(inputs, local, rungs=['r0'], base=base), "'stereo_coef'", 'COMMITTED')


def test_an_mk_dev_key_the_base_lacks_aborts_generation(inputs, local):
    mk = copy.deepcopy(inputs[1])
    mk['buffers']['prior_buffer']['a_new_live_key'] = 1
    _refused(lambda: _build(inputs, local, mk_dev=mk), 'buffers.prior_buffer.a_new_live_key')


def test_a_baseline_severity_violation_aborts_generation(inputs, local):
    base = copy.deepcopy(inputs[0])
    base['xcond_eval']['enabled'] = True
    _refused(lambda: _build(inputs, local, base=base), 'config_invariants',
             'conformer_set_xcond_is_off')


def test_a_temperature_other_than_its_log_aborts_generation(inputs, local):
    base = copy.deepcopy(inputs[0])
    base['energy_config']['temperature'] = 2.0
    _refused(lambda: _build(inputs, local, base=base), 'temperature', '10**log_temperature')


@pytest.mark.parametrize('value, named', [
    (0.0, '0.0'),
    (30.0, '30.0'),
    (None, 'absent'),                      # None = the key removed: the code default, lock off
])
def test_an_arm_off_the_stereo_lock_aborts_generation(inputs, local, value, named):
    """The base's stereo_coeff reaches check_protocol even when the set agrees with it: the
    set is built under the same base, so the energy-kwargs comparison passes and the refusal
    is the lock's own."""
    base = copy.deepcopy(inputs[0])
    if value is None:
        del base['energy_config']['stereo_coeff']
    else:
        base['energy_config']['stereo_coeff'] = value
    _write_set(local[0] / 'qm9_r0', b'c' * 100, (base, inputs[1], inputs[2]))
    _refused(lambda: _build(inputs, local, rungs=['r0'], base=base), 'cc_r0_n6',
             'stereo_coeff', named, 'STEREO_COEFF')


def test_a_duplicate_arm_aborts_generation(inputs, local, monkeypatch):
    rungs = dict(make.RUNGS)
    rungs['rx'] = dict(rungs['r0'])
    monkeypatch.setattr(make, 'RUNGS', rungs)
    _refused(lambda: _build(inputs, local, rungs=['r0', 'rx']), 'cc_r0_n6', 'cc_rx_n6')


def test_an_arm_name_that_is_a_suffix_of_another_aborts_generation():
    cfg = {'energy_config': {'energy_clip': 300.0}}
    arms = {'cc_r0_n6': {'cfg': {**cfg, 'run_name': 'a', 'x': 1}},
            'x_cc_r0_n6': {'cfg': {**cfg, 'run_name': 'b', 'x': 2}}}
    _refused(lambda: make.check_battery(arms), 'cc_r0_n6', 'x_cc_r0_n6')


def test_a_missing_local_artifact_aborts_generation(inputs, local):
    (local[0] / make.RUNGS['r1']['set_id'] / make.CONDITIONS_FILE).unlink()
    _refused(lambda: _build(inputs, local), 'qm9_r1', 'does not exist')


def test_a_set_without_a_manifest_aborts_generation(inputs, local):
    (local[0] / 'qm9_r0' / make.MANIFEST_FILE).unlink()
    _refused(lambda: _build(inputs, local), 'qm9_r0', make.MANIFEST_FILE, 'does not exist')


def test_a_conditions_file_its_manifest_does_not_list_aborts_generation(inputs, local):
    man = _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs)
    (local[0] / 'qm9_r0' / make.CONDITIONS_FILE).write_bytes(b'x' * 100)
    assert man['artifacts'][make.CONDITIONS_FILE]['bytes'] == 100
    _refused(lambda: _build(inputs, local), 'qm9_r0', 'not the file the builder wrote')


def test_a_manifest_of_another_format_aborts_generation(inputs, local):
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs, format='conformer_set_v0')
    _refused(lambda: _build(inputs, local), 'conformer_set_v0')


def test_a_set_built_under_other_energy_kwargs_aborts_generation(inputs, local):
    kwargs = {k: v for k, v in inputs[0]['energy_config'].items()
              if k in inputs[2][0] and k not in make.RUN_SURFACE_KEYS}
    # a builder config without energy_clip ran at the committed default (None), not 300
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs,
               energy={'kwargs': {k: v for k, v in kwargs.items() if k != 'energy_clip'}})
    _refused(lambda: _build(inputs, local), 'energy_clip', 'Rebuild the set')
    # a set built with the lock off -- explicitly, or by a config that did not declare it (the
    # builder then ran at the committed default, 0) -- is not the arm's target
    assert kwargs['stereo_coeff'] == make.STEREO_COEFF and inputs[2][2]['stereo_coeff'] == 0.0
    for off in ({**kwargs, 'stereo_coeff': 0.0},
                {k: v for k, v in kwargs.items() if k != 'stereo_coeff'}):
        _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs, energy={'kwargs': off})
        _refused(lambda: _build(inputs, local), "('stereo_coeff', 300.0, 0.0)", 'Rebuild the set')
    # a builder on uncommitted code passed a parameter the committed signature lacks. This
    # used to be stereo_coeff against the full contract, before the working-tree signature took
    # it; the committed signature lacking it is now reproduced by parsing the contract without
    # it, while the builder's kwargs (the working tree's) carry it
    head = _at_head_without(inputs, 'stereo_coeff')
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs, energy={'kwargs': kwargs})
    _refused(lambda: _build(head, local, rungs=['r0']), 'stereo_coeff', 'not a committed parameter')
    # an absent key whose committed default equals the arm's value is the same set
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs,
               energy={'kwargs': {k: v for k, v in kwargs.items() if k != 'rho_wall'}})
    assert inputs[0]['energy_config']['rho_wall'] == inputs[2][2]['rho_wall']
    _build(inputs, local)


def test_buffer_caps_below_the_per_condition_floor_abort_generation(inputs, local, monkeypatch):
    rungs = dict(make.RUNGS)
    rungs['r3'] = dict(n=5000, set_id='qm9_r3', phase1_max_steps=20_000, epochs=200_000)
    monkeypatch.setattr(make, 'RUNGS', rungs)
    _write_set(local[0] / 'qm9_r3', b'c' * 100, inputs, n=5000)
    _refused(lambda: _build(inputs, local, rungs=['r3']), 'cc_r3_n5000', 'per condition')


def test_the_floors_count_conditions_not_molecules(inputs, local):
    """Stereoisomers are separate conditions: 6 molecules in 3000 conditions put the
    prior-rebuild floor at 20 x 3000 = 60,000 rows, above the 25,000 the caps rebuild to.
    At 6 conditions every floor clears."""
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs, n=6, conditions=3000)
    _refused(lambda: _build(inputs, local, rungs=['r0']), 'prior_rebuild', '3000 conditions')


def test_a_set_of_another_rung_size_aborts_generation(inputs, local):
    _write_set(local[0] / 'qm9_r0', b'c' * 100, inputs, n=7)
    _refused(lambda: _build(inputs, local, rungs=['r0']), 'counts 7 train molecules',
             'the rung is 6 molecules')


def test_buffer_caps_above_the_vram_budget_abort_generation(inputs, local, monkeypatch):
    monkeypatch.setattr(make, 'BUFFER_CAPS', dict(make.BUFFER_CAPS, prior=400_000))
    _refused(lambda: _build(inputs, local, rungs=['r0']), 'cc_r0_n6', 'buffer budget')


def test_sidecars_above_the_disk_budget_abort_generation(inputs, local, monkeypatch):
    arms = _build(inputs, local)
    total = sum(make.sidecar_disk_bytes(a['cfg']) for a in arms.values())
    monkeypatch.setattr(make, 'DISK_BUDGET_BYTES', total - 1)
    _refused(lambda: make.check_battery(arms), 'disk budget')


@pytest.mark.parametrize('dotted, value, named', [
    ('fwd.emp_z_persistent', 0.0, 'emp_z_persistent'),
    ('fwd.reward_grads', 0.0, 'reward_grads'),
    ('fwd.path_grad_last_k', 0, 'path_grad_last_k'),
    ('bwd.freeze_z', 0.0, 'freeze_z'),
    ('bwd.beta', 10.0, 'beta'),
    ('replay.tb_z_source', 'learned', 'tb_z_source'),
])
def test_a_tb_seat_off_the_baseline_aborts_generation(inputs, local, dotted, value, named):
    base = copy.deepcopy(inputs[0])
    _set(make.stage(base, 'tb_conditioning')['loss_coeffs'], dotted, value)
    _refused(lambda: _build(inputs, local, rungs=['r0'], base=base), named, 'baseline seat')


def test_a_seat_key_dropped_from_the_stage_is_judged_at_the_base_value(inputs, local):
    base = copy.deepcopy(inputs[0])
    del make.stage(base, 'tb_conditioning')['loss_coeffs']['fwd']['emp_z_persistent']
    assert base['fwd_loss_coeffs']['emp_z_persistent'] == 0
    _refused(lambda: _build(inputs, local, rungs=['r0'], base=base), 'emp_z_persistent')


def test_a_rollout_draw_off_the_baseline_aborts_generation(inputs, local):
    base = copy.deepcopy(inputs[0])
    base['condition_log_z']['rollout_condition_draw'] = 'iid'
    with pytest.raises(AssertionError, match='iid'):
        _build(inputs, local, rungs=['r0'], base=base)


def test_a_template_without_the_substituted_span_aborts_generation():
    drifted = make.fin.SBATCH.replace('python -u train.py --config', 'python -u train2.py --config')
    _refused(lambda: make.render_sbatch(1, template=drifted), 'train.py launch')
    drifted = make.fin.SBATCH.replace('{{print $6}}', '{{print $7}}')
    _refused(lambda: make.render_sbatch(1, template=drifted), 'prior index columns')


# ----------------------------------------------------------------------------- the sbatch

def _bash():
    exe = shutil.which('bash')
    if exe is None or 'system32' in exe.lower():       # WSL's launcher, not a POSIX bash
        pytest.skip('no POSIX bash on PATH')
    return exe


def _posix(p):
    p = Path(p).resolve()
    if os.name == 'nt':
        s = p.as_posix()
        return '/' + s[0].lower() + s[2:]
    return p.as_posix()


def test_the_rendered_sbatch_parses(tmp_path):
    script = tmp_path / 'submit.sbatch'
    script.write_text(make.render_sbatch(3), encoding='utf-8', newline='\n')
    r = subprocess.run([_bash(), '-n', _posix(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


_SHIMS = {
    'module': '',
    'nvidia-smi': '',
    'scontrol': '',
    'sacct': '',
    'stdbuf': '',
    'srun': 'echo "SRUN $*"\nif [ -n "${FAKE_UNRECOVERABLE:-}" ]; then echo "UNRECOVERABLE: fake"; fi\n',
}


def _deploy(tmp_path, arms, local):
    """A fake cluster tree under tmp_path, reused across calls: the battery directory the
    generator writes (rewritten each call), the data root holding copies of the local
    artifacts, the checkpoint directory, and command shims."""
    root = tmp_path / 'proj'
    arms_dir = root / 'gfn-diffusion' / 'energy_sampling' / 'configs' / make.BATTERY
    arms_dir.mkdir(parents=True, exist_ok=True)
    make.write(arms_dir, arms)
    data, ckpts, shims = tmp_path / 'data', tmp_path / 'ckpts', tmp_path / 'bin'
    (data / 'qm9_r0').mkdir(parents=True, exist_ok=True)
    shutil.copy(local[0] / 'qm9_r0' / make.CONDITIONS_FILE, data / 'qm9_r0' / make.CONDITIONS_FILE)
    shutil.copy(local[1] / PRIOR, data / PRIOR)
    ckpts.mkdir(exist_ok=True)
    shims.mkdir(exist_ok=True)
    for name, body in _SHIMS.items():
        path = shims / name
        path.write_text('#!/bin/sh\n' + body, encoding='utf-8', newline='\n')
        path.chmod(0o755)
    text = make.render_sbatch(len(arms), ckpts=_posix(ckpts), data=_posix(data))
    old = 'PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond'
    assert text.count(old) == 1
    script = tmp_path / 'submit.sbatch'
    script.write_text(text.replace(old, f'PROJECT_ROOT={_posix(root)}'), encoding='utf-8',
                      newline='\n')
    return dict(script=script, data=data, ckpts=ckpts, shims=shims, index=arms_dir / 'INDEX_a.tsv',
                logs=arms_dir / 'joblogs', arm=next(iter(arms)))


@pytest.fixture
def cluster(inputs, local, tmp_path):
    return _deploy(tmp_path, _build(inputs, local, rungs=['r0']), local)


def _run(cluster, **env):
    e = dict(os.environ, SLURM_ARRAY_TASK_ID='0', SLURM_JOB_ID='777', SLURM_NODELIST='node0',
             SHIMS=_posix(cluster['shims']), **env)
    r = subprocess.run([_bash(), '-c', 'export PATH="$SHIMS:$PATH"; exec bash "$0"',
                        _posix(cluster['script'])], capture_output=True, text=True, env=e)
    resolved = cluster['logs'] / f'{cluster["arm"]}_777.yaml'
    cfg = yaml.safe_load(resolved.read_text(encoding='utf-8')) if resolved.exists() else None
    log = cluster['logs'] / f'{cluster["arm"]}_777.trainlog'
    return r, cfg, (log.read_text(encoding='utf-8') if log.exists() else '')


def _touch(d, name):
    (d / name).write_bytes(b'')


def _stem(arm, slug='abc123'):
    return f'confc_{arm}_conformer_torsions-noprior-T1-{slug}'


def test_fresh_launch_resolves_a_null_checkpoint(cluster):
    r, cfg, log = _run(cluster)
    assert r.returncode == 0, r.stderr
    assert 'FRESH' in r.stdout
    assert cfg['checkpoint_name'] is None and cfg['load_weights_only'] is False
    assert 'SRUN' in log and 'python -u conformer_modeller.py --config' in log
    # the index's RDKit version reaches the in-container guard
    assert f"assert rdkit.__version__ == '{RDKIT}'" in log


def test_resubmit_resumes_the_arms_own_running_checkpoint(cluster):
    own = f'{_stem(cluster["arm"])}_running.pt'
    _touch(cluster['ckpts'], own)
    r, cfg, _log = _run(cluster)
    assert r.returncode == 0, r.stderr
    assert 'RESUME' in r.stdout
    assert cfg['checkpoint_name'] == own and cfg['load_weights_only'] is False


def test_a_regenerated_arm_neither_resumes_nor_skips_on_the_old_arms_files(inputs, local,
                                                                           tmp_path, monkeypatch):
    old = _deploy(tmp_path, _build(inputs, local, rungs=['r0']), local)
    _touch(old['ckpts'], f'{_stem(old["arm"])}_running.pt')
    _touch(old['ckpts'], f'{old["arm"]}.dead')
    rungs = copy.deepcopy(make.RUNGS)
    rungs['r0']['epochs'] = 50_000
    monkeypatch.setattr(make, 'RUNGS', rungs)
    new = _deploy(tmp_path, _build(inputs, local, rungs=['r0']), local)
    assert new['arm'] != old['arm'] and new['ckpts'] == old['ckpts']
    r, cfg, log = _run(new)
    assert r.returncode == 0, r.stderr
    assert 'FRESH' in r.stdout and 'skipping' not in r.stdout
    assert cfg['checkpoint_name'] is None and 'SRUN' in log


def test_ck_step_restarts_from_the_archive_and_refuses_a_missing_sidecar(cluster):
    stem = _stem(cluster['arm'])
    _touch(cluster['ckpts'], f'{stem}_running.pt')
    _touch(cluster['ckpts'], f'{stem}_step50.pt')
    r, _cfg, log = _run(cluster, CK_STEP='50')
    assert r.returncode == 1 and 'no frozen buffers sidecar' in r.stderr
    assert 'SRUN' not in log
    _touch(cluster['ckpts'], f'{stem}_step50_buffers.pt')
    r, cfg, _log = _run(cluster, CK_STEP='50')
    assert r.returncode == 0, r.stderr
    assert cfg['checkpoint_name'] == f'{stem}_step50.pt'


def test_ck_step_refuses_an_ambiguous_match(cluster):
    for slug in ('aaa', 'bbb'):
        _touch(cluster['ckpts'], f'{_stem(cluster["arm"], slug)}_step50.pt')
    r, _cfg, log = _run(cluster, CK_STEP='50')
    assert r.returncode == 1 and '2 matches' in r.stderr
    assert 'SRUN' not in log


def test_an_artifact_of_the_wrong_size_stops_the_task_before_srun(cluster):
    path = cluster['data'] / 'qm9_r0' / make.CONDITIONS_FILE
    path.write_bytes(path.read_bytes()[:-1])
    r, _cfg, log = _run(cluster)
    assert r.returncode == 1
    assert 'FATAL' in r.stderr and 'bytes' in r.stderr and make.CONDITIONS_FILE in r.stderr
    assert 'SRUN' not in log


def test_an_artifact_of_the_right_size_and_other_bytes_stops_the_task_before_srun(cluster):
    path = cluster['data'] / PRIOR
    path.write_bytes(b'q' * path.stat().st_size)
    r, _cfg, log = _run(cluster)
    assert r.returncode == 1
    assert 'FATAL' in r.stderr and 'sha256' in r.stderr and PRIOR in r.stderr
    assert 'SRUN' not in log


def test_an_index_row_without_artifacts_stops_the_task_before_srun(cluster):
    lines = cluster['index'].read_text(encoding='utf-8').splitlines()
    lines[1] = '\t'.join(lines[1].split('\t')[:5])
    cluster['index'].write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    r, _cfg, log = _run(cluster)
    assert r.returncode == 1 and 'no artifact' in r.stderr
    assert 'SRUN' not in log


def test_a_dead_arm_is_skipped_and_an_unrecoverable_leg_marks_it(cluster):
    r, _cfg, _log = _run(cluster, FAKE_UNRECOVERABLE='1')
    assert r.returncode == 0, r.stderr
    assert (cluster['ckpts'] / f'{cluster["arm"]}.dead').exists()
    r, _cfg, _log = _run(cluster)
    assert r.returncode == 0 and 'skipping' in r.stdout
