r"""conformer_cond -- the conditional-conformer run ladder on QM9 subsets (level `full`, force
field `mmff`): one arm per rung, each a FRESH run that trains its own phase 1 and then phase 2
in one process. Cluster battery.

    python configs/conformer_cond/make.py                  # every rung in RUNGS
    python configs/conformer_cond/make.py --rungs r0,r1

WHAT IT READS. Both canonical configs from git HEAD, never the working tree: the base is
`git show HEAD:energy_sampling/configs/conformer_mk.yaml`, and the committed mk_dev is read to
hold the base to every key mk_dev carries (CRYSTAL_ONLY_KEYS aside). The energy contract --
the ConformerTorsions.__init__ parameters with their defaults, and
conformer_modeller._NON_ENERGY_KEYS -- is parsed from the COMMITTED sources as well, so an
energy_config key whose handler is uncommitted is refused here rather than at every arm's
construction. It refuses outright when git cannot produce a file, and warns on a dirty tree.
Per rung it reads the set builder's (build_conformer_set.py) output directory
LOCAL_SETS/<set_id>/: CONDITIONS_FILE, and MANIFEST_FILE, which must be format
MANIFEST_FORMAT, list CONDITIONS_FILE with the file's own sha256 and bytes, count the rung's n molecules on
its train side, name the RDKit version it was built under, and carry energy kwargs whose
effective values (committed defaults filling an absent key) equal the arm's, stereo_coeff
included, so a set built with the lock off or at another coefficient is refused. The manifest's
train CONDITIONS (stereoisomers are separate conditions) set the per-condition floors. The builder's held-out and prior files are
not shipped: ConformerModeller builds no held-out set, and the prior dataset is drawn from
the fitted InternalPrior (prior_path null).

WHAT EACH ARM IS. The committed conformer_mk with:
  - run_name `cc_<rung>_n<M>_<digest>`, tag TAG, checkpoints_dir the cluster root. <digest>
    is arm_digest: 8 hex of sha256 over the finished config (run_name aside), every
    artifact's (path, bytes, sha256) and the builder's RDKit version. The sbatch finds an
    arm's checkpoints and its dead sentinel by the arm name, so a regenerated arm that
    differs in any of those is a new run with no checkpoint, and an unchanged one resumes;
  - checkpoint_name WARM_CHECKPOINT_PLACEHOLDER, which the sbatch resolves: YAML null on a
    first launch (FRESH), the arm's own _running.pt on a resubmit (RESUME), its _step<N>.pt
    under CK_STEP=<N>. load_weights_only is false on every leg, so a resumed leg is a full
    load;
  - molecules_path and energy_config.internal_prior_path as absolute cluster paths under
    CLUSTER_SETS; prior_path and test_molecules_path null;
  - the rung's phase-1 bound (train_prior max_steps), epochs, and buffer caps;
  - tb_conditioning's tracker feed, flags.update_log_z = UPDATE_LOG_Z;
  - cuda_memory_fraction CUDA_MEMORY_FRACTION.

WHAT IT REFUSES, naming the offender: a drive-letter or backslash path anywhere in an arm; a
relative data path not committed at HEAD; an energy_config key outside the committed energy
contract; an mk_dev key the base lacks that is not in CRYSTAL_ONLY_KEYS; a protocol, stage,
TB-seat, tripwire, energy-clip, stereo-lock (STEREO_COEFF, declared) or temperature setting
other than check_protocol's; a condition set whose
manifest is missing, of another format, disagrees with its file, or was built under other
energy kwargs; two arms equal apart from run_name; an arm name that is a suffix of another
(the sbatch's `*${ARM}_*` globs would cross); a missing local artifact; buffer caps below the
per-condition floors or above the VRAM budget; buffer sidecars above the disk budget; and any
config_invariants violation, BASELINE included.

WHAT IT WRITES. `<arm>.yaml` per rung, INDEX_a.tsv (arm, rung, start, warm_src, rdkit, then
one (path under CLUSTER_SETS, bytes, sha256) triple per artifact),
submit_conformer_cond_a.sbatch and joblogs/.gitkeep. The sbatch is
configs/final_sep19/make.py::SBATCH with four asserted substitutions (conformer_template):
the index's prior columns become its rdkit column; the crystal MXtalTools wall checks and
single-prior byte check become DATA_GUARD, a bytes and sha256 check over every artifact in the
index row; the inline Niggli assertion becomes an RDKit-version, MMFF and
mxtaltools.conformers import guard; and the launch runs conformer_modeller.py. Everything
else -- the dead-arm sentinel, RESUME and CK_STEP, the placeholder substitution, the
nvidia-smi and sacct sidecars -- is the shared template's. A template that no longer carries
a substituted span exactly once refuses generation.

THE DATA, staged by hand before submission: each rung's `<set_id>/conditions_train.pt` and
the fitted InternalPrior, copied to CLUSTER_SETS with the same relative paths. The sizes and
hashes written into the index are the local files'.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                      # energy_sampling/configs
ES = ROOT.parent                        # energy_sampling/
REPO = ES.parent                        # gfn_diffusion/
TAG = 'confc'
BATTERY = 'conformer_cond'
LEG = 'a'
WALL = '2-00:00:00'
CANONICAL = 'energy_sampling/configs/conformer_mk.yaml'
MK_DEV = 'energy_sampling/configs/mk_dev.yaml'
TORSIONS_SRC = 'energy_sampling/energies/conformer_torsions.py'
MODELLER_SRC = 'energy_sampling/conformer_modeller.py'
PROTOCOL = 'conformer_conditional_tb'
STAGES = ['train_prior', 'tb_conditioning']

CLUSTER_SETS = '/scratch/mk8347/data/conformer_sets'
LOCAL_SETS = pathlib.Path(r'D:\crystal_datasets\conformer_sets')
#: the set builder's training conditions (cfg:molecules_path) and its manifest
CONDITIONS_FILE = 'conditions_train.pt'
MANIFEST_FILE = 'manifest.json'
MANIFEST_FORMAT = 'conformer_set_v1'
CUDA_MEMORY_FRACTION = 0.97
ENERGY_CLIP = 300.0                     # owner 2026-09-26; one value across the ladder
#: energy_config.stereo_coeff every arm carries, declared (an absent key is the code default
#: 0, the lock off). Owner decision: stereoisomers are distinct conditions pinned by tagged
#: SMILES with the lock on. 300 kcal/mol is the coefficient energies/stereo_lock.py::MIN_MARGIN
#: is set against.
STEREO_COEFF = 300.0
#: tb_conditioning's flags.update_log_z, the tracker feed: True = the mixed feed (forward and
#: backward evidence; owner default 2026-09-26), False = forward rollouts only
UPDATE_LOG_Z = True

#: The ladder. n = molecules in the rung's condition set (its manifest's train molecules;
#: each molecule contributes one or more stereoisomer conditions). phase1_max_steps and epochs are
#: WORKING ASSUMPTIONS: no conformer set has run phase 1 at these sizes. Revisit from the R0
#: run's phase-1 exit step (FORCED EXIT or gate) and its tracker trust onset.
RUNGS = {
    'r0': dict(n=6, set_id='qm9_r0', phase1_max_steps=10_000, epochs=40_000),
    'r1': dict(n=50, set_id='qm9_r1', phase1_max_steps=20_000, epochs=100_000),
    'r2': dict(n=500, set_id='qm9_r2', phase1_max_steps=20_000, epochs=200_000),
}
#: Buffer caps in rows, every rung. The floors are rows per condition: the prior-buffer
#: rebuild target (max_size x init_fraction) and the prior dataset each >= 20 per condition,
#: the anchor buffer >= 40 (stage-0 audit TRK-8).
BUFFER_CAPS = dict(prior=100_000, anchor=100_000, replay=50_000, prior_sample=50_000)
FLOOR_PER_CONDITION = dict(prior_rebuild=20, prior_sample=20, anchor=40)
#: Bytes of one stored conformer row: the top of the stage-0 measurement, 23-41 KB per row at
#: float32, QM9 at `full` (audit CFG-8). Every budget below multiplies it.
BYTES_PER_ROW_MAX = 41_000
#: VRAM budget for the rows held on cfg:buffer_device (the three buffers at their caps and
#: the prior dataset): card x cfg:cuda_memory_fraction x BUFFER_SHARE. LOCAL_CARD_BYTES is
#: the local card (audit CFG-8), which tests/config/test_conformer_canonical_contract.py
#: holds configs/conformer_mk.yaml to. CARD_BYTES (the smaller A100) and BUFFER_SHARE are
#: WORKING ASSUMPTIONS; revisit from the vram ledger of the first R1 leg.
CARD_BYTES = 40e9
LOCAL_CARD_BYTES = 16.3e9
BUFFER_SHARE = 0.5
#: Disk budget for the ladder's buffer sidecars: per arm, one frozen copy per archive
#: (epochs // archive_period) plus the rolling one, each (prior + anchor + replay max_size)
#: x BYTES_PER_ROW_MAX. WORKING ASSUMPTION: the ladder's share of the cluster scratch quota,
#: which no file in the repository records.
DISK_BUDGET_BYTES = 1.0e12

#: mk_dev keys the conformer route does not carry: crystal energy_config keys that are no
#: ConformerTorsions parameter. The first group is refused by
#: config_invariants.conformer_energy_config_keys_are_read; the second is in
#: conformer_modeller._NON_ENERGY_KEYS, popped and read by nothing on this route.
CRYSTAL_ONLY_KEYS = frozenset({
    'energy_config.lambda_mix', 'energy_config.prior_flow_path',
    'energy_config.physical_energy_clip', 'energy_config.mlip_compile',
    'energy_config.mlip_edge_chunk_size', 'energy_config.mlip_activation_checkpointing',
    'energy_config.density_coeff', 'energy_config.reduction_coeff',
    'energy_config.analyze_kwargs', 'energy_config.internal_oom_recovery',
    'energy_config.reward_range',
})
#: energy_config keys the route drops and still reads elsewhere (utils.problem_slug)
TOLERATED_ENERGY_KEYS = frozenset({'temperature'})
#: ConformerTorsions parameters the set builder never passes
#: (build_conformer_conditions._RUN_SURFACE_KEYS), so a manifest's energy kwargs omit them
RUN_SURFACE_KEYS = frozenset({'smiles', 'device', 'dtype', 'temperature_conditioning',
                              'embedding_conditioning', 'embedding_conditioning_dim',
                              'log_temperature_range'})
#: keys holding a file path a run opens or writes
DATA_PATH_KEYS = ('molecules_path', 'test_molecules_path', 'prior_path', 'checkpoints_dir',
                  'mlip_path', 'energy_config.internal_prior_path',
                  'energy_config.prior_dataset_path', 'buffers.anchor_buffer.shape_path')
#: tb_conditioning's loss_coeffs per branch as the baseline (configs/cond_tb_sep25) sets
#: them; check_protocol compares the EFFECTIVE value, stage override over the base block
TB_SEAT = {
    'fwd': dict(tb=1.0, freeze_policy=0.0, freeze_z=0.0, repeats=1.0, beta=80.0,
                tb_z_source='persistent', emp_z_persistent=1.0, reward_grads=1.0,
                path_grad_last_k=1),
    'bwd': dict(tb=1.0, freeze_z=1.0, repeats=1.0, beta=80.0, tb_z_source='persistent'),
    'replay': dict(tb=1.0, freeze_z=1.0, repeats=1.0, beta=80.0, tb_z_source='persistent'),
}
_DRIVE = re.compile(r'^[A-Za-z]:[\\/]')
_NO_DEFAULT = object()                  # a keyword-only parameter without a default

_spec = importlib.util.spec_from_file_location('fin19make', ROOT / 'final_sep19' / 'make.py')
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)
w3 = fin.w3
PLACEHOLDER = fin.PLACEHOLDER
CLUSTER_CKPTS = w3.CLUSTER_CKPTS


# ----------------------------------------------------------------------------- git

def git_show(rel):
    """A file's COMMITTED text. Raises -- never falls back to the working tree."""
    out = subprocess.run(['git', 'show', f'HEAD:{rel}'], capture_output=True, text=True,
                         cwd=str(REPO))
    if out.returncode != 0:
        raise SystemExit(f'REFUSING: git cannot produce HEAD:{rel} ({out.stderr.strip()}); the '
                         f'cluster runs HEAD, so a battery is generated from HEAD or not at all')
    return out.stdout


def committed_at_head(rel_to_es):
    """True when `rel_to_es` (relative to energy_sampling/) is a file in HEAD."""
    rel = 'energy_sampling/' + pathlib.PurePosixPath(rel_to_es).as_posix()
    return subprocess.run(['git', 'cat-file', '-e', f'HEAD:{rel}'], capture_output=True,
                          cwd=str(REPO)).returncode == 0


def energy_contract(torsions_src, modeller_src):
    """(ConformerTorsions.__init__ parameter names, conformer_modeller._NON_ENERGY_KEYS,
    {parameter: default}), parsed from source text. The first two are the sets
    init_energy_function filters energy_config with; a parameter without a default maps to
    _NO_DEFAULT."""
    tree = ast.parse(torsions_src)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'ConformerTorsions'), None)
    init = next((n for n in (cls.body if cls else []) if isinstance(n, ast.FunctionDef)
                 and n.name == '__init__'), None)
    if init is None:
        raise SystemExit('REFUSING: no ConformerTorsions.__init__ in the committed source')
    a = init.args
    pos = a.posonlyargs + a.args
    params = frozenset(x.arg for x in pos + a.kwonlyargs) - {'self'}
    defaults = {x.arg: _NO_DEFAULT for x in pos + a.kwonlyargs if x.arg != 'self'}
    for arg, node in (list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
                      + [(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]):
        try:
            defaults[arg.arg] = ast.literal_eval(node)
        except ValueError:
            raise SystemExit(f'REFUSING: ConformerTorsions.__init__ default of {arg.arg} is not '
                             f'a literal ({ast.unparse(node)}); energy_kwargs_mismatches needs it')
    for node in ast.parse(modeller_src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_NON_ENERGY_KEYS'
                                                for t in node.targets):
            return params, frozenset(ast.literal_eval(node.value)), defaults
    raise SystemExit('REFUSING: no literal _NON_ENERGY_KEYS in the committed conformer_modeller.py')


def dirty_warnings():
    """Uncommitted state the cluster would not have."""
    lines = list(w3.dirty_files())
    out = subprocess.run(['git', 'status', '--porcelain', '--', CANONICAL], capture_output=True,
                         text=True, cwd=str(REPO)).stdout
    lines += [ln[3:] for ln in out.splitlines() if ln.strip()]
    if w3.MXT.exists():
        lines += [f'MXtalTools: {p}' for p in w3.mxt_dirty() if 'conformers' in p]
    else:
        lines.append(f'MXtalTools: not found at {w3.MXT}; its dirty state was NOT checked')
    return sorted(set(lines))


# ----------------------------------------------------------------------------- dict helpers

def flat(d, prefix=''):
    """{dotted.key: leaf} over nested dicts; a non-empty dict is a branch, anything else a leaf."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict) and v:
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def get(cfg, dotted):
    node = cfg
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def stage(cfg, name):
    hits = [s for s in cfg['protocols'][PROTOCOL]['stages'] if s['name'] == name]
    assert len(hits) == 1, f'{PROTOCOL} has {len(hits)} stages named {name!r}'
    return hits[0]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _plain(v):
    """Tuples as lists, so a parsed default compares equal to its YAML or JSON spelling."""
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


# ----------------------------------------------------------------------------- checks

def missing_mk_dev_keys(cfg, mk_dev):
    """mk_dev keys (protocols aside) that `cfg` does not carry and CRYSTAL_ONLY_KEYS does not
    name. A key mk_dev gains is then a refusal here until the conformer base carries it."""
    want = flat({k: v for k, v in mk_dev.items() if k != 'protocols'})
    have = flat({k: v for k, v in cfg.items() if k != 'protocols'})
    return sorted(k for k in want if k not in have and k not in CRYSTAL_ONLY_KEYS)


def refuse_local_paths(cfg, name, committed=committed_at_head):
    """No machine-local path may reach a cluster arm.

    w3._scan_local_paths refuses a c:/d: prefix or a backslash anywhere; this adds any drive
    letter, and for the data keys a RELATIVE path, which the run opens against its working
    directory: legal only for a file committed at HEAD, which a cluster pull delivers."""
    try:
        w3._scan_local_paths(cfg, name)
    except AssertionError as e:
        raise SystemExit(f'REFUSING {e}') from None
    for trail, v in flat(cfg).items():
        if isinstance(v, str) and _DRIVE.match(v):
            raise SystemExit(f'REFUSING {name}: local drive path at {trail}: {v!r}')
    for key in DATA_PATH_KEYS:
        v = get(cfg, key)
        if v is None:
            continue
        if not isinstance(v, str) or not v.strip():
            raise SystemExit(f'REFUSING {name}: {key}={v!r} is not a path')
        if not v.startswith('/') and not committed(v):
            raise SystemExit(f'REFUSING {name}: {key}={v!r} is a relative path to a file not '
                             f'committed at HEAD, so a cluster clone does not have it')


def buffer_rows(cfg):
    """Rows the run holds on cfg:buffer_device at its caps: the three buffers and the prior
    dataset."""
    b = cfg['buffers']
    return (int(b['prior_buffer']['max_size']) + int(b['anchor_buffer']['max_size'])
            + int(b['replay_buffer']['max_size']) + int(cfg['energy_config']['prior_sample_size']))


def sidecar_rows(cfg):
    """Rows in one buffer sidecar at the caps (checkpointing.Checkpointer.buffer_state)."""
    b = cfg['buffers']
    return (int(b['prior_buffer']['max_size']) + int(b['anchor_buffer']['max_size'])
            + int(b['replay_buffer']['max_size']))


def sidecar_disk_bytes(cfg):
    """Bytes of every buffer sidecar a run leaves on disk at its caps: one frozen copy per
    archive, plus the rolling one."""
    period = int(cfg.get('archive_period') or 0)
    n_frozen = int(cfg['epochs']) // period if period > 0 and cfg.get('archive_buffers') else 0
    return (n_frozen + 1) * sidecar_rows(cfg) * BYTES_PER_ROW_MAX


def refuse_over_vram(cfg, name, card_bytes):
    """Buffer rows x BYTES_PER_ROW_MAX must fit card x cuda_memory_fraction x BUFFER_SHARE
    when the buffers live on the card."""
    if not str(cfg['buffer_device']).startswith('cuda'):
        return
    need = buffer_rows(cfg) * BYTES_PER_ROW_MAX
    budget = card_bytes * float(cfg['cuda_memory_fraction']) * BUFFER_SHARE
    if need > budget:
        raise SystemExit(f'REFUSING {name}: {buffer_rows(cfg):,} rows on {cfg["buffer_device"]} '
                         f'(prior, anchor and replay max_size and prior_sample_size) x '
                         f'{BYTES_PER_ROW_MAX:,} B = {need / 1e9:.1f} GB, above the '
                         f'{budget / 1e9:.1f} GB buffer budget of a {card_bytes / 1e9:.1f} GB card')


def check_protocol(cfg, name):
    """The conditional route as the baseline sets it: protocol and stages, the TB seat, the
    tracker feed, the tripwire and its backstops, the energy tier, clip, stereo lock and
    temperature. Shared by every arm and by the canonical file's contract test."""
    ec = cfg['energy_config']
    assert cfg['protocol'] == PROTOCOL, (name, cfg['protocol'])
    assert [s['name'] for s in cfg['protocols'][PROTOCOL]['stages']] == STAGES, name
    assert cfg['energy_function'] == 'conformer_torsions', name
    assert ec['level'] == 'full' and ec['force_field'] == 'mmff', (name, ec['level'], ec['force_field'])
    assert float(ec['energy_clip']) == ENERGY_CLIP, (name, ec['energy_clip'])
    sc = ec.get('stereo_coeff')
    if sc is None or float(sc) != STEREO_COEFF:
        raise SystemExit(f'REFUSING {name}: energy_config.stereo_coeff is '
                         f'{"absent (the code default, 0: the lock off)" if sc is None else repr(sc)}'
                         f'; every arm locks its stereoisomer at STEREO_COEFF {STEREO_COEFF:g}')
    t, lt = ec.get('temperature'), ec.get('log_temperature')
    if t is None or lt is None or not math.isclose(float(t), 10 ** float(lt), rel_tol=1e-9):
        raise SystemExit(f'REFUSING {name}: energy_config.temperature {t!r} is not '
                         f'10**log_temperature ({lt!r}). The energy runs at 10**log_temperature; '
                         f'utils.problem_slug names the checkpoint from temperature')
    assert cfg['compile_policy'] is False, (name, cfg['compile_policy'])
    assert cfg['model']['policy_kind'] == 'set' and int(cfg['model']['dplr_rank']) == 0, name
    assert cfg['embedding_conditioning'] is True, name
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors', name
    assert cfg['condition_log_z']['rollout_condition_draw'] == 'cycle', \
        (name, cfg['condition_log_z']['rollout_condition_draw'])

    tp, tb = stage(cfg, 'train_prior'), stage(cfg, 'tb_conditioning')
    assert tp.get('skip_if') == 'weights_loaded', name
    assert int(tp['max_steps']) < int(cfg['epochs']), \
        f'{name}: phase 1 is bounded at {tp["max_steps"]} but epochs is {cfg["epochs"]}'
    assert tp['loss_coeffs']['bwd']['tb_z_source'] == 'persistent', name
    assert tb['flags']['update_log_z'] is bool(UPDATE_LOG_Z), (name, tb['flags']['update_log_z'])
    for mode, want in TB_SEAT.items():
        have = {**cfg[f'{mode}_loss_coeffs'], **tb['loss_coeffs'].get(mode, {})}
        off = {k: have.get(k) for k, v in want.items() if have.get(k) != v}
        if off:
            raise SystemExit(f'REFUSING {name}: tb_conditioning {mode} loss_coeffs {off}, where '
                             f'the baseline seat sets {({k: want[k] for k in off})}')
    assert tb['fracs'] == {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}, (name, tb['fracs'])
    assert int(tb['fwd_rollout_every']) == 0, name
    assert 'freeze_pb' in tb['on_enter'], (name, tb['on_enter'])
    # replay must stay DORMANT (protocol.Stage.read_modes): a stage with no balance block
    # reads every mode and force-refreshes replay every controller.refresh_every steps
    assert tb.get('balance') is not None, f'{name}: tb_conditioning needs its balance block'
    for st in cfg['protocols'][PROTOCOL]['stages']:
        for action in st.get('on_enter', []) + st.get('on_exit', []):
            assert not str(action).startswith('snapshot_prior'), (name, st['name'], action)

    # --- the tripwire the baseline took out of service, and the backstops it kept
    hf = cfg['lr_control']['hard_failure']
    assert float(hf['loss_excursion_k']) >= 1.0e6, (name, hf['loss_excursion_k'])
    for k in ('grad_excursion_x', 'loss_abs', 'grad_abs'):
        assert k in hf and hf[k] is not None, \
            f'{name}: lr_control.hard_failure.{k} absent -- it would read as a code default'


def energy_kwargs_mismatches(ec, built, contract):
    """[(key, arm value, builder value)] over the chart parameters, each side's absent key
    taking the committed default; plus every builder key the committed signature lacks."""
    params, _non_energy, defaults = contract
    out = [(k, '<not a committed parameter>', built[k]) for k in sorted(set(built) - params)]
    for k in sorted(params - RUN_SURFACE_KEYS):
        a = _plain(ec[k]) if k in ec else _plain(defaults[k])
        b = _plain(built[k]) if k in built else _plain(defaults[k])
        if a is _NO_DEFAULT and b is _NO_DEFAULT:
            continue
        if a is _NO_DEFAULT or b is _NO_DEFAULT or a != b:
            out.append((k, '<absent>' if a is _NO_DEFAULT else a,
                        '<absent>' if b is _NO_DEFAULT else b))
    return out


def read_set(set_dir, name, n_molecules, ec, contract):
    """(bytes, sha256, RDKit version, train conditions) for one rung's builder output,
    checked against its manifest, the rung's molecule count and the arm's energy_config."""
    set_dir = pathlib.Path(set_dir)
    cond, man_path = set_dir / CONDITIONS_FILE, set_dir / MANIFEST_FILE
    for p in (cond, man_path):
        if not p.is_file():
            raise SystemExit(f'REFUSING {name}: {p} does not exist; build the set with '
                             f'build_conformer_set.py --out-dir {set_dir}')
    man = json.loads(man_path.read_text(encoding='utf-8'))
    if man.get('format') != MANIFEST_FORMAT:
        raise SystemExit(f'REFUSING {name}: {man_path} is format {man.get("format")!r}, '
                         f'not {MANIFEST_FORMAT!r}')
    listed = (man.get('artifacts') or {}).get(CONDITIONS_FILE) or {}
    size, sha = cond.stat().st_size, sha256_of(cond)
    if listed.get('sha256') != sha or listed.get('bytes') != size:
        raise SystemExit(f'REFUSING {name}: {cond} is {size} B sha256 {sha[:12]}, but '
                         f'{man_path} lists {listed.get("bytes")} B sha256 '
                         f'{str(listed.get("sha256"))[:12]} -- not the file the builder wrote')
    train = (man.get('rungs') or {}).get('train') or {}
    if train.get('molecules') != n_molecules or int(train.get('conditions') or 0) < n_molecules:
        raise SystemExit(f'REFUSING {name}: {man_path} counts {train.get("molecules")} train '
                         f'molecules in {train.get("conditions")} conditions; the rung is '
                         f'{n_molecules} molecules')
    rdkit = str((man.get('versions') or {}).get('rdkit') or '')
    if not re.fullmatch(r'[0-9A-Za-z.+_-]+', rdkit):
        raise SystemExit(f'REFUSING {name}: {man_path} names no usable RDKit version '
                         f'({rdkit!r}); the sbatch checks the cluster RDKit against it')
    bad = energy_kwargs_mismatches(ec, (man.get('energy') or {}).get('kwargs') or {}, contract)
    if bad:
        raise SystemExit(f'REFUSING {name}: {man_path} was built under other energy kwargs '
                         f'than the arm runs, (key, arm, builder): {bad}. Rebuild the set from '
                         f'the committed configs/conformer_mk.yaml')
    return size, sha, rdkit, int(train['conditions'])


def check_arm(cfg, name, spec, n_conditions, mk_dev, contract, committed=committed_at_head):
    """Every battery property, re-asserted on the FINISHED dict. The buffer floors are per
    CONDITION: n_conditions is the set's own count."""
    params, non_energy, _defaults = contract
    ec = cfg['energy_config']
    unknown = sorted(set(ec) - params - non_energy - TOLERATED_ENERGY_KEYS)
    if unknown:
        raise SystemExit(f'REFUSING {name}: energy_config carries {unknown}, outside the COMMITTED '
                         f'ConformerTorsions.__init__ signature and _NON_ENERGY_KEYS -- a key whose '
                         f'handler is not committed, or a misspelling')
    missing = missing_mk_dev_keys(cfg, mk_dev)
    if missing:
        raise SystemExit(f'REFUSING {name}: the committed mk_dev carries {missing}, which the base '
                         f'does not and CRYSTAL_ONLY_KEYS does not name. Carry them in '
                         f'configs/conformer_mk.yaml, or name them there as crystal-only')
    refuse_local_paths(cfg, name, committed)
    check_protocol(cfg, name)
    assert cfg['prior_model_name'] is None and cfg['load_weights_only'] is False, name
    assert cfg['continue_from_checkpoint'] is False and cfg['checkpoint_name'] == PLACEHOLDER, name
    assert cfg['archive_buffers'] is True, name       # CK_STEP needs the frozen sidecar
    assert int(stage(cfg, 'train_prior')['max_steps']) == spec['phase1_max_steps'], name

    # --- buffers against the per-condition floors and the VRAM budget
    b, n = cfg['buffers'], n_conditions
    rows = {'prior_rebuild': b['prior_buffer']['max_size'] * b['prior_buffer']['init_fraction'],
            'prior_sample': ec['prior_sample_size'], 'anchor': b['anchor_buffer']['max_size']}
    for key, floor in FLOOR_PER_CONDITION.items():
        if rows[key] < floor * n:
            raise SystemExit(f'REFUSING {name}: {key} holds {rows[key]:,.0f} rows, below '
                             f'{floor} per condition x {n} conditions')
    refuse_over_vram(cfg, name, CARD_BYTES)

    # --- the rule set, every severity: this is a canonical-derived production arm
    sys.path.insert(0, str(ES))
    import config_invariants
    loadable = copy.deepcopy(cfg)
    loadable['checkpoint_name'] = None
    bad = config_invariants.check(loadable)
    if bad:
        raise SystemExit(f'REFUSING {name}: config_invariants: ' + '; '.join(map(str, bad)))
    fin.load_check(cfg, name, STAGES)


def check_battery(arms):
    """Cross-arm properties: no duplicates, no glob collisions, one energy_clip, and the
    ladder's buffer sidecars within the disk budget."""
    names = list(arms)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            ia = {k: v for k, v in arms[a]['cfg'].items() if k != 'run_name'}
            ic = {k: v for k, v in arms[c]['cfg'].items() if k != 'run_name'}
            if ia == ic:
                raise SystemExit(f'REFUSING: arms {a} and {c} are identical apart from run_name')
    for a in names:
        for c in names:
            if a != c and c.endswith(a):
                raise SystemExit(f'REFUSING: arm {a} is a suffix of {c}; the sbatch glob '
                                 f'*{a}_* would match the other arm\'s checkpoints')
    clips = {float(arm['cfg']['energy_config']['energy_clip']) for arm in arms.values()}
    if len(clips) != 1:
        raise SystemExit(f'REFUSING: energy_clip differs across the ladder: {sorted(clips)}')
    disk = sum(sidecar_disk_bytes(arm['cfg']) for arm in arms.values())
    if disk > DISK_BUDGET_BYTES:
        raise SystemExit(f'REFUSING: the ladder\'s buffer sidecars come to {disk / 1e9:,.0f} GB '
                         f'at the caps, above the {DISK_BUDGET_BYTES / 1e9:,.0f} GB disk budget '
                         f'(per arm: ' + ', '.join(f'{n} {sidecar_disk_bytes(a["cfg"]) / 1e9:,.0f} GB'
                                                  for n, a in arms.items()) + ')')


# ----------------------------------------------------------------------------- build

def arm_digest(cfg, artifacts, rdkit):
    """8 hex of sha256 over the config (run_name aside), the artifacts and the RDKit version."""
    body = {k: v for k, v in cfg.items() if k != 'run_name'}
    blob = json.dumps([body, [list(a) for a in artifacts], rdkit], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:8]


def arm_name(rung, spec, digest):
    return f"cc_{rung}_n{spec['n']}_{digest}"


def build_arm(base, rung, spec, local_sets, local_es, contract):
    """{'cfg', 'artifacts' [(path under CLUSTER_SETS, bytes, sha256), ...], 'rung', 'rdkit',
    'conditions'} for one rung. The run name is set last, from the digest of everything else."""
    cfg = copy.deepcopy(base)
    label = f'cc_{rung}_n{spec["n"]}'
    cfg['run_name'] = None
    cfg['tag'] = TAG
    cfg['checkpoints_dir'] = CLUSTER_CKPTS
    cfg['checkpoint_name'] = PLACEHOLDER
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False
    cfg['prior_model_name'] = None
    cfg['cuda_memory_fraction'] = CUDA_MEMORY_FRACTION
    cfg['epochs'] = int(spec['epochs'])
    stage(cfg, 'train_prior')['max_steps'] = int(spec['phase1_max_steps'])
    stage(cfg, 'tb_conditioning')['flags']['update_log_z'] = bool(UPDATE_LOG_Z)
    b = cfg['buffers']
    b['prior_buffer']['max_size'] = BUFFER_CAPS['prior']
    b['anchor_buffer']['max_size'] = BUFFER_CAPS['anchor']
    b['replay_buffer']['max_size'] = BUFFER_CAPS['replay']
    cfg['energy_config']['prior_sample_size'] = BUFFER_CAPS['prior_sample']

    size, sha, rdkit, n_conditions = read_set(pathlib.Path(local_sets) / spec['set_id'], label,
                                              int(spec['n']), cfg['energy_config'], contract)
    cond_rel = f"{spec['set_id']}/{CONDITIONS_FILE}"
    prior_local = pathlib.Path(base['energy_config']['internal_prior_path'])
    if not prior_local.is_absolute():
        prior_local = pathlib.Path(local_es) / prior_local
    if not prior_local.is_file():
        raise SystemExit(f'REFUSING {label}: {prior_local} does not exist; its size and sha256 '
                         f'go into the index, and the cluster copy is checked against them')
    prior_rel = prior_local.name
    artifacts = [(cond_rel, size, sha),
                 (prior_rel, prior_local.stat().st_size, sha256_of(prior_local))]
    for rel, _s, _h in artifacts:
        if any(c.isspace() for c in rel):
            raise SystemExit(f'REFUSING {label}: whitespace in artifact path {rel!r}')
    cfg['molecules_path'] = f'{CLUSTER_SETS}/{cond_rel}'
    cfg['energy_config']['internal_prior_path'] = f'{CLUSTER_SETS}/{prior_rel}'
    cfg['prior_path'] = None
    cfg['test_molecules_path'] = None
    cfg['run_name'] = arm_name(rung, spec, arm_digest(cfg, artifacts, rdkit))
    return dict(cfg=cfg, artifacts=artifacts, rung=rung, rdkit=rdkit, conditions=n_conditions)


def build(base, mk_dev, contract, rungs, local_sets=LOCAL_SETS, local_es=ES,
          committed=committed_at_head):
    """{arm: {'cfg', 'artifacts', 'rung', 'rdkit', 'conditions'}} for the named rungs, every
    arm checked."""
    if base.get('protocol') != PROTOCOL or PROTOCOL not in (base.get('protocols') or {}):
        raise SystemExit(f'REFUSING: the base does not select protocol {PROTOCOL!r} (it selects '
                         f'{base.get("protocol")!r}). The base is the COMMITTED '
                         f'configs/conformer_mk.yaml; commit the conformer master config first')
    arms = {}
    for rung in rungs:
        spec = RUNGS[rung]
        arm = build_arm(base, rung, spec, local_sets, local_es, contract)
        name = arm['cfg']['run_name']
        if name in arms:
            raise SystemExit(f'REFUSING: two arms named {name}')
        check_arm(arm['cfg'], name, spec, arm['conditions'], mk_dev, contract, committed)
        arms[name] = arm
    check_battery(arms)
    return arms


# ----------------------------------------------------------------------------- sbatch

#: The crystal template's two index reads of the prior (columns 5 and 6), replaced by the
#: read of this index's rdkit column.
_PRIOR_COLUMNS = r"""PRIOR=$(awk -F'\t' -v n=${{ROW}} 'NR==n {{print $5}}' ${{INDEX}})
PRIOR_BYTES=$(awk -F'\t' -v n=${{ROW}} 'NR==n {{print $6}}' ${{INDEX}})
"""
_RDKIT_COLUMN = r"""RDKIT=$(awk -F'\t' -v n=${{ROW}} 'NR==n {{print $5}}' ${{INDEX}})
"""
#: The crystal template's MXtalTools wall checks and single-prior byte check, from its first
#: line to its last, replaced whole by DATA_GUARD.
_WALLS_START = '# THE WALLS.'
_WALLS_END = ('    echo "FATAL: ${{DATA}}/${{PRIOR}} is ${{HAVE}} bytes, expected ${{PRIOR_BYTES}}" '
              '>&2; exit 1\nfi\n')
DATA_GUARD = r"""# DATA GUARD. Every artifact this arm reads, as (path under DATA, bytes, sha256) triples from
# column 6 of its index row on. A size or hash that differs is an unfinished upload or another
# file; a row with no triple or no rdkit column is a malformed index. Each exits before srun.
# NIG_EXPORT is the crystal template's MXtalTools export, empty on this route.
NIG_EXPORT=""
if [ -z "${{RDKIT}}" ]; then echo "FATAL: no rdkit version in row ${{ROW}} of ${{INDEX}}" >&2; exit 1; fi
BAD=0
NT=0
for T in $(awk -F'\t' -v n=${{ROW}} 'NR==n {{for (i = 6; i + 2 <= NF; i += 3) print $i "|" $(i + 1) "|" $(i + 2)}}' ${{INDEX}}); do
    NT=$((NT + 1))
    F=${{T%%|*}}; REST=${{T#*|}}; WANT=${{REST%%|*}}; SHA=${{REST#*|}}
    HAVE=$(stat -c %s ${{DATA}}/${{F}} 2>/dev/null || echo 0)
    if [ "${{HAVE}}" != "${{WANT}}" ]; then
        echo "FATAL: ${{DATA}}/${{F}} is ${{HAVE}} bytes, expected ${{WANT}}" >&2; BAD=1
    elif [ "$(sha256sum ${{DATA}}/${{F}} | cut -d' ' -f1)" != "${{SHA}}" ]; then
        echo "FATAL: ${{DATA}}/${{F}} does not have sha256 ${{SHA}}" >&2; BAD=1
    fi
done
if [ "${{NT}}" = "0" ]; then echo "FATAL: no artifact in row ${{ROW}} of ${{INDEX}}" >&2; exit 1; fi
if [ "${{BAD}}" != "0" ]; then exit 1; fi
"""
_NIGGLI_GUARD = ("""python -c \\"import mxtaltools.common.sym_utils as s; assert getattr(s, """
                 """'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\\" || exit 1""")
_CONFORMER_GUARD = ("""python -c \\"import rdkit; from rdkit.Chem import AllChem; """
                    """assert rdkit.__version__ == '${{RDKIT}}', 'RDKit %s; the condition sets """
                    """were built under ${{RDKIT}}' % rdkit.__version__; """
                    """assert hasattr(AllChem, 'MMFFGetMoleculeForceField'), 'RDKit without MMFF'; """
                    """import mxtaltools.conformers.builder\\" || exit 1""")
_LAUNCH = ('python -u train.py --config', 'python -u conformer_modeller.py --config')

SEED_FRESH = """    # FRESH: no checkpoint of this arm exists, so the rung starts from scratch and trains its own
    # phase 1 first. `basename null` is `null`, so the resolved checkpoint_name is YAML null.
    CK=null
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  FRESH (no checkpoint)\""""


def _replace_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'REFUSING: the shared sbatch template holds {n} copies of the {what} '
                         f'span this generator substitutes (need exactly 1) -- '
                         f'configs/final_sep19/make.py::SBATCH changed; re-derive the substitution')
    return text.replace(old, new)


def conformer_template(template):
    """final_sep19's SBATCH with the four conformer substitutions, still a format string."""
    if template.count(_WALLS_START) != 1 or template.count(_WALLS_END) != 1:
        raise SystemExit('REFUSING: the shared sbatch template no longer carries exactly one '
                         'MXtalTools wall/prior-byte block (configs/final_sep19/make.py::SBATCH)')
    i = template.index(_WALLS_START)
    j = template.index(_WALLS_END) + len(_WALLS_END)
    if j <= i:
        raise SystemExit('REFUSING: the wall block ends before it starts in the shared template')
    text = template[:i] + DATA_GUARD + template[j:]
    text = _replace_once(text, _PRIOR_COLUMNS, _RDKIT_COLUMN, 'prior index columns')
    text = _replace_once(text, _NIGGLI_GUARD, _CONFORMER_GUARD, 'inline Niggli assertion')
    return _replace_once(text, *_LAUNCH, 'train.py launch')


def render_sbatch(n_arms, ckpts=CLUSTER_CKPTS, data=CLUSTER_SETS, template=None):
    template = fin.SBATCH if template is None else template
    return conformer_template(template).format(
        wall=WALL, tag=TAG, battery=BATTERY, ckpts=ckpts, data=data, last=n_arms - 1, leg=LEG,
        seed_block=SEED_FRESH,
        what='conditional conformer ladder, one FRESH arm per rung (phase 1 then phase 2).')


def index_rows(arms):
    rows = []
    for name, arm in arms.items():
        row = [name, arm['rung'], 'fresh', '-', arm['rdkit']]
        for rel, size, sha in arm['artifacts']:
            row += [rel, str(size), sha]
        rows.append(row)
    return rows


def write(out_dir, arms):
    out_dir = pathlib.Path(out_dir)
    for stale in out_dir.glob('cc_*.yaml'):
        stale.unlink()
    (out_dir / 'joblogs').mkdir(exist_ok=True)
    (out_dir / 'joblogs' / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, arm in arms.items():
        path = out_dir / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(arm['cfg'], f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == arm['cfg'], \
            f'{name}: read-back differs'
    width = max(len(r) for r in index_rows(arms))
    header = ['arm', 'rung', 'start', 'warm_src', 'rdkit'] + [
        f'{kind}{i}' for i in range((width - 5) // 3) for kind in ('artifact', 'bytes', 'sha256')]
    with (out_dir / f'INDEX_{LEG}.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(header) + '\n')
        for row in index_rows(arms):
            f.write('\t'.join(row) + '\n')
    with (out_dir / f'submit_{BATTERY}_{LEG}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(render_sbatch(len(arms)))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--rungs', default=','.join(RUNGS))
    a = ap.parse_args(argv)
    rungs = [r for r in a.rungs.split(',') if r]
    unknown = [r for r in rungs if r not in RUNGS]
    if unknown:
        raise SystemExit(f'unknown rung(s) {unknown}; RUNGS holds {list(RUNGS)}')
    dirty = dirty_warnings()
    if dirty:
        print('WARNING: uncommitted state (the base is read from HEAD; the cluster runs HEAD):\n  '
              + '\n  '.join(dirty))
    base = yaml.safe_load(git_show(CANONICAL))
    mk_dev = yaml.safe_load(git_show(MK_DEV))
    contract = energy_contract(git_show(TORSIONS_SRC), git_show(MODELLER_SRC))
    arms = build(base, mk_dev, contract, rungs)
    write(HERE, arms)
    for i, (name, arm) in enumerate(arms.items()):
        cfg = arm['cfg']
        print(f"[{i}] {name:<24} M={RUNGS[arm['rung']]['n']:<5} C={arm['conditions']:<5} phase-1 bound "
              f"{stage(cfg, 'train_prior')['max_steps']:,} of epochs {cfg['epochs']:,} | RDKit "
              f"{arm['rdkit']} | " + ' '.join(f'{rel} ({size:,} B)' for rel, size, _h in arm['artifacts']))
        print(f"    buffer sidecars at the caps: {sidecar_disk_bytes(cfg) / 1e9:,.0f} GB on disk; the "
              f"rolling one, {sidecar_rows(cfg) * BYTES_PER_ROW_MAX / 1e9:.1f} GB, is rewritten every "
              f"{cfg['eval_period']} steps")
    print(f'stage to {CLUSTER_SETS}/ before submitting: the files above, same relative paths')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
