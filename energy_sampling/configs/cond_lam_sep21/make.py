r"""cond_lam_sep21 -- the conditional lambda restart on the NIGGLI-V2 priors, LOCAL.

The triclinic reduction penalty changed (mxtaltools 9e179673: tri_niggli_reduction_penalty
is always on, MXT_NIGGLI_TRICLINIC is retired and RAISES if set), and the conditional prior
was re-expressed in Niggli cells as qm9c100k_prior_niggli_v2.pt (2026-09-17, 52,329 -> 52,181
rows, 48.1% beta/gamma flipped). Three consequences drive this battery:

  1. prior_path is in get_problem_definition, so the new prior is a NEW PROBLEM. Every qm9c
     checkpoint on disk carries hash 5e5294 (the old qm9c100k_prior.pt). Phase 1 re-runs.
  2. qm9c100k_prior_flow_T20.pt was fitted by build_prior_flow.py to draws from a phase-1
     policy trained on the OLD convention. prior_flow_path is NOT in the problem hash, so
     nothing catches a stale flow at load. It is refit off this battery's phase-1 exit.
  3. The reduction penalty sits inside the PHYSICAL leg, which carries weight lambda. At
     lambda=0 the target is exactly the fitted flow and the penalty is out of it entirely,
     so the lambda=0 leg tests the new prior and the refit flow without the changed penalty;
     the penalty only enters as lambda rises.

THE SEAT is the forward seat, the developmental workhorse (owner 2026-09-21): fwd 0.5 /
bwd 0.5, replay 0, pooled_source fwd, freeze_policy 0 on fwd -- the qm9c_null_fwdseat shape.
The replay seat (stored_force_k on stored rows) is for production later. The forward seat is
also the only one that admits the LIVE terminal force: gflownet_losses.py:351 refuses the
reshaped reward path on a freeze_policy branch, which is what var_conditioning's fwd_z_sidecar
makes the forward branch in the canonical mk_dev protocol.

THE LADDER starts at lambda=0 and steps up (owner 2026-09-21); a smooth anneal comes later.
No anneal_coeffs on any arm here -- lambda_mix is pinned per arm, so a force effect is read
at one fixed target rather than across a rung's ~10-15k-step recovery.

THE ORDER, from energy_sampling/ with PYTHONPATH carrying mxtaltools and gfn_diffusion:

  1. python configs/cond_lam_sep21/make.py p1
     python -u train.py --config configs/cond_lam_sep21/p1.yaml
     THE GATE IS NOT RELIED ON (owner 2026-09-21): the convergence point is eyeballed and
     the run stopped by hand, so expect NO <prefix>_phase1_exit.pt. The seed is whichever
     checkpoint the owner picks -- archive_period 5000 writes <prefix>_step5000.pt and so
     on, beside the rolling _best.pt / _running.pt / _last_ok.pt.

     FOR SCALE, not as a target. The old conditional phase 1 (dev_qm9c_t20, old prior,
     T=20) wrote its phase1_exit at step 16,760. Its w1r/worst sat flat at 14.5-15.3 from
     step 500 to 7,250 and only broke downward after ~13,000, reaching 10.62 at 15,000
     against its 10.0 bar; w1r/median was under its 5.0 bar from ~step 3,250. A flat
     w1r/worst in the first several thousand steps is this phase's normal shape.

     WHAT IS HARDER ON THE V2 PRIOR. The flip rule truncates beta and gamma at exactly 90
     degrees: in qm9c100k_prior_niggli_v2 both have min 90.00 and p5 90.5/90.9, i.e. a hard
     density edge at latent 0 with mass piled against it, where the old file spread them
     over 58-122 degrees with no edge. Alpha moved from median 99.2 to 90.5, now symmetric
     about 90. The progress gate's worst column is 5 -- gamma -- and cl21_p1 runs ~45% above
     the old run's w1r/worst at matched steps (22.1 vs 14.6 at step 4,000). The 10.0 bar was
     calibrated on a prior with no such edge; if it does not fire, that bar is the thing to
     re-examine, not the run.

  2. python configs/cond_lam_sep21/make.py seeds        # lists candidates with their steps
     python build_prior_flow.py \
         --checkpoint D:\crystal_datasets\gfn_checkpoints\<the picked seed> \
         --conditions D:\crystal_datasets\conditional\priors\qm9c100k_conditions.pt \
         --out D:\crystal_datasets\conditional\priors\qm9c100k_niggli_v2_prior_flow_T25.pt
     --traj-T defaults to the checkpoint's train_T, which is 25.

  3. python configs/cond_lam_sep21/make.py lam --seed <the same seed>
     The seed must be the SAME file step 2 sampled: check_flow compares the flow's stored
     checkpoint, conditions file, traj_T and problem_hash against it. Nothing downstream
     would catch a mismatch, because prior_flow_path is outside the problem hash.

  l0_ctrl   lambda 0, reward_grads 0                     the null on the new priors
  l0_f      lambda 0, k 1, reward_grads 1                the plain terminal force
  l0_fc     lambda 0, k 1, reward_grads 1, |F| clip      the clipped force (--with-clip)

  The GATE arm (reward_grad_gate) is deliberately absent at lambda=0: the gate zeroes rows
  whose TB residual is at or below its value, and at a converged null the residuals sit near
  zero, so a gate calibrated on the unconditional search regime (force_sep18's 1.0) would
  gate off nearly every row and the arm would read as the control. It belongs on the lambda>0
  leg, with its threshold set from the residual spread l0_ctrl actually shows.

IDENTITY: all three paths are local (D:\). This battery is LOCAL ONLY -- a D:\ path in a
cluster arm kills it. The lambda arms' problem hash is asserted equal to p1's.
"""
import argparse
import glob
import os
import sys
from argparse import Namespace
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ES = HERE.parent.parent
P1_BASE = HERE.parent / 'qm9c_t20.yaml'                # the config the existing phase1_exit came from
LAM_BASE = HERE.parent / 'qm9c_null_fwdseat.yaml'      # the lambda=0 forward-seat null
TAG = 'cl21c'        # generation c: lambda 0.05, P_B frozen again (owner 2026-09-21)
P1_TAG = 'cl21'      # phase 1 ran under the original tag; its checkpoints keep that prefix
P1_RUN = 'p1'          # train.py prefixes the tag: the run is cl21_p1
CKPT_DIR = 'D:\\crystal_datasets\\gfn_checkpoints'
PRIORS = 'D:\\crystal_datasets\\conditional\\priors'
PRIOR = PRIORS + '\\qm9c100k_prior_niggli_v2.pt'
CONDS = PRIORS + '\\qm9c100k_conditions.pt'            # molecules, not cells -- unchanged by the migration
TEST_CONDS = PRIORS + '\\qm9c100k_test_conditions.pt'  # likewise
OLD_PRIOR_HASH = '5e5294'
TRAJ_T = 25            # owner 2026-09-21; the qm9c line ran at 20. T is absent from the problem
                       # def, so only PriorFlow.verify_against_policy catches a flow/run mismatch.
FLOW = PRIORS + f'\\qm9c100k_niggli_v2_prior_flow_T{TRAJ_T}.pt'
LAM_STEPS = 8000       # bounded so four concurrent arms cannot hold the GPU indefinitely
LR_SCALE = 0.2         # the settled phase-2 scale (the Sep 12 _lr4x sweep off the null's 0.05)
FORCE_CLIP = 50.0      # |d log R / d x_T| cap. With the flow leg's gradient switched on
                       # (prior_flow.py, 2026-09-21) the force at lambda 0.01 measures
                       # median 14, p99 43, max 47 in log R units on phase-1 draws, so 50
                       # barely binds -- deliberately. The reshape's load-bearing job here
                       # is its nan_to_num on a non-finite force, which the plain path has
                       # no equivalent of; the clip is the tail guard, not the mechanism.
# FOUR ARMS SHARE ONE 16 GB CARD. Pinned batch and a fixed memory fraction per arm: a
# ctrl/force pair is only comparable if neither the sizer nor an OOM cut moved the batch
# under one of them, and the sizer is occupancy-driven so concurrency alone would move it.
LAM_BATCH = 1000
CUDA_FRAC = 0.23       # each arm also holds a frozen prior GFN beside the live one


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def problem_hash_of(cfg):
    # gfn_diffusion for `energy_sampling.*`, and the sibling MXtalTools checkout, which
    # utils.py imports at module scope. PyCharm supplies both as content roots.
    for p in (ES.parent, ES.parent.parent / 'mxtaltools'):
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)
    from energy_sampling.utils import get_problem_definition, normalize_problem_def, problem_hash
    return problem_hash(normalize_problem_def(get_problem_definition(_ns(cfg))))


def load(path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write(cfg, name):
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out


def identity(cfg, name):
    """The three data paths and the run's name. Shared by p1 and the lambda arms."""
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['checkpoints_dir'] = CKPT_DIR
    cfg['prior_path'] = PRIOR
    cfg['molecules_path'] = CONDS
    cfg['test_molecules_path'] = TEST_CONDS
    assert os.path.exists(PRIOR), f'missing prior {PRIOR}'
    assert os.path.exists(CONDS), f'missing conditions {CONDS}'
    assert os.path.exists(TEST_CONDS), f'missing test conditions {TEST_CONDS}'
    # both bases are T=20 files; T moves together or the flow's traj_T guard is the only
    # thing standing between a mismatched pair and a silently wrong lambda=0 target.
    cfg['integrator']['T'] = TRAJ_T
    cfg['eval_T'] = TRAJ_T
    assert cfg['embedding_conditioning'] is True, 'conditional route expected'
    return cfg


# ---------------------------------------------------------------- phase 1

def make_p1():
    cfg = identity(load(P1_BASE), P1_RUN)
    # FRESH WEIGHTS. The old-prior phase-1 exit is a different problem and its policy is
    # tuned to the beta/gamma convention 48% of the rows no longer carry.
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False
    cfg['checkpoint_name'] = None
    cfg['prior_model_name'] = None      # nothing to load, so train_prior's skip_if cannot fire
    ec = cfg['energy_config']
    # phase 1 is the PHYSICAL target: no flow exists yet, and MolecularCrystal.__init__
    # raises on lambda_mix != 1 without one.
    assert ec.get('prior_flow_path') is None and float(ec.get('lambda_mix', 1.0)) == 1.0
    stages = cfg['protocols'][cfg['protocol']]['stages']
    tp = [s for s in stages if s['name'] == 'train_prior'][0]
    assert tp['skip_if'] == 'prior_loaded', tp.get('skip_if')
    assert 'snapshot:phase1_exit' in tp['on_exit'] and 'snapshot_prior' in tp['on_exit'], tp['on_exit']
    h = problem_hash_of(cfg)
    assert not h.startswith(OLD_PRIOR_HASH), f'hash {h} is still the old prior'
    out = write(cfg, 'p1')
    print(f'wrote {out.name:10s} fresh MLE on {os.path.basename(PRIOR)}  T={TRAJ_T}  hash {h}')
    print(f'  phase-1 exit will be   {P1_TAG}_{P1_RUN}_elj-qm9c100k_prior_niggli_v2-T6.9-{h}_phase1_exit.pt')
    return h


# ---------------------------------------------------------------- lambda arms

def p1_checkpoints():
    """Every cl21_p1 checkpoint, with the step and stage each stores. For picking a seed."""
    import torch
    rows = []
    for p in sorted(glob.glob(os.path.join(CKPT_DIR, f'{P1_TAG}_{P1_RUN}_*.pt'))):
        if p.endswith('_buffers.pt'):
            continue
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            ms = ck.get('modeller_state') or {}
            rows.append((os.path.basename(p), int(ms.get('step_ind', -1)),
                         ms.get('stage'), int(ck.get('train_T', -1))))
        except Exception as e:                      # a half-written _running.pt
            rows.append((os.path.basename(p), -1, f'UNREADABLE: {type(e).__name__}', -1))
    return rows


def resolve_seed(name):
    """The named phase-1 checkpoint, and the hash and step it actually stores.

    NO DEFAULT and no 'newest' rule: the convergence point on this prior is eyeballed
    (owner 2026-09-21), the progress gate is not relied on, and a silently-picked seed is
    the one mistake that would not show up anywhere downstream -- every arm would train
    from the wrong policy and the flow would be fitted to a different one again.
    """
    import torch
    p = os.path.join(CKPT_DIR, name)
    assert os.path.exists(p), (
        f'no such checkpoint: {p}\navailable:\n' +
        '\n'.join(f'  {n:70s} step {s:>7}  {st}' for n, s, st, _ in p1_checkpoints()))
    ck = torch.load(p, map_location='cpu', weights_only=False)
    ms = ck.get('modeller_state') or {}
    assert int(ck['train_T']) == TRAJ_T, f'{name}: train_T {ck["train_T"]}, this battery is {TRAJ_T}'
    assert ms.get('stage') == 'train_prior', (
        f'{name}: stage is {ms.get("stage")!r}, not train_prior -- this is meant to be the '
        f'MLE policy, and a var_conditioning checkpoint would seed the lambda arms from a '
        f'run that already trained against the physical target')
    return os.path.basename(p), ck['problem_hash'], int(ms.get('step_ind', -1))


def check_flow(p1_exit, p1_hash):
    """The flow must exist, be fitted at this run's T, and come from THIS phase-1 exit.

    prior_flow_path is NOT in the problem hash, so a stale flow reaches the lambda=0 target
    silently. build_prior_flow stamps the checkpoint it sampled and that run's problem_hash
    into the provenance; both are checked here because nothing downstream will.
    """
    assert os.path.exists(FLOW), (
        f'missing {FLOW} -- refit it first:\n'
        f'  python build_prior_flow.py --checkpoint {os.path.join(CKPT_DIR, p1_exit)} '
        f'--conditions {CONDS} --out {FLOW}')
    import torch
    blob = torch.load(FLOW, map_location='cpu', weights_only=False)
    assert int(blob['traj_T']) == TRAJ_T, f'flow was fitted at T={blob["traj_T"]}, run integrates {TRAJ_T}'
    prov = dict(blob.get('provenance') or {})
    assert prov.get('checkpoint') == p1_exit, (
        f'flow was fitted to {prov.get("checkpoint")!r}, not this battery\'s {p1_exit!r}')
    assert prov.get('problem_hash') == p1_hash, (
        f'flow provenance carries problem_hash {prov.get("problem_hash")!r}, not {p1_hash!r}')
    assert os.path.basename(CONDS) == prov.get('conditions'), (
        f'flow was fitted over {prov.get("conditions")!r}, not {os.path.basename(CONDS)!r}')
    return blob, prov


def arm(name, lam, k, rg, p1_exit, p1_hash, gate=None, force_clip=0.0, steps=LAM_STEPS,
        pclip=None, path_grad_scale=1, freeze_pb=False):
    cfg = identity(load(LAM_BASE), name)
    cfg['checkpoint_name'] = p1_exit
    cfg['load_weights_only'] = True          # the policy only; buffers and step count start fresh
    cfg['continue_from_checkpoint'] = False
    # THE SKIP. train_prior carries skip_if: prior_loaded, and protocol.py:2046 fires it
    # only when `m.prior_model` exists, which needs prior_model_name to load. With it null
    # the arm re-runs the MLE stage instead of entering var_conditioning, and the whole
    # lambda/VarGrad phase never happens -- fwd/ stays empty and pooled/* is never logged.
    # Phase 1 was stopped by hand, so its on_exit snapshot_prior never ran and there is no
    # _prior.pt; the seed checkpoint carries model_eval and gfn_config, which is all
    # train.py:3706 builds the frozen prior GFN from, and it IS the MLE policy that
    # snapshot would have held. prior_buffer.source is 'anchors', so nothing samples it.
    cfg['prior_model_name'] = p1_exit
    cfg['epochs'] = int(steps)
    cfg['lr_control']['fixed_scale'] = LR_SCALE
    assert cfg['lr_control']['mode'] == 'fixed', cfg['lr_control']['mode']
    cfg['cuda_memory_fraction'] = CUDA_FRAC
    cfg['batch_size'] = LAM_BATCH
    cfg['max_batch_size'] = LAM_BATCH
    cfg['grow_batch_size'] = False
    cfg['batch_sizer_retest_steps'] = 0      # no re-measure, so the batch cannot drift apart
    cfg['batch_util_target'] = 0.0           # occupancy target off: four arms share the card
    # PER-ARM COST, trimmed because four of these run at once on one card.
    # compile_policy takes false | true | 'auto' | 'step'. ANY OTHER STRING IS TRUTHY:
    # maybe_compile_policy falls through to `enable = bool(setting)`, so 'off' would turn
    # compilation ON, here on Windows where inductor has no CUDA backend -- dynamo then
    # fails per frame and degrades to eager with a traceback apiece. False is the off switch.
    cfg['compile_policy'] = False
    cfg['eval_period'] = 250                 # was 100
    cfg['eval_num_samples'] = 2500           # was 10000
    cfg['test_eval_num_samples'] = 1000
    cfg['figs_period'] = 2000                # was 200
    cfg['archive_period'] = 0                # no GB-scale buffer sidecars from a test arm
    cfg['archive_buffers'] = False
    # NB eval_num_samples sets the w1r resolution floor (w1r/perfect_*), so these arms'
    # w1r numbers are comparable to EACH OTHER but not to phase 1's, which ran at 10000.

    ec = cfg['energy_config']
    ec['prior_flow_path'] = FLOW
    ec['lambda_mix'] = float(lam)
    # PHYSICAL-LEG TAIL CLIP. log-compresses the crystal term inside the physical leg only,
    # so the lambda=0 endpoint is still exactly the flow. The reason it matters here:
    # gflownet_losses.py records that d log R / d x_T for an LJ-type energy is near-singular
    # whenever atoms clash, and lambda is what puts that term in the differentiated graph at
    # all -- at lambda 0 generator_energy returns the flow leg untouched and the physical
    # graph is never built. Precedent: qm9c_hold_l0p018_fwdseat_pbfrozen_pclip600 at 600.
    if pclip is not None:
        ec['physical_energy_clip'] = float(pclip)
    # generator_energy RAISES when a flow and energy_clip are both live, so reward_range
    # must stay null on every arm that carries a flow.
    assert ec['reward_range'] is None, ec['reward_range']

    stages = cfg['protocols'][cfg['protocol']]['stages']
    # The arm must ENTER var_conditioning at step 0, not train MLE again. Both halves of
    # that are asserted: the skip condition is still declared on the stage ahead of it,
    # and nothing here samples the prior model the skip is keyed on.
    tp = [s for s in stages if s['name'] == 'train_prior'][0]
    assert tp.get('skip_if') == 'prior_loaded', (
        f'{name}: train_prior lost skip_if -- the arm would re-run the MLE stage')
    assert stages.index(tp) == 0 and stages[1]['name'] == 'var_conditioning', \
        [s['name'] for s in stages]
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors', \
        'prior_buffer would sample the frozen prior model this arm only loads to trip the skip'
    stage = [s for s in stages if s['name'] == 'var_conditioning'][0]
    # THE FORWARD SEAT, asserted rather than written: fwd trains the policy and is the
    # pooled source, replay carries nothing, and there is no Z sidecar.
    assert stage['fracs'] == {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}, stage['fracs']
    assert stage['fwd_z_sidecar'] is False and stage['fwd_rollout_every'] == 0
    assert 'condition_draw' not in stage, 'condition_draw belongs to the replay seat'
    fwd = stage['loss_coeffs']['fwd']
    assert float(fwd['freeze_policy']) == 0.0 and fwd['pooled_source'] == 'fwd', fwd
    # P_B FREEZE IS EXPLICIT, never inherited. The base carries no freeze_pb. Generation b
    # ran it UNFROZEN at the owner's request and that alone moved grad_norm_pre_clip from
    # 404 to 719 and the forward/backward variances by ~4% -- large enough that a frozen and
    # an unfrozen arm are not comparable, so the flag is stated per arm and asserted.
    on_enter = stage.setdefault('on_enter', [])
    if freeze_pb:
        if 'freeze_pb' not in on_enter:
            on_enter.append('freeze_pb')
    assert ('freeze_pb' in on_enter) == bool(freeze_pb), on_enter

    # THE FORCE, on the run-level fwd block (set_loss_coeffs writes the stage's over it,
    # and the stage names none of these keys, so the run-level value is what reaches the loss).
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = int(k)
    fc['reward_grads'] = float(rg)
    fc['reward_grad_clip'] = 0.0
    fc['reward_grad_gate'] = None if gate is None else float(gate)
    fc['reward_grad_force_clip'] = float(force_clip)
    # PATH_GRAD_SCALE. 1 = the live step's path gradient reaches its noise SCALE as well as
    # its mean; 0 detaches the scale on the path route only (its density-route gradient is
    # untouched). The two channels of d f_t/d theta carry dt and sqrt(dt), so the scale sees
    # sqrt(dt)/dt = 1/sqrt(dt) times the drift's share -- 5x at T=25. A terminal force
    # therefore widens the policy far more than it moves it, which is what cl21b_lam0p01_f
    # showed (fwd step_var +11%, Mean F Var +17%, while logw_std went DOWN).
    fc['path_grad_scale'] = int(path_grad_scale)
    fc['traj_grads'] = 0.0
    # the reshaped path needs a live terminal and a trained policy; the plain path needs
    # a live terminal too, or reward_grads is inert (the reward runs under no_grad).
    if float(rg) != 0.0:
        assert int(k) > 0, f'{name}: reward_grads without path_grad_last_k is inert'
    if gate is not None or float(force_clip) > 0:
        assert float(rg) != 0.0 and int(k) > 0, f'{name}: reshape keys need reward_grads and a live terminal'

    h = problem_hash_of(cfg)
    assert h == p1_hash, f'{name}: problem hash {h} != phase 1\'s {p1_hash}'
    out = write(cfg, name)
    print(f'wrote {out.name:12s} lambda={lam}  k={k}  reward_grads={rg}  gate={gate}  '
          f'force_clip={force_clip}  steps={steps}  lr_scale={LR_SCALE}  hash ok')
    return out


def make_lam(seed_name, with_clip, lambdas, pclip=None, force_clip_only=None, suffix='',
             ps0_only=False, freeze_pb=False, steps=LAM_STEPS):
    seed, p1_hash, step = resolve_seed(seed_name)
    blob, prov = check_flow(seed, p1_hash)
    print(f'seed  {seed}  step {step}  hash {p1_hash}')
    print(f'flow  {os.path.basename(FLOW)}  T={blob["traj_T"]}  '
          f'{prov.get("n_samples")} draws from {prov.get("checkpoint")}')
    for lam in lambdas:
        tag = 'lam0' if lam == 0 else 'lam' + ('%g' % lam).replace('0.', '0p')
        if pclip is not None:
            # NOT COMPARABLE TO THE OTHER ARMS. physical_energy_clip is absent from
            # utils.py::_NON_IDENTITY_ENERGY_CONFIG_KEYS -- unlike reward_range and
            # energy_clip, which are exempt there as tail reshapes -- so setting it is a
            # different problem_def and a different hash. arm()'s hash assertion refuses
            # it on purpose. Use --force-clip, which is a loss coefficient and leaves the
            # identity alone, or lift the exemption deliberately first.
            arm(f'{tag}_fp', lam, 1, 1.0, seed, p1_hash, pclip=pclip)
            continue
        if ps0_only:
            arm(f'{tag}_ps0', lam, 1, 1.0, seed, p1_hash, path_grad_scale=0,
                freeze_pb=freeze_pb, steps=steps)
            continue
        if force_clip_only is not None:
            # follow-up: the force arm again with |F| bounded per row, ONE variable off
            # lam0p01_f. Same problem hash, so it is directly comparable to it.
            arm(f'{tag}_f{suffix}', lam, 1, 1.0, seed, p1_hash, force_clip=force_clip_only)
            continue
        arm(f'{tag}_ctrl', lam, 0, 0.0, seed, p1_hash, freeze_pb=freeze_pb, steps=steps)
        arm(f'{tag}_f', lam, 1, 1.0, seed, p1_hash, freeze_pb=freeze_pb, steps=steps)
        if with_clip:
            arm(f'{tag}_fc', lam, 1, 1.0, seed, p1_hash, force_clip=FORCE_CLIP)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('leg', choices=['p1', 'seeds', 'lam'])
    ap.add_argument('--seed', help='the phase-1 checkpoint to train the lambda arms from, '
                                   'by basename in the checkpoints dir. Required for `lam`; '
                                   '`seeds` lists the candidates with their steps.')
    ap.add_argument('--with-clip', action='store_true',
                    help='also write the |F|-clipped arm (set FORCE_CLIP from l0_f\'s '
                         'rewardgrad/force_norm_* before using it)')
    ap.add_argument('--lambdas', type=float, nargs='+', default=[0.0],
                    help='one pair of arms per value; 0.0 is the null leg')
    ap.add_argument('--freeze-pb', action='store_true', dest='freeze_pb',
                    help='append freeze_pb to the stage on_enter (P_B frozen at entry)')
    ap.add_argument('--steps', type=int, default=LAM_STEPS,
                    help=f'epochs per arm (default {LAM_STEPS})')
    ap.add_argument('--ps0', action='store_true', dest='ps0_only',
                    help='write only <tag>_ps0: the force arm with path_grad_scale 0')
    ap.add_argument('--suffix', default='c',
                    help="name suffix for the --force-clip arm; default 'c' gives <tag>_fc")
    ap.add_argument('--force-clip', type=float, default=None, dest='force_clip_only',
                    help='follow-up: write only <tag>_fc, the force arm with reward_grad_force_clip '
                         'at this value. A loss coefficient, so the problem hash is unchanged.')
    ap.add_argument('--pclip', type=float, default=None,
                    help='energy_config.physical_energy_clip on the force arm; names it _fp')
    a = ap.parse_args()
    if a.leg == 'p1':
        make_p1()
    elif a.leg == 'seeds':
        rows = p1_checkpoints()
        if not rows:
            raise SystemExit(f'no {P1_TAG}_{P1_RUN}_*.pt in {CKPT_DIR} yet')
        print(f'{"checkpoint":70s} {"step":>7}  {"stage":16s} {"T":>3}')
        for n, s, st, t in sorted(rows, key=lambda r: r[1]):
            print(f'{n:70s} {s:>7}  {str(st):16s} {t:>3}')
    else:
        if not a.seed:
            raise SystemExit('lam needs --seed: pick the convergence point yourself '
                             '(`make.py seeds` lists them). There is no default.')
        make_lam(a.seed, a.with_clip, a.lambdas, pclip=a.pclip,
                 force_clip_only=a.force_clip_only, suffix=a.suffix,
                 ps0_only=a.ps0_only, freeze_pb=a.freeze_pb, steps=a.steps)
