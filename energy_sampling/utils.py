import argparse
import gc
import hashlib
import json
import math
import os
import random
from argparse import Namespace
from asyncio import sleep
from pathlib import Path
from typing import Optional

import PIL
import numpy as np
import psutil
import torch
import yaml
from scipy.stats import median_abs_deviation
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mxtaltools.common.adaptive_batching import adaptive_batched_analysis
from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.common.geometry_utils import batch_molecule_principal_axes_torch, simple_latent_distance
from mxtaltools.common.geometry_utils import compute_latent_distance
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
from mxtaltools.models.utils import load_encoder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def dcp(tensor):
    return tensor.detach().cpu()


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow,
                      conditional_flow_model=False,
                      use_weight_decay=False, weight_decay=1e-7):
    param_groups = [{'params': gfn_model.t_model.parameters()},
                    {'params': gfn_model.s_model.parameters()},
                    {'params': gfn_model.forward_policy.parameters()},
                    {'params': gfn_model.backward_policy.parameters()},
                    {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow}
                    ]

    if conditional_flow_model:
        param_groups += [{'params': gfn_model.conditions_embedding_model.parameters(),
                          'lr': lr_policy}]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer


def get_exploration_std(iter, exploratory, max_steps: int = 5000, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        exploration_factor = 0
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - iter / max_steps)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl


def get_train_args():
    parser = argparse.ArgumentParser(description='GFN Linear Regression')
    args, remaining = parser.parse_known_args()

    return resolve_derived_config(preflight_config(dict2namespace(load_yaml(remaining[1]))))


# ---------------------------------------------------------------------------
# Derived-config resolution. Values that are pure functions of the primitives
# (W = model hidden width, T = integrator.T) are computed here instead of being
# hand-materialized in the YAML, so they can never drift out of sync.
#
# `auto` MEANS TWO DIFFERENT THINGS and the difference is deliberate:
#   gradient_norm_clip: auto  -> DERIVED from (W, T) by the anchor rule below.
#   lr_*:               auto  -> SERVO-MANAGED. Seeded low and then owned by the
#                                alpha* loop in controller.py, which probes the
#                                usable peak per stage. An explicit float is a
#                                FIXED peak the servo never touches.
#
# The old T-scaling rule for the LRs (anchor x 25/T) is GONE. It was a single
# battery's number promoted to a law -- aug02 measured one point, at one energy,
# one T, one W, one clip -- and decisions.md D3 (revisited) records why that
# generalisation fails: the problem shifts constantly, so there is no stable
# `here` for an anchor to be anchored to. Every run is a transfer. The servo
# replaces it with a per-run measurement.
# ---------------------------------------------------------------------------
_SCALING_W_REF = 512
_SCALING_T_REF = 25
# Seed for a servo-managed (`auto`) LR. Deliberately BELOW every measured
# optimum on this codebase (aug02 4e-4 at T=25, ty4xdlzo ~8e-4, local_aug08
# ~1.25e-4 at T=10) rather than near one: alpha* > 1 is affirmative permission
# to grow, so the servo climbs out of an undershoot cheaply and on evidence,
# whereas it can only learn about an overshoot by taking one. Seeding low costs
# a few thousand steps of climb; seeding high costs a detonation.
_SERVO_SEED_LR = 1.0e-5
_LR_KEYS = ('lr_policy', 'lr_replay', 'lr_back', 'lr_fused')
_CLIP_ANCHOR = 250.0
_GRAD_MEDIAN = {10: 1.0e3, 25: 6.6e3, 100: 1.7e4}  # empirical pre-clip grad medians (mipcas)

# Keys deleted from the schema, with the reason the config author needs. Checked
# at LOAD, not at first use: the aug02 battery lost all 16 arms' entire phase 1
# (1.1-7.8 h each) because a retired-key guard lived inside manage_replay_buffer,
# which first runs at the phase-1 -> 2 transition (module_buffers.md B4).
_RETIRED_KEYS = {
    'gpu_util_floor':
        "deleted 2026-08-13 -- the occupancy rule it drove is gone from "
        "increment_batch_size. It grew the batch whenever GPU utilization fell under "
        "this threshold, on the premise that occupancy rises with batch. MEASURED "
        "FALSE on the MLIP route (umaperf0812 c_controller): batch 100->741 took "
        "utilization 52->42% and samples/sec 57.7->24.3, four growths, every one "
        "driven by this floor, the throughput gate refusing all of them and being "
        "overridden because the floor outranked it. It was also inert on the slow "
        "arms it was written for -- a 900 s window cannot fill from a per-10-step "
        "sample at 200 s/step. Occupancy is still LOGGED (gpu/util_recent, "
        "gpu/util_policy); it just is not actuated from batch size, because the "
        "levers that move it are work per kernel launch and unpaired host stalls.",
    'batch_growth_max_step_regression':
        "deleted -- replaced by `batch_growth_min_throughput_gain`. It bounded how much "
        "SLOWER a step may get, i.e. it optimised loop-iterations/hour. That is not the "
        "objective: with the update size pinned at fused_grad_accum_min_samples, "
        "optimizer-steps/sec = samples_per_sec / accum_target, so opt-step throughput IS "
        "samples/sec and step time does not enter. The regression gate rejected jumps "
        "that bought +15% samples/sec for a 43% slower step -- a 15% win scored as a "
        "loss. The gate is now throughput saturation alone; max_step_seconds survives "
        "as a far runaway guard.",
    'batch_growth_min_gain':
        "deleted -- replaced by `batch_growth_max_step_regression`. It asked for a "
        "fixed samples/sec gain, which is not a knee criterion: at growth factor f a "
        "rung returning exactly min_gain has grown step time by f/(1+min_gain), so the "
        "shipped 1.65/0.15 pair ACCEPTED every jump that made steps 43% slower and "
        "walked prod0810's batch to 12226 before OOM stopped it. The replacement bounds "
        "the step-time regression directly (accept iff t_new/t_old <= 1 + tol).",
    'step_probe':
        "deleted with the online alpha* servo (controller v8) -- replaced by the "
        "`ray_calibration` block. The online probe estimated alpha* as a ratio whose "
        "denominator straddled zero (per-probe sd/mean ~3.3), which is what forced the "
        "censoring, windowing and quorum machinery that has now gone with it.",
    'adaptive_lr.servo':
        "deleted (controller v8). seed_lr moved to adaptive_lr.seed_lr and bounds to "
        "adaptive_lr.bounds; target/clip/period/min_readings/max_bad_rate belonged to "
        "the online median servo, which no longer exists.",
    'adaptive_lr.trigger':
        "deleted (controller v8) -- the quorum-below-a-bar trigger. Its statistic put "
        "censored readings in the denominator but never the numerator, so its quorum "
        "became unreachable exactly as the LR got dangerous (lrdisc v1).",
    'adaptive_lr.boost':
        "deleted (controller v8) -- growth now comes only from the periodic calibration.",
    'adaptive_lr.discovery':
        "deleted (controller v8) -- the ramp/cruise state machine. Periodic "
        "recalibration provides re-discovery without a state machine.",
    'adaptive_lr.damage':
        "deleted (controller v8) -- the relative fwd/tb_err tripwire. Between "
        "calibrations only catastrophic protection remains, by design: a loss-based "
        "hot indicator on this problem fires too late to be a useful LR sensor.",
    'reuse_prior':
        "deleted -- point checkpoint_name/prior_model_name at the prior explicitly, "
        "or let the train_prior stage rebuild it. Auto-refinding a prior by run_name "
        "silently reloaded pre-fix checkpoints across scoring changes.",
    'terminal_logw_std':
        "deleted with _terminal_policy_state -- _frozen_training_state (bitwise-"
        "constant grad norm) separates dead from live runs with no calibration.",
    'terminal_box_violation': "deleted with _terminal_policy_state (see terminal_logw_std).",
    'adaptive_lr.enabled':
        "deleted -- the envelope is unconditional. To run flat LRs set "
        "lr_warmup_ratio: 1 (which makes the ramp a no-op).",
    'adaptive_lr.hold_steps':
        "deleted with the decay leg -- the envelope now holds at 1.0 forever.",
    'adaptive_lr.decay_halflife_steps':
        "deleted (decisions.md D7/N2) -- there is no step budget to schedule "
        "against, and under the servo a deterministic multiplier on the LR is "
        "absorbed: alpha* rates peak x envelope, so the servo just raises peak.",
    'adaptive_lr.decay_floor_scale':
        "deleted -- redundant with min_lr, which already floors every group in "
        "_apply_lrs, and the decay leg it floored is gone.",
    'adaptive_lr.cut_loss_abs':
        "deleted with the cut tier (decisions.md D4). Use adaptive_lr."
        "divergence_loss_abs for the coarse ~1e9 reload bar.",
    'adaptive_lr.cut_grad_abs':
        "deleted with the cut tier -- see adaptive_lr.divergence_grad_abs.",
    'adaptive_lr.reset_loss_abs': "renamed -> adaptive_lr.divergence_loss_abs.",
    'adaptive_lr.reset_grad_abs': "renamed -> adaptive_lr.divergence_grad_abs.",
    'adaptive_lr.cut_ratio': "renamed -> adaptive_lr.divergence_cut.",
    'adaptive_lr.fire_cooldown_steps':
        "deleted -- no cooldown is needed on a bar that only non-finite values "
        "and 1e9 blow-ups can reach.",
    'adaptive_lr.recovery_target_frac':
        "deleted with the recovery ramp -- the servo re-climbs continuously on "
        "alpha*, so there is nothing to recover from a latch.",
    'adaptive_lr.recovery_wait_steps': "deleted with the recovery ramp.",
    'adaptive_lr.recovery_ramp_steps': "deleted with the recovery ramp.",
    'adaptive_lr.servo.ceiling_halflife_steps':
        "deleted -- the divergence ceiling is permanent for the run. The "
        "discovery ramp re-tests the edge on sensor evidence, so relaxing the "
        "same cap on a timer was a second mechanism with nothing measuring it.",
    'adaptive_lr.servo.trigger':
        "moved -> adaptive_lr.trigger. The trigger both stops a ramp and brakes "
        "during cruise, so it is not a property of the servo's mechanics.",
    'adaptive_lr.servo.discovery': "moved -> adaptive_lr.discovery.",
    'integrator.min_traj_length':
        "deleted -- the discretizer is always uniform with a static strategy, so "
        "T is the only trajectory-length parameter.",
    'integrator.max_traj_length': "deleted (see integrator.min_traj_length).",
    'integrator.traj_length_strategy': "deleted -- always 'static'.",
    'integrator.discretizer': "deleted -- always 'uniform'.",
    'integrator.discretizer_max_ratio': "deleted -- only the 'random' discretizer read it.",
    'max_reloads':
        "renamed -> max_reloads_per_1k_steps, and it is now a RATE. A fixed "
        "count aborts a long run for the same per-step behaviour a short one "
        "survives.",
    'terminal_frozen_steps':
        "deleted with _frozen_training_state. Its job -- catching an "
        "unrecoverable run -- is covered by the divergence bar plus the rewind "
        "budget (max_reloads_per_1k_steps).",
    'z_calibration.unclipped':
        "deleted. The Z sidecar now always uses the fwd TB loss's own beta, so "
        "sensor and actuator share a fixed point by construction.",
    'buffers.anchor_buffer.health_gate_r2':
        "renamed -> health_gate_floor (the bar is named after its role, not after "
        "whichever metric health_gate_floor_metric currently names).",
    'buffers.anchor_buffer.health_gate_zerr':
        "renamed -> health_gate_ceiling. NB a bar does not survive a ruler swap: "
        "0.5 belongs to tb_resid_clipped (signed, beta-bounded, derived from the D29 "
        "Z-currency invariant), not to tb_err_worst (an unbounded RMS ~18-21 when healthy).",
    'buffers.anchor_buffer.mcmc':
        "deleted -- the Metropolis reheat is archived in git history "
        "(anchor_seed_mcmc_reheat); anchors seed from prior_dataset.",
    'buffers.replay_buffer.admit_cap_max':
        "deleted (decisions.md D5, to_do_rebuild.md Phase 3 step 3) -- replay "
        "admission is now unconditionally uniform over the sane pool. "
        "Selection moved entirely to the draw (buffers.replay_buffer.prioritise), "
        "which already carries an IS correction; a residual-scored admission cap "
        "would re-enter the force spectrum uncorrected on top of it.",
    'buffers.replay_buffer.admit_cap_min': "deleted (see admit_cap_max).",
    'buffers.replay_buffer.admit_cap_health_h0': "deleted (see admit_cap_max).",
    'buffers.replay_buffer.admit_temperature':
        "deleted (see admit_cap_max) -- admission and the displacement purge "
        "both draw uniformly now, so there is no softmax temperature left to set.",
}


def _walk_key(args, dotted):
    """(container, leaf, present) for a dotted config path, or (None, leaf,
    False) if any ancestor is missing."""
    node = args
    parts = dotted.split('.')
    for p in parts[:-1]:
        node = getattr(node, p, None)
        if node is None:
            return None, parts[-1], False
    return node, parts[-1], hasattr(node, parts[-1])


def _check_state_version(args):
    """Check the config's declared project state against this code's.

    SELF-ARMING SEVERITY. While CURRENT_STATE_VERSION is 1 there is nothing for a
    migration to do beyond writing the stamp, so an unstamped config gets a notice
    and runs. The moment a real transition ships, the same unstamped config is
    genuinely stale -- it predates a change whose migration exists -- and this
    becomes fatal without anyone having to remember to tighten it.

    A config AHEAD of the code is always fatal: it may carry keys this revision has
    no schema for, and there is no downgrade path."""
    from config_state import CURRENT_STATE_VERSION, UNSTAMPED_VERSION, VERSION_KEY

    declared = getattr(args, VERSION_KEY, None)
    version = UNSTAMPED_VERSION if declared is None else int(declared)

    if version > CURRENT_STATE_VERSION:
        raise ValueError(
            f"config declares {VERSION_KEY}: {version}, ahead of this code's state "
            f"{CURRENT_STATE_VERSION}. It may carry keys with no schema here. Check "
            f"out the revision that matches, rather than running it against older code.")

    if version < CURRENT_STATE_VERSION:
        msg = (f"config is at {VERSION_KEY} {version}, current is "
               f"{CURRENT_STATE_VERSION}. Migrate it:\n"
               f"    python -m config_state migrate <config.yaml>")
        if CURRENT_STATE_VERSION > 1:
            raise ValueError(msg)
        print(f"NOTE: {msg}\n  (not fatal at state 1 -- the migration only adds the "
              f"stamp. It becomes fatal once a real transition ships.)")

    setattr(args, VERSION_KEY, version)
    return args


def _namespace_to_dict(ns):
    """Namespace tree -> plain dict, for the invariant checks (which are pure
    functions of a raw config dict so they can also run over YAML on disk)."""
    from argparse import Namespace
    if isinstance(ns, Namespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, (list, tuple)):
        return [_namespace_to_dict(v) for v in ns]
    return ns


def _report_config_invariants(args):
    """Print any violated config invariant. REPORTS, does not raise.

    Deliberately not a gate, for now. Of the 2,244 configs in the tree, 47 set a
    `figs_period` that is not a multiple of `eval_period` (so they logged no
    figures at all) and 119 run DPLR unmasked with angular dims (so they die at
    model construction anyway). Making this fatal would block reruns of batteries
    that are otherwise valid, to report a fault the run either already survives
    or already dies of.

    The gate belongs at GENERATION, where a bad config can be fixed before it
    costs anything: the production-config generator treats ERROR as fatal.
    Printing here means an existing config still says so out loud, at load,
    rather than at hour three."""
    try:
        import config_invariants
        violations = config_invariants.check(_namespace_to_dict(args))
    except Exception as e:      # never let a diagnostic stop a run
        print(f'config invariant check skipped ({type(e).__name__}: {e})')
        return
    if not violations:
        return
    print('config invariants:')
    for v in violations:
        print(f'  {v}')


def preflight_config(args):
    """Reject retired keys at load, before a single energy call is spent.

    Hard failure rather than a warning, deliberately. A retired key in a config
    is almost always a value the author believes is doing something: silently
    ignoring `cut_grad_abs` leaves them reading an LR ladder whose brake they
    think they set. The message names the replacement so the fix is mechanical."""
    dead = []
    for dotted, why in _RETIRED_KEYS.items():
        node, leaf, present = _walk_key(args, dotted)
        if present:
            dead.append(f"  {dotted}: {why}")
    if dead:
        raise ValueError(
            "retired config keys found:\n" + "\n".join(dead)
            + "\n\nThe mechanical subset of these can be repaired automatically:\n"
              "    python -m config_state migrate <config.yaml>\n"
              "which renames what is renameable, drops what is dead, and REPORTS "
              "(without changing) the keys whose interpretation moved.")

    _check_state_version(args)
    _report_config_invariants(args)

    # Integration time must agree between training and eval. The stab_july21
    # elj battery ran eval_T = 2T and floored wass_debiased on what turned out
    # to be an integration-dt mismatch artifact, not a modelling result: the
    # policy learns a drift per step at ONE dt, so scoring it at another
    # changes the SDE being integrated. There is no legitimate reason to differ.
    T = int(getattr(getattr(args, 'integrator', None), 'T', 0) or 0)
    eval_T = getattr(args, 'eval_T', None)
    if T and eval_T is not None and int(eval_T) != T:
        raise ValueError(
            f"eval_T ({eval_T}) must equal integrator.T ({T}). The policy's drift and "
            f"variance heads are learned at one dt; evaluating at another integrates a "
            f"different SDE, and the resulting wass/r2 numbers are a dt artifact rather "
            f"than a measurement of the model.")
    return args


def _is_auto(v):
    """A config value the resolver should fill in: absent, null, or 'auto'."""
    return v is None or (isinstance(v, str) and v.strip().lower() == 'auto')


def _grad_median(T):
    """Empirical pre-clip grad-norm median at rollout length T, log-log
    interpolated over _GRAD_MEDIAN (and extrapolated past the table ends via the
    nearest segment). Drives the width/length scaling of gradient_norm_clip."""
    ts = sorted(_GRAD_MEDIAN)
    if T in _GRAD_MEDIAN:
        return _GRAD_MEDIAN[T]
    if T < ts[0]:
        lo, hi = ts[0], ts[1]
    elif T > ts[-1]:
        lo, hi = ts[-2], ts[-1]
    else:
        lo = max(t for t in ts if t <= T)
        hi = min(t for t in ts if t >= T)
    f = (math.log(T) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(_GRAD_MEDIAN[lo]) + f * (math.log(_GRAD_MEDIAN[hi]) - math.log(_GRAD_MEDIAN[lo])))


def resolve_derived_config(args):
    """Fill in (W, T)-derived config values from the primitives, in place, and
    return args. See the block comment above. Idempotent given fixed primitives;
    logs what it derived so the run record shows resolved numbers, not `auto`."""
    integrator = getattr(args, 'integrator', None)
    model = getattr(args, 'model', None)
    T = int(getattr(integrator, 'T', 0) or 0)
    W = int(getattr(model, 'policy_hidden_dim', 0) or 0)  # canonical width (all *_hidden_dim expected equal)
    if not T or not W:
        return args  # no primitives to scale from -- leave everything as written

    resolved = {}

    # LRs: `auto` hands the key to the servo, seeded low. The set of managed
    # keys is recorded on args because _apply_lrs needs to know which groups
    # peak_scale applies to -- once resolved, `auto` and an explicit float are
    # indistinguishable as values, and that distinction IS the semantics.
    seed_lr = float(getattr(getattr(args, 'adaptive_lr', None), 'seed_lr', None)
                    or _SERVO_SEED_LR)
    managed = []
    for name in _LR_KEYS:
        if _is_auto(getattr(args, name, None)):
            setattr(args, name, seed_lr)
            resolved[name] = seed_lr
            managed.append(name)
    args.lr_servo_managed = tuple(managed)
    if managed:
        # `auto` means SERVO-MANAGED now. Silently resolving it to the seed with
        # no servo to move it would leave the run training at 1e-5 forever while
        # the config still says `auto` -- a config that reads as "let the system
        # choose" and behaves as "pinned an order of magnitude low". That is the
        # exact class of silent change the retired-key preflight exists to stop,
        # so it is an error rather than a default.
        cal = getattr(args, 'ray_calibration', None)
        if cal is None or not getattr(cal, 'enabled', False):
            raise ValueError(
                f"{', '.join(managed)} set to `auto` (controller-managed) but "
                f"ray_calibration is absent or disabled. `auto` means the periodic ray "
                f"calibration OWNS this LR; with no sensor it would sit at the seed "
                f"{seed_lr:.3g} forever while the config claims it is adaptive. Either "
                f"enable ray_calibration or write an explicit float for a fixed LR.")

    # grad clip: anchor x grad_median(T)/grad_median(T_REF) x sqrt(W/W_REF).
    # This one IS still derived -- it scales with the gradient's own measured
    # magnitude, which is a property of the rollout length rather than of a
    # tuning choice, and nothing servos it.
    if _is_auto(getattr(args, 'gradient_norm_clip', None)):
        clip = (_CLIP_ANCHOR
                * (_grad_median(T) / _GRAD_MEDIAN[_SCALING_T_REF])
                * math.sqrt(W / _SCALING_W_REF))
        args.gradient_norm_clip = clip
        resolved['gradient_norm_clip'] = clip

    if resolved:
        summary = ', '.join(f'{k}={v:.4g}' for k, v in resolved.items())
        print(f'resolve_derived_config (W={W}, T={T}): {summary}')
    return args


def load_yaml(path):
    """
    Safely load yaml file as dict.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    # encoding pinned: the default is the LOCALE's (cp1252 on Windows), so a
    # config carrying any non-ASCII byte -- a mu, a degree sign, an accented
    # name in a path -- raises UnicodeDecodeError on one machine and loads fine
    # on another.
    with yaml_path.open("r", encoding="utf-8") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def _to_plain(obj):
    """Recursively turn Namespaces/lists into plain, JSON-serializable Python objects."""
    if isinstance(obj, Namespace):
        obj = vars(obj)
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj



# energy_config fields excluded from the problem definition so changing them
# doesn't invalidate checkpoint reuse: the coeffs because they're never varied
# in practice, reward_range because it's a reward-shaping/training knob layered
# on top of the same energy landscape rather than part of its identity,
# internal_oom_recovery because it only chooses HOW a batch is fed to the energy
# (one call vs adaptive sub-batching), not what the energy is -- and flipping it
# would otherwise re-hash every slug and orphan the prod0810 phase1_exit
# checkpoints. normalize_problem_def drops these from BOTH sides of a
# compatibility check, so checkpoints saved before an entry was added still load.
#
# host_gas_phase_reference is the one entry here that DOES move numbers, so it is
# worth being explicit about why it still is not identity. The isolated-molecule
# leg of a lattice energy is mathematically constant for a rigid molecule; hosting
# it replaces a per-sample recomputation with one reference value, so what changes
# is the MLIP's rotation-invariance NOISE, not the energy landscape being sampled.
# Treating that as a different problem would orphan every checkpoint over a
# numerical-precision improvement. Measure the size of it with
# MolecularCrystal.gas_reference_audit before assuming it is small on a new
# molecule set.
_NON_IDENTITY_ENERGY_CONFIG_KEYS = ('density_coeff', 'bounding_coeff', 'reduction_coeff', 'lj_coeff',
                                    'reward_range', 'internal_oom_recovery',
                                    'host_gas_phase_reference')

# Explicit version of the problem_def SCHEMA (the set of fields below that
# constitute a problem's identity). It rides in the dict and therefore in the
# hash, so a bump cleanly orphans every prior on disk with a legible diff line
# ("schema_version: stored N vs current N+1") instead of a silent field-by-field
# mismatch. BUMP THIS whenever the fields of get_problem_definition change --
# that is the whole point: adding a field silently (vec_cond, 2026-07-21)
# orphaned a battery's shared priors with no signal. FREEZE it (leave it be)
# before baking a one-for-all prior meant to be reused across many runs.
PROBLEM_DEF_SCHEMA_VERSION = 1


def get_problem_definition(args) -> dict:
    """
    The subset of the config that defines *what problem* is being solved -
    the energy landscape and prior distribution - as opposed to training/
    optimization hyperparameters (lr, batch size, model width, etc.) which
    can change across runs without changing what a checkpoint represents.

    Omits sg_conditioning/zp_conditioning: they're redundant with
    len(space_groups) > 1 / len(z_primes) > 1, which are already captured here.
    """
    energy_config = _to_plain(args.energy_config)
    for key in _NON_IDENTITY_ENERGY_CONFIG_KEYS:
        energy_config.pop(key, None)

    definition = {
        # explicit schema version -- bump when adding/removing a field below
        # (see PROBLEM_DEF_SCHEMA_VERSION); freeze before baking a shared prior
        'schema_version': PROBLEM_DEF_SCHEMA_VERSION,
        'energy_function': args.energy_function,
        'energy_config': energy_config,
        'prior_path': args.prior_path,
        'space_groups': args.space_groups,
        'z_primes': args.z_primes,
        'mol_cond': args.molecule_conditioning,
        'temp_cond': args.temperature_conditioning,
        # a vector-conditional policy expects `c` inputs an unconditional
        # problem doesn't provide, so the flag is part of the identity
        'vec_cond': getattr(args, 'vector_conditioning', False),
    }

    # Pre-embedded molecule conditioning belongs to the identity for the same reason
    # vec_cond does: the policy expects per-molecule embedding inputs an unconditional
    # problem never supplies. It is added ONLY WHEN ON, rather than as an always-present
    # False, deliberately -- every key here feeds problem_hash, so writing it
    # unconditionally would change the hash of every existing run and orphan checkpoints
    # that are otherwise still valid (there are production runs queued against them). A
    # run that never sets the flag keeps the exact identity it had before this key existed.
    if getattr(args, 'embedding_conditioning', False):
        definition['emb_cond'] = True
        definition['emb_cond_dim'] = getattr(args, 'embedding_conditioning_dim', None)

    return _to_plain(definition)


def normalize_problem_def(problem_def):
    """
    Strip the non-identity energy_config keys from a problem_def dict.
    Freshly built defs never contain them (get_problem_definition pops them),
    but defs stored inside checkpoints keep whatever the exclusion list looked
    like at save time - so compatibility checks must normalize BOTH sides
    before comparing, or growing the exclusion list would orphan every
    checkpoint saved before the key was excluded.
    """
    if not isinstance(problem_def, dict):
        return problem_def
    normalized = dict(problem_def)
    if isinstance(normalized.get('energy_config'), dict):
        energy_config = dict(normalized['energy_config'])
        for key in _NON_IDENTITY_ENERGY_CONFIG_KEYS:
            energy_config.pop(key, None)
        normalized['energy_config'] = energy_config
    return normalized


def problem_hash(problem_def: dict, n_chars: int = 6) -> str:
    """Short, deterministic fingerprint of a problem definition dict."""
    canonical = json.dumps(problem_def, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:n_chars]


def problem_slug(args, problem_def: dict) -> str:
    """
    Human-readable tag for a checkpoint's problem definition, meant for
    filenames only: <energy_function>-<prior file stem>-T<temperature>-<hash>.
    The trailing hash covers the whole problem definition (including fields
    not spelled out in the slug), but this string is never parsed back -
    the checkpoint itself stores the full 'problem_def' dict plus its hash,
    and that stored copy is what compatibility checks compare against, so
    the slug format above is free to change later without breaking reload logic.
    """
    prior_stem = Path(args.prior_path).stem if args.prior_path else 'noprior'
    temperature = args.energy_config.temperature
    return f"{args.energy_function}-{prior_stem}-T{temperature:g}-{problem_hash(problem_def)}"


def get_gfn_init_state(batch_size, ndim, device):
    init_state = torch.zeros(batch_size, ndim).to(device)

    return init_state


def uniform_discretizer(bsz, trajectory_length):
    return torch.linspace(0, 1, trajectory_length + 1).repeat(bsz, 1)


def compute_sample_overlap(ref_x, sample_x=None, ga: float = 1.0, agg='sum'):
    if sample_x is None:
        d = torch.cdist(ref_x, ref_x) + torch.eye(len(ref_x), device=ref_x.device) * 100
    else:
        d = torch.cdist(ref_x, sample_x)

    if agg == 'sum':
        return torch.exp(-ga * d ** 2).sum(dim=0)
    elif agg == 'mean':
        return torch.exp(-ga * d ** 2).mean(dim=0)


def smoothstep(x, t, delta):
    if not torch.is_tensor(x):
        x = torch.tensor([x])
    x_clipped = torch.clamp((x - t) / delta, 0.0, 1.0)
    return float(x_clipped ** 2 * (3 - 2 * x_clipped))  # smoothstep polynomial


def triangle_schedule(it, init, maxval, minval, on, off):
    if it <= on:
        # ramp up
        frac = (it / on) if on > 0 else 1.0
        return init * (1 - frac) + maxval * frac
    elif on < it <= off:
        # ramp down
        frac = (it - on) / (off - on) if (off - on) > 0 else 1.0
        return maxval * (1 - frac) + minval * frac
    else:
        return minval


@torch.no_grad()
def featurize_dataset(dataset, device, energy_function: str, batch_size: int = 500,
                      mlip_path: Optional[str] = None, ):
    cutoff = 10
    computes = ['lj', 'reduction_en']
    if energy_function != 'lj':
        computes.append(energy_function)

    if energy_function == 'uma':
        predictor = init_uma_crystal_predictor(mlip_path, device=device)
    elif energy_function == 'mace':
        predictor = load_mace_model(mlip_path, device=device, dtype=torch.float32)
    else:
        predictor = None

    cursor = 0
    pbar = tqdm(total=len(dataset), unit="reparameterized samples")
    feat_dataset = []
    params = torch.zeros((len(dataset), dataset[0].full_cell_parameters().shape[-1]), dtype=torch.float32)
    while cursor < len(dataset):
        try:
            crystal_batch = collate_data_list(
                [dataset[ind] for ind in range(cursor, min(len(dataset), cursor + batch_size))])
            crystal_batch = crystal_batch.to(device)

            crystal_batch.latent_to_cell_params(crystal_batch.latent_params())
            params[cursor:min(len(dataset),
                              cursor + batch_size)] = crystal_batch.full_cell_parameters()  # record canonicalized cell params
            crystal_batch.analyze(computes,
                                  cutoff=cutoff,
                                  supercell_size=10,
                                  std_orientation=True,
                                  assign_outputs=True,
                                  predictor=predictor
                                  )
            feat_dataset.extend(crystal_batch.cpu().detach().batch_to_list())
            cursor += batch_size
            pbar.update(min(batch_size, len(dataset) - cursor))  # safe final update
            batch_size += 1

        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                batch_size = max(int(batch_size * 0.6), 1)
                print(f"OOM error: dropping batch size to {batch_size}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sleep(0.1)
            else:
                raise e

    for ind, elem in enumerate(feat_dataset):
        elem.set_cell_parameters(params[None, ind])

    return feat_dataset


@torch.no_grad()
def embed_dataset(dataset, autoencoder_path=None, device=None, encoder=None, embedding_type='autoencoder',
                  ):
    batch_size = 500
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False
    )
    if encoder is None and embedding_type == 'autoencoder':
        encoder = load_encoder(autoencoder_path).to(device).eval()

    embeddings = []

    for crystal_batch in tqdm(loader):
        crystal_batch = crystal_batch.to(device)
        # for now, make all the embeddings exactly standardized,
        # and we'll generate also in the standardized basis
        crystal_batch.orient_molecule(mode='standardized',
                                      target_handedness=torch.ones_like(crystal_batch.radius)[:, None],
                                      )
        if embedding_type == 'autoencoder':
            # NOTE - autoencoder trained on global centroids, but recenter molecules in orient_molecule works on heavy atoms now
            crystal_batch.recenter_molecules(center_on_heavy_atoms=False)
            embeddings.append(encoder.encode(crystal_batch).clone().cpu())
        elif embedding_type == 'principal_axes':
            v_embedding_i, s_embedding_i, _ = batch_molecule_principal_axes_torch(
                crystal_batch.pos,
                crystal_batch.batch,
                crystal_batch.num_graphs,
                crystal_batch.num_atoms,
                heavy_atoms_only=True,
                atom_types=crystal_batch.z
            )
            embeddings.append((v_embedding_i * s_embedding_i[:, :, None]).cpu())

        del crystal_batch

    embeddings = torch.cat(embeddings, dim=0)
    for ind, elem in enumerate(dataset):
        elem.embedding = embeddings[None, ind]

    return dataset




def sample_crystal_prior(crystal_batch, std):
    assert False, "This method needs to be rewritten as the latent prior is no longer std normal"
    # rands = torch.randn((crystal_batch.num_graphs, 12), device=crystal_batch.device) * std
    #
    # # enforce the random prior is in the positive niggli plane
    # if not hasattr(crystal_batch, 'latent_transform'):
    #     crystal_batch.init_latent_transform()
    # temp_params = crystal_batch.latent_transform.inverse(rands,
    #                                                      crystal_batch.sg_ind,
    #                                                      crystal_batch.radius)
    # cell_lengths = temp_params[:, :3]
    # cell_angles = temp_params[:, 3:6]
    #
    # # rescale cell lengths for a good packing coeff
    # target_packing_coeff = (torch.randn(crystal_batch.num_graphs, device=crystal_batch.device) * 0.075 + 0.65).clip(
    #     min=0.55, max=0.95)
    # vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
    # cp1 = crystal_batch.mol_volume * crystal_batch.sym_mult / vol1
    # correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
    # cell_lengths *= correction_ratio[:, None]
    #
    # # enforce positive side of niggli plane
    # cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')
    # temp_params[:, 3:6] = cell_angles
    #
    # prior_samples = crystal_batch.latent_transform.forward(temp_params,
    #                                                        crystal_batch.sg_ind,
    #                                                        crystal_batch.radius
    #                                                        ).clip(min=-6, max=6)
    #
    # return prior_samples


@torch.no_grad()
def update_ema(model, ema_model, decay=0.9999):
    """
    Update ema_model parameters towards model parameters.

    Args:
        model:      nn.Module, the training model
        ema_model:  nn.Module, the EMA copy
        decay:      float, EMA decay (close to 1.0 gives long memory)
    """
    if decay is not None:
        if decay > 0:
            msd = model.state_dict()
            emsd = ema_model.state_dict()
            for k in msd.keys():
                if msd[k].dtype.is_floating_point:
                    emsd[k].mul_(decay).add_(msd[k], alpha=1 - decay)
                else:
                    emsd[k] = msd[k]  # copy over non-float buffers (e.g. ints, bools)
    else:
        msd = model.state_dict()
        emsd = ema_model.state_dict()
        for k in msd.keys():  # simply overwrite state dict to EMA model
            emsd[k] = msd[k]


def manual_batch_to_data_list(batch):
    ptr = batch.ptr
    num_graphs = batch.num_graphs

    # Pre-split all tensor attributes into lists of [num_graphs] length
    attr_splits = {}
    for key in batch.keys():
        if key not in ['batch', 'ptr', 'edges_dict',
                       'niggli_energy',
                       'reduction_en',
                       'core_energy',
                       'density_energy',
                       'lj_energy',
                       'bounding_energy',
                       'asym_unit_dict',
                       'latent_transform', ]:
            value = batch[key]
            if torch.is_tensor(value) and value.size(0) == ptr[-1]:
                # node-level attributes
                attr_splits[key] = torch.split(value, torch.diff(ptr).tolist())
            elif torch.is_tensor(value) and value.size(0) == num_graphs:
                # graph-level attributes
                attr_splits[key] = value.unsqueeze(1)
            elif len(value) == num_graphs:
                # graph-level, list attrubutes
                attr_splits[key] = value

    data_list = []
    for ind in range(num_graphs):
        data = MolCrystalData()

        # here assign attributes to object
        for key, splits in attr_splits.items():
            setattr(data, key, splits[ind])

        data_list.append(data)

    return data_list


def iter_forever(loader):
    while True:
        for batch in loader:
            yield batch


def is_cuda_oom(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    s = str(e).lower()
    return (
            ("cuda" in s and "memory" in s)
            or ("cublas" in s and "alloc" in s)
            or ("cusolver" in s and "alloc" in s)
            or ("out of memory" in s)
            or ("nonzero is not supported for tensors with more than int_max elements" in s)
    )


def get_annealing_factor(start_value, stop_value, total_time, step_iters):
    assert stop_value > 0, "Setting final value as zero breaks this module"
    return (stop_value / start_value) ** (1 / (total_time / step_iters))


@torch.no_grad()
def substitute_prior(noised_fraction, log_noise_range,
                     crystal_batch, energy_function,
                     samples, ):
    # noise buffer samples with gaussian magnitude steps
    rand_dir = torch.randn_like(samples)
    rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
    # rand_magnitude = torch.randn(len(samples), device=samples.device).abs() * noise_level
    u = torch.rand(len(samples))
    rand_magnitude = 10 ** (log_noise_range[0] + (log_noise_range[1] - log_noise_range[0]) * u)
    noised_samples = (samples + rand_dir * rand_magnitude[:, None]).clip(min=-1, max=1)

    if noised_fraction < 1:
        new_samples = samples.clone()
        num_to_replace = max(1, int(len(samples) * noised_fraction))
        inds_to_replace = np.random.choice(len(samples), num_to_replace, replace=False)
        new_samples[inds_to_replace] = noised_samples[inds_to_replace]
    else:
        new_samples = noised_samples

    # have to update the rewards if we are using any loss functions that require them
    crystal_batch, log_T_tensor, condition, condition_id = energy_function.condition_samples(
        crystal_batch,
        sg_inds=crystal_batch.sg_ind,
        z_primes=crystal_batch.z_prime)

    if log_T_tensor is not None:
        log_temperature = log_T_tensor
    else:
        log_temperature = None

    with torch.no_grad():
        crystal_batch.orient_molecule(mode='std')
        new_rewards = energy_function.log_reward(new_samples.to(energy_function.device),
                                                 crystal_batch.to(energy_function.device),
                                                 log_temperature.to(energy_function.device),
                                                 False).to(samples.device)

    return new_rewards, new_samples


@torch.no_grad()
def mc_relax_buffer(buffer, energy_function, turnover_log_sigma: float,
                    max_steps: int = 500, conv_eps=1e-2, conv_hist=10):
    noised_fraction = 1.0
    log_noise_range = [turnover_log_sigma - 1, turnover_log_sigma]

    samples, rewards, crystal_batch, condition = buffer.sample(
        override_batch=len(buffer),
        randomize_orientations=False,
        override_sampler=None,
        override_sample_inds=np.arange(len(buffer)),
    )
    sample_record = torch.zeros((max_steps, samples.shape[0], samples.shape[1]))
    reward_record = torch.zeros((max_steps, len(samples)))
    state_reward_record = torch.zeros((max_steps, len(samples)))

    state_rewards = rewards.clone()
    state_samples = samples.clone()
    for step_ind in tqdm(range(max_steps)):
        noised_rewards, noised_samples = substitute_prior(
            noised_fraction, log_noise_range, crystal_batch.clone(),
            energy_function, state_samples)

        sample_record[step_ind] = noised_samples
        reward_record[step_ind] = noised_rewards

        rep_inds = noised_rewards >= state_rewards
        state_samples[rep_inds] = noised_samples[rep_inds]
        state_rewards[rep_inds] = noised_rewards[rep_inds]

        state_reward_record[step_ind] = state_rewards

        if step_ind > conv_hist:
            window = state_reward_record[step_ind - conv_hist:step_ind]
            delta = window[-1] - window[0]
            converged = (delta.abs() < conv_eps).all()
            if converged:
                break

    deduped_samples, deduped_rewards = dedupe_mc_outputs(sample_record[:step_ind],
                                                         reward_record[:step_ind],
                                                         d_cut=10 ** turnover_log_sigma
                                                         )

    return state_samples, state_rewards, deduped_samples, deduped_rewards


def dedupe_mc_outputs(sample_record, reward_record, d_cut: float):
    T, N, D = sample_record.shape

    kept_samples = []
    kept_rewards = []

    for i in range(N):
        traj = sample_record[:, i, :]  # [T, D]
        rewards = reward_record[:, i]  # [T]

        dmat = torch.cdist(traj, traj)  # [T, T]

        keep = torch.ones(T, dtype=torch.bool, device=traj.device)

        for t in range(T):
            if not keep[t]:
                continue
            # suppress all *future* points within d_cut
            close = (dmat[t, t + 1:] < d_cut)
            keep[t + 1:][close] = False

        kept_samples.append(traj[keep])
        kept_rewards.append(rewards[keep])

    return torch.cat(kept_samples), torch.cat(kept_rewards)


@torch.no_grad()
def calibrate_prior_noise(buffer, energy_function,
                          log_min=-3, log_max=-0.5,
                          low_cut=0.05, high_cut=10.0  # rewards are in units of kT already
                          ):
    samples, rewards, crystal_batch, condition = buffer.sample(
        override_batch=len(buffer),
        randomize_orientations=False,
        override_sampler=None,
        override_sample_inds=np.arange(len(buffer)),
    )
    # noise buffer samples with gaussian magnitude steps
    rand_dir = torch.randn_like(samples)
    rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
    rand_magnitude = torch.logspace(log_min, log_max, len(samples))

    noised_samples = (samples + rand_dir * rand_magnitude[:, None]).clip(min=-1, max=1)
    new_samples = noised_samples  # todo confirm right latents / dists

    # have to update the rewards if we are using any loss functions that require them
    crystal_batch, log_T_tensor, condition, condition_id = energy_function.condition_samples(
        crystal_batch,
        sg_inds=crystal_batch.sg_ind,
        z_primes=crystal_batch.z_prime)

    if log_T_tensor is not None:
        log_temperature = log_T_tensor
    else:
        log_temperature = None

    with torch.no_grad():
        crystal_batch.orient_molecule(mode='std')
        new_rewards = energy_function.log_reward(new_samples.to(energy_function.device),
                                                 crystal_batch.to(energy_function.device),
                                                 log_temperature.to(energy_function.device),
                                                 False).to(samples.device)

    'calibration'
    x = torch.nan_to_num(rand_magnitude.log10())
    y = torch.nan_to_num((rewards - new_rewards).abs().log10())
    m, b = np.polyfit(x, y, 1)
    # bins = np.linspace(x.min(), x.max(), 40)
    # medians = np.array([y[(bins[ind] <= x) * (x < bins[ind + 1])].median() for ind in range(len(bins) - 1)])

    x_min = (np.log10(low_cut) - b) / m
    x_max = (np.log10(high_cut) - b) / m

    eps = 1e-6
    delta = (rewards - new_rewards) / (rewards.abs() + eps)
    delta = delta.cpu().numpy()

    x = rand_magnitude.log10().cpu().numpy()

    m = np.median(delta)
    s = median_abs_deviation(delta) + eps
    c = 5.0

    delta_h = np.where(
        np.abs(delta - m) <= c * s,
        delta,
        m + c * s * np.sign(delta - m)
    )

    # evaluation grid (independent of N)
    x_grid = np.linspace(x.min(), x.max(), 300)

    bandwidth = 0.15  # in log10(σ) units; this is the only tuning knob

    # kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)

    # weighted regression via Nadaraya–Watson
    y_smooth = np.zeros_like(x_grid)

    for i, xi in enumerate(x_grid):
        w = np.exp(-0.5 * ((x - xi) / bandwidth) ** 2)
        y_smooth[i] = np.sum(w * delta_h) / np.sum(w)
    dy = np.gradient(y_smooth, x_grid)
    d2y = np.gradient(dy, x_grid)

    turnover_idx = np.argmax(d2y)
    turnover_log_sigma = x_grid[turnover_idx]
    turnover_sigma = 10 ** turnover_log_sigma
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=x_grid, y=y_smooth,
    #                          mode="lines", name="robust mean"))
    # fig.add_trace(go.Scatter(x=[turnover_log_sigma],
    #                          y=[y_smooth[turnover_idx]],
    #                          marker_color="red", marker_size=10,
    #                          name="turnover"))
    # fig.show()

    del crystal_batch

    return new_rewards.detach().clone(), new_samples.detach().clone(), [x_min, x_max], turnover_log_sigma


@torch.no_grad()
def new_calibrate_prior_noise(sample_batch, energy_function,
                              en_scaling_factor, kT,
                              log_min=-3, log_max=-0.5,
                              low_cut=0.05, high_cut=10.0,  # in units of kT
                              predictor=None,
                              device='cuda'
                              ):
    energy = sample_batch[energy_function]

    latents = sample_batch.latent_params()
    rewards = -en_scaling_factor * energy / kT

    noised_batch = sample_batch.clone()
    noised_batch.log_noise_latent_parameters(log_min, log_max)
    rand_magnitude = simple_latent_distance(noised_batch.latent_params(), latents)
    with torch.no_grad():  # reprocess with corrected latents
        noised_batch = adaptive_batched_analysis(
            noised_batch,
            analyses=[energy_function], state={},
            initial_batch_size=100, predictor=predictor,
            device=device,
        )
    new_energy = noised_batch[energy_function]
    new_rewards = -en_scaling_factor * new_energy / kT

    'calibration'
    x = torch.nan_to_num(rand_magnitude.log10()).cpu()
    y = torch.nan_to_num(((rewards - new_rewards).abs() + 1e-4).log10()).cpu()
    m, b = np.polyfit(x, y, 1)

    x_min = (np.log10(low_cut) - b) / m
    x_max = (np.log10(high_cut) - b) / m

    return [x_min, x_max]


#
# def noise_buffer_ramped(max_noise_level, noised_fraction, buffer, energy_function, reward_range,
#                         noise_step,
#                         sample_inds: Optional[torch.Tensor] = None):
#     # sample full buffer
#     samples, rewards, crystal_batch, condition = buffer.sample(
#         override_batch=len(buffer),
#         randomize_orientations=False,
#         override_sampler=None,
#         override_sample_inds=sample_inds,
#     )
#
#     sample_record = []
#     reward_record = []
#     noise_level = 0
#     while True:
#         noise_level += noise_step
#         if noise_level > 0:
#             condition, noised_rewards, noised_samples, crystal_batch = substitute_prior(
#                 noised_fraction, noise_level, crystal_batch.clone(),
#                 energy_function, rewards, samples, buffer)
#         else:
#             noised_rewards = rewards.clone()
#             noised_samples = samples.clone()
#
#         reward_record.extend(noised_rewards.detach().cpu())
#         sample_record.extend(noised_samples)
#         rewards_within_range = noised_rewards >= (rewards - reward_range)
#
#         if rewards_within_range.float().mean() < 0.5:
#             break
#         if noise_level >= max_noise_level:
#             break
#
#     print(f"final noise level {noise_level}")
#     print(f"tot num samples {len(reward_record)}")
#     print(f"batch size {crystal_batch.num_graphs}")
#     return reward_record, sample_record, noise_level


def noise_buffer(log_noise_range, buffer, energy_function,
                 sample_inds: Optional = None):
    noised_fraction = 1
    samples, rewards, crystal_batch, condition = buffer.sample(
        override_batch=len(buffer),
        randomize_orientations=False,
        override_sampler=None,
        override_sample_inds=sample_inds,
    )

    noised_rewards, noised_samples = substitute_prior(
        noised_fraction, log_noise_range, crystal_batch.clone(),
        energy_function, samples)

    del crystal_batch

    return noised_rewards.detach().clone(), noised_samples.detach().clone()


def stdz(x, eps: float = 1e-6):
    return (x - x.mean()) / (x.std() + eps)


@torch.no_grad()
def batched_crystal_analysis(samples, device, computes: list[str] = ['lj'],
                             do_uma: bool = False, init_batch_size: int = 500,
                             uma_path: Optional[str] = None, cutoff: float = 10,
                             max_batch_size: int = 10000, grow_batch_size: bool = True):
    def analyze_batch(samples, cursor, batch_size, sample_outputs, uma_predictor=None):
        crystal_batch = collate_data_list(
            [samples[ind] for ind in range(cursor, min(len(samples), cursor + batch_size))])
        crystal_batch = crystal_batch.to(device)
        crystal_batch.box_analysis()
        out = crystal_batch.analyze(computes,
                                    cutoff=cutoff,
                                    supercell_size=5,
                                    std_orientation=True,
                                    assign_outputs=True,
                                    )

        out = {key: val.cpu().detach() for key, val in out.items()}
        if do_uma == 'uma':
            cry_en = crystal_batch.compute_crystal_uma(
                predictor=uma_predictor,
                std_orientation=True).cpu().detach() * 96.485  # output in kJ/mol (of unit cells)
            gas_en = crystal_batch.compute_lattice_gas_phase_uma(
                predictor=uma_predictor, std_orientation=True).cpu().detach() * 96.485
            out.update({'uma_gas_pot': gas_en,
                        'uma_pot': cry_en,
                        'uma': cry_en / (
                                crystal_batch.sym_mult.cpu().detach() * crystal_batch.z_prime.cpu().detach()) - gas_en})  # lattice energy
            for key in ['uma_gas_pot', 'uma_pot', 'uma']:
                crystal_batch.add_graph_attr(out[key], key)

        sample_outputs.extend(crystal_batch.cpu().detach().batch_to_list())
        return sample_outputs

    if do_uma:
        assert uma_path is not None
        uma_predictor = init_uma_crystal_predictor(uma_path, device=device)
    else:
        uma_predictor = None

    num_samples = len(samples)
    batch_size = int(1 * init_batch_size)
    cursor = 0
    already_oomed = False
    pbar = tqdm(total=len(samples), unit="reparameterized samples")
    sample_outputs = []
    with tqdm(total=len(samples)) as pbar:
        while cursor < len(samples):
            try:
                sample_outputs = analyze_batch(samples, cursor, batch_size, sample_outputs, uma_predictor=uma_predictor)
                cursor += batch_size
                if ((batch_size <= max_batch_size) and (
                        batch_size < num_samples) and not already_oomed) and grow_batch_size:
                    batch_size += max(int(batch_size * 0.01), 1)
                pbar.update(min(batch_size, len(samples) - cursor))  # safe final update

            except (RuntimeError, ValueError) as e:
                if is_cuda_oom(e):
                    if batch_size == 1 and already_oomed:
                        assert False, "Cascading OOM failure in molecule energy evaluation"
                    batch_size = max(int(init_batch_size * 0.6), 1)
                    print(f"OOM error: dropping batch size to {init_batch_size}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    already_oomed = True
                    sleep(0.1)
                else:
                    raise e

    return sample_outputs


def thin_large_dmat(latents, d_cut, energy, B: int = 100000):
    # todo replace this with a very beautiful method
    sort_inds = torch.argsort(energy, descending=False)
    N = len(latents)
    keep = torch.zeros(N, dtype=torch.bool, device=latents.device)

    for i in tqdm(sort_inds):
        kept_inds = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if len(kept_inds) > 0:
            # chunked distance check
            found = False
            for j in range(0, len(kept_inds), B):
                chunk = kept_inds[j:j + B]
                if (compute_latent_distance(latents[None, i], latents[chunk]) < d_cut).any():
                    found = True
                    break
            if found:
                continue
        keep[i] = True

    return keep


def thin_large_dmat_block(latents, energy, d_cut, target_entries=5_000_000):
    # Fixed argsort: simplest signature to avoid TypeError
    sort_inds = torch.argsort(energy)
    latents = latents[sort_inds]
    N, K = latents.shape
    keep_mask = torch.zeros(N, dtype=torch.bool, device=latents.device)
    kept_indices = []
    i = 0
    pbar = tqdm(total=N)
    while i < N:
        num_kept = len(kept_indices)
        # Adaptive block size
        if num_kept > 0:
            B = max(1, min(2048, target_entries // num_kept))
        else:
            B = 1024
        end = min(i + B, N)
        block_latents = latents[i:end]
        b_size = block_latents.shape[0]
        # 1. Batch Check: Block vs. All Previously Kept
        if num_kept > 0:
            kept_latents = latents[kept_indices]
            # Expansion logic for [n, k] distance function
            lat1_exp = block_latents[:, None, :].expand(b_size, num_kept, K).reshape(-1, K)
            lat2_exp = kept_latents[None, :, :].expand(b_size, num_kept, K).reshape(-1, K)
            d_to_past = compute_latent_distance(lat1_exp, lat2_exp).view(b_size, num_kept)
            is_far_from_past = ~(d_to_past < d_cut).any(dim=1)
        else:
            is_far_from_past = torch.ones(b_size, dtype=torch.bool, device=latents.device)
        # 2. Sequential Check within the block
        candidate_sub_indices = is_far_from_past.nonzero(as_tuple=True)[0]
        for idx in candidate_sub_indices:
            curr_idx = i + idx.item()
            curr_latent = latents[curr_idx:curr_idx + 1]  # [1, K]
            # Re-check against latents added from THIS SAME BLOCK
            newly_added_indices = [j for j in kept_indices if j >= i]
            if newly_added_indices:
                recent_tensor = latents[newly_added_indices]  # [R, K]
                # Your distance function accepts [1, K] and [R, K]
                if (compute_latent_distance(curr_latent, recent_tensor) < d_cut).any():
                    continue
            kept_indices.append(curr_idx)
            keep_mask[curr_idx] = True
        pbar.update(end - i)
        i = end
    pbar.close()
    final_keep = torch.zeros(N, dtype=torch.bool, device=latents.device)
    final_keep[sort_inds] = keep_mask
    return final_keep


def atomic_save(state_dict, path):
    try:
        tmp_path = path + ".tmp"
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        print("Save failed to", path)


def get_discretizer(int_cfg):
    """Uniform steps over integrator.T, always.

    The variable-length / alternative-discretizer branches (random,
    low_discrepancy, equidistant; 'sampled' trajectory lengths between
    min_traj_length and max_traj_length) were deleted with their five config
    keys. Nothing on any live route selected them, and a non-uniform dt is not
    a free choice here: the policy learns a drift and variance PER STEP at one
    dt, and P_B's reference bridge is written in accumulated-variance time, so
    changing the step schedule changes the SDE being integrated rather than
    just its quadrature. The alternatives are in git history if a variable-dt
    experiment is ever wanted; they should come back with the density
    correction that would make them valid, not as a config switch."""
    return lambda bsz: uniform_discretizer(bsz, int_cfg.T)


def drain_elapsed_times(times):
    """
    Turn completed <name>_start / <name>_end pairs into <name>_time, and CONSUME
    them: the start key is deleted so the same interval is not re-reported until
    a new one is timed.

    MUTATES `times`. Without the drain, any pair that is written once and never
    rewritten is re-emitted on every single report -- 'initialization_time' was
    logging the same constant 204 times in a run, and a flat line that long
    reads like a measurement rather than a leftover.

    Only the START key is dropped. The ends are left in place because the eval
    path chains off one of them (inter_eval_start = times['eval_step_end']), and
    an end with no start can't re-form a pair on its own anyway.
    """
    elapsed_times = {}
    for start_key in [k for k in times if 'start' in k]:
        name = start_key.split('_start')[0]
        end_key = name + '_end'
        if end_key in times:
            elapsed_times[name + '_time'] = times[end_key] - times[start_key]
            del times[start_key]

    return elapsed_times


class MetricTracker:
    """Step-aware EMA over arbitrary scalars, keyed by (direction, name).
    Also tracks running min/max of the EMA per key."""

    def __init__(self, period: float = 25.0):
        self.period = period
        self.values = {}  # (direction, name) -> float
        self.best = {}  # (direction, name) -> [min, max]
        self.last_it = {}  # direction -> step
        self.changed_keys = set()

    def update(self, direction, scalars: dict, step: int):
        dt = max(step - self.last_it.get(direction, step), 1)
        alpha = 1.0 - np.exp(-dt / self.period)

        for name, v in scalars.items():
            v = v.item() if torch.is_tensor(v) else float(v)
            if not np.isfinite(v):
                continue
            key = (direction, name)
            prev = self.values.get(key)
            nv = v if prev is None else (1 - alpha) * prev + alpha * v

            self.values[key] = nv
            self.changed_keys.add(key)  # mark as changed

            b = self.best.get(key)
            if b is None:
                self.best[key] = [nv, nv]
            else:
                if nv < b[0]: b[0] = nv
                if nv > b[1]: b[1] = nv

        self.last_it[direction] = step

    def get(self, direction, name, default=None):
        return self.values.get((direction, name), default)

    def get_best(self, direction, name, mode='max', default=None):
        b = self.best.get((direction, name))
        if b is None:
            return default
        return b[1] if mode == 'max' else b[0]

    def snapshot(self, changed_only=False):
        if changed_only:
            result = {f'{d}/{n}': v for (d, n), v in self.values.items()
                      if v is not None and (d, n) in self.changed_keys}
            self.changed_keys.clear()  # reset now that this snapshot has consumed them
            return result
        return {f'{d}/{n}': v for (d, n), v in self.values.items() if v is not None}

    def state_dict(self):
        nested, best = {}, {}
        for (d, n), v in self.values.items():
            nested.setdefault(d, {})[n] = v
        for (d, n), mm in self.best.items():
            best.setdefault(d, {})[n] = list(mm)
        return {'period': self.period, 'last_it': dict(self.last_it),
                'values': nested, 'best': best}

    def load_state_dict(self, sd):
        self.period = sd.get('period', self.period)
        self.last_it = dict(sd.get('last_it', {}))
        self.values = {(d, n): v for d, kv in sd.get('values', {}).items() for n, v in kv.items()}
        self.best = {(d, n): list(mm) for d, kv in sd.get('best', {}).items() for n, mm in kv.items()}

    def rebase(self, step):
        """After restoring values from a checkpoint, reset the EMA clock to `step`
        so the first post-reload update uses a small dt, not the gap since the save."""
        for d in self.last_it:
            self.last_it[d] = step


def quick_tb_stats(log_pf, log_pb, log_Z, log_r, reward_floor=None, ramp_width=None,
                   clip_beta=None, condition_id=None, worst_quantile=0.5):
    """
    reward_floor/ramp_width gate under_coverage by a per-sample reward ramp:

        w_raw = clamp((log_r - reward_floor) / ramp_width, 0, 1)

    reward_floor is per-sample (M_c - ramp_floor, from the sample's own
    condition's anchor max; see Modeller._reward_ramp_kwargs for the
    depth-space definition) and ramp_width is the width of the linear
    transition band sitting directly above the floor: weight 0 at or below
    reward_floor, saturating to 1 at reward_floor + ramp_width and staying 1
    up through M_c. Weights are self-normalized, so a heavy low-reward tail
    can't inflate under_coverage while real (even if modest) modes still
    register. ramp_width must be > 0 (asserted -- a non-positive width is the
    silent-inversion bug this parameterization replaced). If no sample in the
    batch clears the floor, under_coverage is nan ("no qualifying samples")
    rather than a fake 0; MetricTracker skips non-finite updates so EMAs hold.
    Pass both as None to fall back to the old uniform-RMS behavior (always
    also reported as 'under_coverage_uniform').

    clip_beta (the direction's Huber beta from *_loss_coeffs) adds
    'tb_resid_clipped': the signed batch mean of the residual clamped to
    [-beta, beta], which IS dL/dZ (up to the constant beta scale) when Z
    trains via Huber TB. It reads ~0 exactly at the loss's own fixed point
    (the self-consistent beta-Winsorized mean of log w) and, unlike
    'tb_resid' (the Jensen delta -- offset by the clipped-off tail mass) or
    'tb_err' (RMS, floored at std(log w)), is bounded by beta, so fat or
    skewed tails can't inflate it and a lagging Z shows as a persistent
    sign. None (e.g. legacy callers) skips the metric. The Huber beta must
    be held FIXED across the whole protocol for this to mean one thing.

    THE CONTROL-METRIC FAMILY (all in nats, all EMA-safe per-sample means --
    never ratios; the conditional r2 family this replaced could not be EMA'd
    and was unreachable at ~2-3 samples/condition):

      'tb_err'          pooled batch RMS residual: the global quality of fit.
                        Uncentered second moment about zero, so unlike an r2
                        it has no group-mean denominator to collapse or
                        Bessel-bias at small groups.
      'cond_tb_err'     the TYPICAL condition: unweighted mean over conditions
                        of each one's own RMS residual. NOT the same number as
                        tb_err -- sqrt is concave, so this sits at or below the
                        pooled value (equality only when every condition is
                        fit equally well). The gap between them is itself a
                        reading: pooled tb_err squares before averaging and so
                        is dominated by the worst conditions, while this is
                        what a randomly chosen condition looks like.
      'tb_err_worst'    the worst-case CONDITION: the `worst_quantile` upper
                        tail across per-condition RMS residuals. This is the
                        control metric (exit gates, calibration guards).
      'z_grad_worst'    same construction on the CLIPPED SIGNED per-condition
                        mean -- the per-condition dL/dZ ruler, level-only
                        (spread averages out before the abs). Pairs with the
                        pooled 'tb_resid_clipped'.

    Whether a residual is level or spread is exactly the actuator question:
    E[r^2] = mean(r)^2 + Var(r), so z_grad_worst is the part Z training can
    fix and the excess of tb_err_worst over it is the part only policy
    training can. worst_quantile is the fraction of conditions allowed to sit
    beyond the bar (0.5 = median condition, 0.05 = 95% must clear).

    With condition_id=None (unconditional runs, or any caller without a
    condition axis) the whole batch is ONE group, so cond_tb_err/tb_err_worst
    degrade exactly to tb_err and z_grad_worst to |tb_resid_clipped| -- the
    same metric names carry the same meaning on conditional and unconditional
    runs, so protocol rules need no per-problem rewriting.

    'logw_std_within' is the pooled WITHIN-condition std of log w: the
    batch-wide 'logw_std' with the between-condition component removed. That
    between-condition part -- the spread of the per-condition Jensen means /
    log Z(c) -- dominates the plain batch-wide std whenever the condition set
    is large and dissimilar (hundreds of nats over a broad library; ~365 nats
    between just two conditions in the 2-cond toy), and it is NOT what
    condition-grouped VarGrad reduces. So 'logw_std' is a misleading
    convergence signal at scale (it barely moves as VarGrad works, and RISES
    when conditions are made more dissimilar); this is the quantity VarGrad
    actually optimizes. Unlike the control family above it is conditional-ONLY
    and is the one metric here that can be absent: with condition_id=None the
    single group spans the batch and it would merely duplicate 'logw_std', and
    only multi-member groups carry signal -- a singleton's deviation from its
    own mean is trivially 0, so singletons are masked out of numerator and
    denominator rather than diluting the estimate with zeros. When no group has
    >= 2 members (e.g. a forward batch drawn with repeats == 1 over a large
    library) the key is OMITTED rather than reported as nan or 0, so neither
    MetricTracker nor a raw wandb.log ever sees a spurious value.

    'relative_under' is the under_coverage computation re-centered on THIS
    batch's own empirical normalizer (z_jensen = mean log w) instead of
    log_Z. The collective level gap (learned Z vs buffer-implied Z) is not
    something backward training can act on -- E_mu[log P_F] is capped at
    -H(mu) by normalization, so the whole cloud cannot translate -- and it
    makes the Z-anchored under_coverage read "everything is under-covered"
    whenever Z lags, starving the controller's other modes. Re-centered,
    'under-covered' means under-covered relative to the rest of the batch:
    the spread component that IS the policy's to fix. The phase-3 controller
    keys backward allocation on this; the Z-anchored under_coverage stays
    reported as the absolute-merge gauge (its gap to relative_under, like
    jensen_z vs log_Z_learned, is the has-forward-caught-up signal).

    condition_id, when given, re-centers relative_under's z_jensen PER
    CONDITION (each sample against its own condition's group mean of log_w,
    same scatter-mean pattern as logw_std_within / cond_tb_err)
    instead of the single batch-wide pooled z_jensen. A pooled z_jensen mixes
    conditions with different true log Z into one mean, which is not any
    condition's own normalizer -- cazwlyy1: bwd's per-condition log_Z_learned
    had converged to the per-condition level while the pooled 'jensen_z'
    metric sat far off from either, and relative_under (built on the pooled
    value) read as a large spurious violation. condition_id=None reproduces
    the old pooled behavior exactly (unconditional callers unaffected). The
    reported 'jensen_z'/'z_gap' metrics stay pooled either way -- only
    relative_under's own centering changes.

    'relative_under_wcen' is relative_under with that centre computed under the
    SAME reward-ramp weights the RMS uses, instead of unweighted. As written,
    relative_under scores only ramp-qualifying samples but centres on all of
    them, so the two halves run over different populations and the number
    inherits a floor from BATCH COMPOSITION: a bwd batch that is a fraction f
    of low-reward prior-buffer draws sitting Delta below the anchor-sourced
    material puts the centre at mu_scored - f*Delta, and every scored sample
    picks up that offset before the one-sided clamp (~f*Delta of apparent
    under-coverage with a perfectly fit scored population). f is a buffer knob
    -- anchor top-up rate, churn, weighted_bwd_beta, purge -- so the metric is
    not comparable across a run whose mix drifts, and the phase-3 controller
    that keys backward allocation on it also moves the mix that sets it.
    Weighting the centre removes f from both sides. Keep BOTH: their gap is the
    composition reading (large gap == the batch is mostly material the ramp
    scores at ~0), and 'ramp_ess_frac' says how many samples the weighted
    centre is actually averaging. Identical to relative_under by construction
    when the ramp is unconfigured (uniform weights).
    """
    x = (log_pb + log_r).detach()
    y = (log_pf + log_Z).detach()
    resid = y - x  # TB residual
    xc, yc = x - x.mean(), y - y.mean()
    slope = (xc * yc).sum() / (xc * xc).sum().clamp_min(1e-8)
    intercept = y.mean() - slope * x.mean()
    # r2 = 1 - ((yc - slope * xc) ** 2).sum() / (yc * yc).sum().clamp_min(1e-8)
    r2 = 1 - (resid ** 2).sum() / (yc * yc).sum().clamp_min(1e-8)  # CCC style - against the diagonal

    log_w = (log_r + log_pb - log_pf).detach()  # per-traj log importance weight
    z_jensen = log_w.mean()  # Jensen LB:  E[log w] <= log E[w] (pooled, batch-wide)
    z_emp = torch.logsumexp(log_w, dim=0) - np.log(log_w.shape[0])  # logmeanexp estimate
    z_learned = log_Z.detach().mean()

    if condition_id is not None:
        cid = condition_id.detach().flatten().to(log_w.device)
        uniq, inverse = torch.unique(cid, return_inverse=True)
        k = uniq.numel()
        counts = torch.zeros(k, device=log_w.device, dtype=log_w.dtype).scatter_add_(
            0, inverse, torch.ones_like(log_w))
        group_sum = torch.zeros(k, device=log_w.device, dtype=log_w.dtype).scatter_add_(
            0, inverse, log_w)
        z_jensen_g = group_sum / counts.clamp(min=1)  # per-condition Jensen mean
    else:
        # ONE group covering the batch: the conditional metrics below then
        # reduce exactly to their pooled counterparts (see docstring)
        inverse = torch.zeros_like(resid, dtype=torch.long)
        counts = torch.full((1,), float(resid.numel()), device=resid.device, dtype=resid.dtype)
        k = 1
        z_jensen_g = z_jensen.reshape(1)  # old pooled behavior
    z_jensen_ref = z_jensen_g[inverse]  # each sample's OWN group's Jensen mean

    # TB residual skew
    resid_c = (resid)  # center it on zero
    skew = (resid_c.pow(3).mean() / resid_c.pow(2).mean().pow(1.5).clamp_min(1e-8))

    # Under-weighted trajectories
    neg = resid.clamp(max=0)  # 0 where resid >= 0, negative elsewhere (RAW resid, not centered)
    under_severity_uniform = neg.pow(2).mean().sqrt().item()  # RMS of negative residuals
    if reward_floor is not None and ramp_width is not None:
        assert ramp_width > 0, f"ramp_width must be positive, got {ramp_width}"
        w_raw = ((log_r.detach() - reward_floor) / ramp_width).clamp(0, 1)
        total = w_raw.sum()
        if total > 0:
            under_severity = ((w_raw / total) * neg.pow(2)).sum().sqrt().item()  # reward-weighted RMS of negatives
        else:
            under_severity = float('nan')  # no sample cleared its condition's floor this batch
    else:
        # the docstring's promised no-ramp fallback (was an UnboundLocalError
        # if this path was ever reached; production always has a ramp once the
        # anchor buffer seeds, so it lay dormant)
        under_severity = under_severity_uniform
    pos = resid.clamp(min=0)  # over_coverage stays uniform: replay must see over-weighted junk
    over_severity = pos.pow(2).mean().sqrt().item()

    # relative_under: same negative-tail RMS, centered on z_jensen_ref (per-
    # condition when condition_id is given, else the pooled batch mean --
    # see docstring) instead of log_Z -- the level gap drops out, leaving the
    # within-batch spread component the policy can actually fix. Same reward-ramp
    # weighting as under_coverage, for the same low-reward-tail hygiene.
    neg_rel = (z_jensen_ref - log_w).clamp(max=0)
    if reward_floor is not None and ramp_width is not None:
        relative_under = (((w_raw / total) * neg_rel.pow(2)).sum().sqrt().item()
                          if total > 0 else float('nan'))
    else:
        relative_under = neg_rel.pow(2).mean().sqrt().item()

    # relative_under_wcen: relative_under with the CENTERING moved onto the same
    # reward-ramp weights the RMS already uses (per group, so the per-condition
    # re-centering above is preserved). 'relative_under' centers on the
    # UNWEIGHTED group mean of log_w while scoring only ramp-qualifying samples,
    # so the two are computed over different populations: a bwd batch mixing
    # low-reward prior-buffer draws (ramp weight ~0, low log_w) with anchor-
    # sourced states (weight 1, high log_w) has its reference dragged down by
    # material that contributes nothing to the numerator. Writing that mixture
    # as a fraction f of junk sitting Delta below the scored population,
    #     z_ref = mu_scored - f*Delta,
    # every scored sample picks up a +f*Delta offset before the one-sided
    # clamp, so relative_under carries a floor of ~f*Delta set purely by batch
    # composition -- a buffer/controller quantity (anchor top-up rate, churn,
    # weighted_bwd_beta) rather than a policy one. Weighting the centre makes
    # the junk drop out of numerator and reference alike, leaving the spread
    # among the samples the ramp says matter.
    #
    # This is NOT a strictly-better restatement: the junk/anchor gap Delta is a
    # real level-free defect (P_F over-weighting junk relative to anchors) that
    # backward training can fix. 'relative_under' keeps that in and so moves
    # with the buffer mix; this one takes it out and so is comparable across a
    # drifting mix. Both are reported -- their GAP is the composition reading.
    #
    # With no ramp configured the weights are uniform and the two are identical
    # by construction, so the pre-anchor warmup is unaffected. Groups where no
    # sample clears the floor fall back to the unweighted group mean (those
    # samples carry weight 0 and contribute nothing regardless).
    if reward_floor is not None and ramp_width is not None:
        if total > 0:
            wsum_g = torch.zeros(k, device=log_w.device, dtype=log_w.dtype).scatter_add_(
                0, inverse, w_raw)
            wdot_g = torch.zeros(k, device=log_w.device, dtype=log_w.dtype).scatter_add_(
                0, inverse, w_raw * log_w)
            z_wcen_g = torch.where(wsum_g > 0, wdot_g / wsum_g.clamp_min(1e-12), z_jensen_g)
            neg_rel_w = (z_wcen_g[inverse] - log_w).clamp(max=0)
            relative_under_wcen = ((w_raw / total) * neg_rel_w.pow(2)).sum().sqrt().item()
        else:
            relative_under_wcen = float('nan')
    else:
        relative_under_wcen = relative_under  # uniform weights => identical centre

    log_ess = 2 * torch.logsumexp(log_w, dim=0) - torch.logsumexp(2 * log_w, dim=0)
    ess_frac = torch.exp(log_ess - np.log(log_w.shape[0]))
    mets = {
        'slope_err': (slope - 1).abs().item(),
        'intercept_err': intercept.abs().item(),
        'scatter_err': resid.std(unbiased=False).item(),
        'r2': r2.item(),
        'tb_resid': resid.mean().item(),
        'tb_err': resid.pow(2).mean().sqrt().item(),
        'jensen_z_err': (log_w - log_Z.detach()).abs().mean().item(),
        'emp_z_err': (z_emp - z_learned).abs().item(),
        'under_coverage': under_severity,
        'under_coverage_uniform': under_severity_uniform,
        'relative_under': relative_under,
        'relative_under_wcen': relative_under_wcen,
        'over_coverage': over_severity,
        'z_gap': (z_emp - z_jensen).item(),
        'resid_p05': resid.detach().quantile(0.05).item(),
        'resid_p95': resid.detach().quantile(0.95).item(),
        'resid_skew': skew.item(),  # sign tells you over- vs under-sampling dominance
        'jensen_z': z_jensen.item(),
        'emp_z': z_emp.item(),
        "logw_std": log_w.std(unbiased=False).item(),
        "ess_frac": ess_frac.item(),
    }

    if reward_floor is not None and ramp_width is not None:
        # Kish ESS of the reward-ramp weights as a fraction of the batch:
        # (sum w)^2 / (n * sum w^2). relative_under_wcen's centre is a weighted
        # mean over exactly this population, so a small value means a noisy
        # centre -- read it before trusting either reward-weighted metric as a
        # controller input. 0.0 when nothing cleared the floor (the same batch
        # that sends under_coverage/relative_under* to nan). Omitted, not 1.0,
        # when the ramp is unconfigured, so 'ramp is off' and 'ramp is wide
        # enough to score everything' stay distinguishable.
        mets['ramp_ess_frac'] = (total.pow(2) / w_raw.pow(2).sum().clamp_min(1e-12)
                                 / w_raw.numel()).item()

    # --- the control-metric family (see docstring): per-condition RMS residual
    # and per-condition clipped signed mean, reduced to a worst-case quantile
    # across conditions. Groups of size 1 are kept: the RMS here is about ZERO,
    # not about a group mean, so a singleton contributes |resid| -- a perfectly
    # valid one-sample estimate of that condition's fit. (The r2 family this
    # replaced had to drop singletons and Bessel-correct the survivors, which is
    # exactly what made it unreachable at ~2-3 samples/condition.)
    q_hi = min(max(1.0 - float(worst_quantile), 0.0), 1.0)  # tb_err: larger is worse
    ss_resid = torch.zeros(k, device=resid.device, dtype=resid.dtype).scatter_add_(
        0, inverse, resid ** 2)
    cond_tb_err = (ss_resid / counts.clamp(min=1)).sqrt()
    # mean and quantile over the SAME population (conditions, unweighted), so
    # 'the typical condition' and 'the worst condition' are directly comparable
    mets['cond_tb_err'] = cond_tb_err.mean().item()
    mets['tb_err_worst'] = torch.quantile(cond_tb_err, q_hi).item()

    if clip_beta is not None:
        clipped = resid.clamp(-clip_beta, clip_beta)
        mets['tb_resid_clipped'] = clipped.mean().item()
        # per-condition dL/dZ: clip, group-mean (spread averages out), THEN abs
        cond_z_grad = torch.zeros(k, device=resid.device, dtype=resid.dtype).scatter_add_(
            0, inverse, clipped) / counts.clamp(min=1)
        mets['z_grad_worst'] = torch.quantile(cond_z_grad.abs(), q_hi).item()

    # within-condition spread of log w (see docstring): each sample centered on
    # its OWN condition's Jensen mean, which z_jensen_ref already is on this
    # branch, so this reuses the grouping above rather than repeating the
    # scatter. Conditional-only and singleton-masked -- key omitted, not nan.
    if condition_id is not None:
        multi = counts[inverse] >= 2
        if bool(multi.any()):
            centered_w = (log_w - z_jensen_ref)[multi]
            mets['logw_std_within'] = centered_w.pow(2).mean().sqrt().item()

    return mets


def per_condition_fraction(indicator, condition_id, bar, worst_quantile=0.5,
                           higher_is_worse=False):
    """
    Per-CONDITION breakdown of a per-sample 0/1 indicator -- the conditional
    counterpart of the pooled batch fractions ('Reasonable Sample Fraction',
    'Nonthermal Fraction'), which are the same number for two very different
    models:

        every condition at 50% good     -> pooled 0.5
        half the conditions at 0% good  -> pooled 0.5

    The second has entirely lost half the library. That is a worse object than
    a uniformly mediocre one, it is the failure a pooled fraction cannot see,
    and it is the one that matters on a conditional run. So: group the
    indicator by condition_id, take each condition's own fraction p_c, and
    reduce THAT distribution.

      'failing_frac'  frac(conditions that fail `bar`) -- the headline.
                      Direction-aware: p_c < bar when higher is better
                      (reasonable samples), p_c > bar when higher is worse
                      (non-thermal samples). Fails the bar == "this condition
                      is broken", so the metric always reads larger-is-worse.
      'worst'         the `worst_quantile` BAD tail of p_c across conditions,
                      same knob and convention as quick_tb_stats' tb_err_worst
                      (worst_quantile = the fraction of conditions allowed to
                      sit beyond the bar).
      'per_condition' the p_c themselves, for the histogram -- the full
                      distribution, of which the two scalars are readings.

    BOTH SCALARS ARE REPORTED because they are inverse readings of one CDF and
    saturate at OPPOSITE ends. Early in a run nearly every condition fails, so
    failing_frac pins at 1.0 while 'worst' still moves; late, 'worst' pins at
    the good end while failing_frac still resolves the last broken conditions.
    Either one alone is censored across half of a run.

    CHECK RESOLUTION FIRST. p_c is a Bernoulli mean over that condition's n_c
    samples in this batch. At n_c == 1 it is 0 or 1, failing_frac degenerates
    to the pooled fraction (no lie, but no added information), and the
    histogram's spikes at 0 and 1 are binomial artifacts rather than structure.
    'n_conditions' is returned so the caller can publish it: samples per
    condition is batch / n_conditions. Conditions are deliberately NOT
    count-weighted -- each contributes its p_c once, which is the only reason
    this differs from the pooled fraction at all.

    Returns None -- the family is OMITTED, never nan or 0 -- when there is no
    condition axis (condition_id None), fewer than 2 conditions, no samples, or
    no bar. With one group failing_frac would be a 0/1 step function and the
    histogram a single spike, so an unconditional run's metric surface is left
    exactly as it was (module_metrics.md 4, the logw_std_within rule).
    """
    if indicator is None or condition_id is None or bar is None:
        return None
    ind = torch.as_tensor(indicator).detach().flatten().float()
    if ind.numel() == 0:
        return None
    cid = torch.as_tensor(condition_id).detach().flatten().long().to(ind.device)
    if cid.numel() != ind.numel():
        # structural: both come off the same pooled batch, so a mismatch is an
        # upstream alignment bug. Raise -- a per-condition metric computed
        # against the wrong conditions is worse than no metric.
        raise ValueError(f"per_condition_fraction: {ind.numel()} indicator rows "
                         f"vs {cid.numel()} condition ids")
    uniq, inverse = torch.unique(cid, return_inverse=True)
    k = uniq.numel()
    if k < 2:
        return None
    counts = torch.zeros(k, device=ind.device, dtype=ind.dtype).scatter_add_(
        0, inverse, torch.ones_like(ind))
    hits = torch.zeros(k, device=ind.device, dtype=ind.dtype).scatter_add_(0, inverse, ind)
    p_c = hits / counts.clamp(min=1)

    bar = float(bar)
    failing = (p_c > bar) if higher_is_worse else (p_c < bar)
    # the bad tail: upper when higher_is_worse (as tb_err_worst reads it),
    # lower otherwise -- 'worst' must always name the same end as 'failing'
    q = 1.0 - float(worst_quantile) if higher_is_worse else float(worst_quantile)
    q = min(max(q, 0.0), 1.0)
    return {
        'failing_frac': failing.float().mean().item(),
        'worst': torch.quantile(p_c, q).item(),
        'spread': _debiased_condition_spread(p_c, counts),
        'per_condition': p_c,
        'n_conditions': int(k),
    }


def _debiased_condition_spread(p_c, counts):
    """
    Between-condition standard deviation of the TRUE per-condition fraction,
    with the binomial sampling noise subtracted off. The one reading in
    per_condition_fraction that survives a change in samples per condition, and
    therefore the only one that may be compared ACROSS streams (eval_fwd vs
    eval_test, whose n_c differ by ~2x on cond_aug11).

    p_hat_c is a Bernoulli mean, so its spread across conditions is inflated by
    noise that shrinks as n_c grows:

        Var(p_hat) = Var(p) + E[p(1-p)/n]

    Both terms are estimable. The sample variance over conditions (the /(k-1)
    convention, which is what makes the identity exact under heterogeneous n_c)
    gives the left side; p_hat(1-p_hat)/(n-1) is unbiased for p(1-p)/n per
    condition, since E[p_hat(1-p_hat)] = p(1-p)(n-1)/n. Subtract and take the
    root:

        every condition at 50%      -> 0.0   (all spread was noise)
        half the library dead       -> 0.5   (at ANY n_c)

    which is exactly the geometry a pooled fraction cannot see, stated in a way
    that a 2000-sample stream and a 10000-sample one can be held against each
    other. Unbiased in both, so their difference is signal; the smaller stream's
    estimate is merely noisier, and noise is symmetric where the raw
    failing-fraction bias is not.

    Clamped at 0 before the root: a negative estimate means the observed spread
    is within binomial noise, i.e. no detectable concentration, which reads as
    0. Singletons carry no information about Var(p) (their correction term is
    undefined) and are dropped from BOTH terms -- the same singleton mask
    logw_std_within uses, and under uniform condition draws the sample count is
    independent of sample quality. None -- key omitted -- when fewer than two
    conditions have 2+ samples.
    """
    multi = counts >= 2
    if int(multi.sum().item()) < 2:
        return None
    p_m, n_m = p_c[multi], counts[multi]
    var_hat = p_m.var(unbiased=True) - (p_m * (1.0 - p_m) / (n_m - 1.0)).mean()
    return var_hat.clamp_min(0.0).sqrt().item()


def online_tb_coverage(log_pf, log_pb, log_Z, log_r, log_w_clamp=10.0):
    """
    Fast per-batch coverage proxy. No reward bins, no max_reward, no min_count.
    delta = log_pf + log_Z - log_pb - log_r        (delta < 0 at a terminal => P_F under-weights it)

    Target-reweighted mean residual: E_pi[delta] estimated by self-normalized IS from
    the on-policy / buffer batch, with weights w ∝ exp(log_r + log_pb - log_pf) = exp(-delta + log_Z).
    This is the un-binned limit of  sum_b pi(b) * delta_bar_b  — the same forward-KL proxy,
    without having to resolve the high-reward bin.
    """
    delta = (log_pf + log_Z - log_pb - log_r).detach()  # (B,)

    # log importance weight toward pi (drop the constant log_Z; self-norm cancels it)
    log_w = (log_r + log_pb - log_pf).detach()
    log_w = log_w - log_w.max()  # stabilize
    log_w = log_w.clamp_min(-log_w_clamp)  # tame the light tail; heavy tail already capped at 0
    w = log_w.exp()
    w_sum = w.sum().clamp_min(1e-12)

    # self-normalized target-weighted mean residual  ~  E_pi[delta]
    wmean = (w * delta).sum() / w_sum
    ess = w_sum.pow(2) / w.pow(2).sum().clamp_min(1e-12)  # Kish effective sample size

    return {
        'resid_wmean': wmean.item(),  # target-weighted: THE coverage gauge (<0 => missing high-r mass)
        'capture_proxy': wmean.clamp_max(0).exp().item(),  # exp(min(.,0)) — Jensen-style captured-fraction proxy
        'ess': ess.item(),  # trust the wmean only when this isn't ~1
        'ess_frac': (ess / delta.numel()).item(),
    }


def binned_tb_residual(log_pf, log_pb, log_Z, log_r,
                       bin_width=10.0, max_reward=None, n_bins=5, min_count=10):
    resid = (log_pf + log_Z - log_pb - log_r).detach()  # δ ; <0 ⇒ missed mass
    r = log_r.detach()
    r_max = r.max() if max_reward is None else \
        torch.as_tensor(max_reward, device=r.device, dtype=r.dtype)

    out = {}
    for i in range(n_bins):
        lo = r_max - (i + 1) * bin_width
        hi = r_max - i * bin_width
        m = (r >= lo) if i == 0 else (r >= lo) & (r < hi)  # top bin catches r == r_max
        n = int(m.sum())
        if n >= min_count:
            mean_d = resid[m].mean()
            out[f'bin{i}_mean_resid'] = mean_d.item()  # signed δ̄_b
            out[f'bin{i}_capture'] = mean_d.clamp_max(0).exp().item()  # Jensen LB on captured mass
        else:
            out[f'bin{i}_mean_resid'] = float('nan')
            out[f'bin{i}_capture'] = float('nan')
        out[f'bin{i}_n'] = n

    lo_tail = r_max - n_bins * bin_width  # coarse catch-all tail
    m = r < lo_tail
    n = int(m.sum())
    out['tail_mean_resid'] = resid[m].mean().item() if n >= min_count else float('nan')
    out['tail_n'] = n
    return out


import torch


def residual_reward_curve(log_pf, log_pb, log_Z, log_r,
                          bin_width=5.0, max_reward=None, min_count=3,
                          lengthscale=None, signal_var=None, n_grid=200):
    """
    Signal: δ̄(r) = E[δ | reward], δ = log_pf + log_Z - log_pb - log_r.
    δ < 0 ⇒ missed mass at that reward level; capture = exp(min(δ̄, 0)).
    Bins on the reward axis (comparable across runs), then a heteroscedastic GP
    on the per-bin means with noise = SE_b^2, so sparse bins get wide bands.
    Read capture_lo (the pessimistic band) for a conservative coverage statement.
    """
    resid = (log_pf + log_Z - log_pb - log_r).detach()  # δ
    r = log_r.detach()
    dev, dt = r.device, r.dtype
    r_max = r.max() if max_reward is None else torch.as_tensor(max_reward, device=dev, dtype=dt)

    # --- bin on reward axis, summarize ---
    idx = ((r_max - r) / bin_width).floor().long().clamp_min(0)
    nb = int(idx.max()) + 1
    ones = torch.ones_like(resid)
    cnt = torch.zeros(nb, device=dev, dtype=dt).scatter_add_(0, idx, ones)
    s1 = torch.zeros(nb, device=dev, dtype=dt).scatter_add_(0, idx, resid)
    s2 = torch.zeros(nb, device=dev, dtype=dt).scatter_add_(0, idx, resid * resid)
    mean = s1 / cnt.clamp_min(1)
    var = (s2 / cnt.clamp_min(1) - mean ** 2).clamp_min(0)
    se = (var / cnt.clamp_min(1)).sqrt()
    ctr = r_max - (torch.arange(nb, device=dev, dtype=dt) + 0.5) * bin_width

    keep = cnt >= min_count
    X, Y, N = ctr[keep], mean[keep], se[keep] ** 2 + 1e-6

    # --- exact GP, RBF kernel, fixed hypers (heuristic; fine for a diagnostic) ---
    ell = (2.0 * bin_width) if lengthscale is None else lengthscale
    sf2 = (Y.var().clamp_min(1.0)) if signal_var is None else signal_var
    rbf = lambda a, b: sf2 * torch.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell ** 2)

    K = rbf(X, X) + torch.diag(N)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(Y[:, None], L)

    grid = torch.linspace(r.min(), r_max, n_grid, device=dev, dtype=dt)
    Ks = rbf(grid, X)
    mu = (Ks @ alpha).squeeze(-1)
    v = torch.cholesky_solve(Ks.t(), L)
    sd = (sf2 - (Ks * v.t()).sum(-1)).clamp_min(0).sqrt()

    return {
        # simple per-bin view (this alone may be all you need)
        "bin_center": ctr[keep], "bin_mean": Y, "bin_se": se[keep], "bin_n": cnt[keep],
        # smooth GP view
        "grid": grid, "gp_mean": mu, "gp_sd": sd,
        "capture": mu.clamp_max(0).exp(),
        "capture_lo": (mu - 2 * sd).clamp_max(0).exp(),  # pessimistic — read this
    }


def _snapshot(opt):
    return [p.detach().clone() for g in opt.param_groups
            for p in g['params'] if p.requires_grad]


def _update_ratio(opt, snap):
    cur = [p for g in opt.param_groups for p in g['params'] if p.requires_grad]
    dsq, wsq = 0.0, 0.0
    for p, p0 in zip(cur, snap):
        dsq += (p.detach() - p0).norm() ** 2
        wsq += p.detach().norm() ** 2
    return (dsq ** 0.5) / max(wsq ** 0.5, 1e-12)


def _uw_from_snaps(pre, post):
    upd_sq, wt_sq, max_ratio = 0.0, 0.0, 0.0
    for prev, cur in zip(pre, post):
        un = (cur - prev).norm().item()
        wn = prev.norm().item()
        upd_sq += un * un
        wt_sq += wn * wn
        if wn > 0:
            max_ratio = max(max_ratio, un / wn)
    return {'uw_global': (upd_sq ** 0.5) / (wt_sq ** 0.5 + 1e-12),
            'uw_max': max_ratio}
