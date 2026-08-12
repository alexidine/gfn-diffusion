"""
v0.1 unconditional conformational GFlowNet -- single molecule, CPU.

Runs parallel to train.py rather than through it. The model, trajectory machinery,
discretizer and TB objective are all the existing ones; what is stripped out is the
conditioning, the protocol/stage controller, the buffer controllers and the crystal
metric surface, none of which earn their keep before the sampler is shown to work at all.

Why the existing GFN works unmodified: a single molecule has a *fixed* number of
rotatable torsions, so the state is a fixed-dimension vector and none of the
variable-dimension machinery is needed. That only becomes necessary when conditioning
over molecules of different sizes.

Two things are deliberately different from the crystal setup:

*Every state dimension is periodic.* Torsions live on a torus, so ``TorsionGFN`` overrides
the crystal-specific periodic mask rather than editing ``gfn.py`` -- the running crystal
job is untouched. The sin/cos policy encoding and post-step wrapping already existed for
aunit periodicity and are reused as-is.

*The TB residual accumulates per-step differences.* ``(log_pf - log_pb).sum(-1)``, never
``log_pf.sum() - log_pb.sum()``. Each sum is O(d*T) and the residual is O(1), so the
subtraction-last form burns precision proportional to dimension: harmless at the crystal's
d=12, ruinous by d~100. Differencing first keeps every term O(1). The accumulator is
float64 regardless of policy dtype -- it is one scalar per trajectory.

    python train_conformer.py                                  # default config
    python train_conformer.py --yaml configs/conformer_dev.yaml

Config resolves exactly as train.py does (dict2namespace + resolve_derived_config),
via load_config() so a no-argument IDE run picks up the default rather than crashing.
"""

import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import wandb

from mxtaltools.common.training_utils import flatten_wandb_params

from energies.conformer_torsions import ConformerTorsions
from models.gfn import GFN
from controller import LRController
from train import Modeller, safe_histogram
from eval.evaluations import flow_parity_plot
from utils import (dict2namespace, load_yaml, preflight_config, quick_tb_stats,
                   resolve_derived_config, uniform_discretizer)

WANDB_PROJECT = "GFN Conformers"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "conformer_dev.yaml"


def load_config(default: Path = DEFAULT_CONFIG):
    """Config path from argv, falling back to the default.

    ``utils.get_train_args`` indexes ``remaining[1]`` directly, so launching with no
    arguments -- the normal case from an IDE run button -- raises IndexError before
    anything useful is printed. This accepts ``--yaml <path>``, a bare path, or
    nothing at all, and otherwise resolves the config exactly as train.py does.
    """
    candidates = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = Path(candidates[0]) if candidates else Path(default)
    if not path.exists():
        raise SystemExit(f"config not found: {path}\n"
                         f"usage: python train_conformer.py [--yaml <config.yaml>]  "
                         f"(default: {default})")
    if not candidates:
        print(f"no config given; using default {path}")
    return resolve_derived_config(preflight_config(dict2namespace(load_yaml(path))))


class TorsionGFN(GFN):
    """GFN whose entire state lives on a torus.

    ``get_periodic_dimensions`` in the base class hard-codes the crystal state layout
    (6 box params, then centroids, then orientations) and raises on any other dimension.
    Torsions are the simpler case -- every dimension wraps -- so the mask is overridden
    rather than the base class being taught about a second layout.
    """

    def get_periodic_dimensions(self, device, do_periodic_angles: bool = True,
                                periodic_centroid_axes=None,
                                dead_latent_rows=None, dead_latent_values=None):
        # Only the LAYOUT is overridden -- every torsion dimension wraps. The index-set
        # bookkeeping (block widths, expanded_dim, the ang/lin/dead partition and its
        # assertions, the pinned dead values) is delegated to the base class, so this
        # override cannot drift out of step with it. It previously set those six
        # attributes by hand, which is how the dead-row work broke conformer
        # construction: the base class grew seven more and this copy did not.
        #
        # dead rows are passed through rather than dropped: torsions have no
        # crystal-system projection so nothing is dead today (the conformer builder never
        # sets them, giving ()), but hardcoding None here would silently ignore them if
        # that ever changes.
        self.periodic_centroid_axes = ()
        self._finalize_dim_partition(device, [True] * self.dim,
                                     dead_latent_rows=dead_latent_rows,
                                     dead_latent_values=dead_latent_values)


def build_gfn(dim: int, mdl, device) -> TorsionGFN:
    gfn = TorsionGFN(
        dim=dim,
        s_emb_dim=mdl.s_emb_dim,
        conditions_dim=0,
        harmonics_dim=mdl.harmonics_dim,
        t_dim=mdl.t_dim,
        s_hidden_dim=mdl.policy_hidden_dim,
        policy_hidden_dim=mdl.policy_hidden_dim,
        policy_layers=mdl.policy_layers,
        flow_hidden_dim=mdl.flow_hidden_dim,
        flow_layers=mdl.flow_layers,
        t_scale=mdl.t_scale,
        learned_variance=True,
        conditional=False,
        learn_pb=False,
        zero_init=mdl.zero_init,
        clipping=mdl.clipping,
        gfn_clip=mdl.gfn_clip,
        device=device,
        do_periodic_angles=True,
    )
    return gfn.to(device)


@torch.no_grad()
def exact_references(energy, grid: int):
    """Quadrature references, or None past k=3 where the grid dies.

    Returns ``(log Z, E_p[log R], H[p], H_max)``. ``E_p[log R]`` is the one that
    matters most day to day: comparing the sampler's mean log-reward against it is a
    **Z-free** calibration check, so it stays meaningful when nothing else does.
    """
    k = energy.data_ndim
    if not grid or k > 3:
        return None
    ax = torch.linspace(-1.0, 1.0, grid + 1, dtype=energy.dtype, device=energy.device)[:-1]
    pts = torch.cartesian_prod(*([ax] * k)).reshape(-1, k)
    log_r = torch.cat([-energy.energy(pts[i:i + 65536])
                       for i in range(0, len(pts), 65536)])
    log_z = (torch.logsumexp(log_r, 0) + k * np.log(2.0 / grid)).item()
    w = torch.softmax(log_r, 0)
    e_log_r = (w * log_r).sum().item()
    return log_z, e_log_r, log_z - e_log_r, k * np.log(2.0)


@torch.no_grad()
def boltzmann_reference(energy, grid: int, n: int = 4096, seed: int = 0):
    """Exact Boltzmann draw by grid inversion. k <= 3 only, same cost as the references.

    Multinomial over the quadrature cells, then jittered uniformly within the drawn
    cell so the sample is continuous rather than a lattice. This is the *true* target
    to overlay the sampler against -- the buffer is a set of minima, which is a
    different thing and would make a correct sampler look over-dispersed.
    """
    k = energy.data_ndim
    if not grid or k > 3:
        return None
    ax = torch.linspace(-1.0, 1.0, grid + 1, dtype=energy.dtype, device=energy.device)[:-1]
    pts = torch.cartesian_prod(*([ax] * k)).reshape(-1, k)
    log_r = torch.cat([-energy.energy(pts[i:i + 65536])
                       for i in range(0, len(pts), 65536)])
    g = torch.Generator(device=energy.device).manual_seed(seed)
    idx = torch.multinomial(torch.softmax(log_r, 0), n, replacement=True, generator=g)
    cell = 2.0 / grid
    jitter = torch.rand((n, k), generator=g, dtype=energy.dtype, device=energy.device) * cell
    return (pts[idx] + jitter).cpu().numpy()


def torsion_latent_figure(samples, reference=None, n_kde: int = 200,
                          bw_factor: float = 0.05):
    """Per-torsion latent distributions, mirroring ``plot_batch_cell_params``.

    The conformer analogue of the crystal cell-parameter panel: one violin per state
    dimension, sampler overlaid on a reference. Coded rather than reused because that
    method is built around the crystal feature labels, Z' blocks and cell ranges, none
    of which have a meaning here -- the shared part is only the overlaid-violin idiom.

    Torsions are plotted in degrees on a fixed [-180, 180] range so panels stay
    comparable across steps, and so a distribution piling up at the wrap point is
    visible as mass at both edges rather than being silently re-centred.
    """
    from plotly.subplots import make_subplots

    from mxtaltools.reporting.utils import lightweight_one_sided_violin

    k = samples.shape[1]
    dists = [("sampler", 180.0 * samples, "rgba(60,120,216,0.55)")]
    if reference is not None:
        dists.append(("reference", 180.0 * reference, "rgba(235,104,52,0.45)"))

    fig = make_subplots(rows=1, cols=k,
                        subplot_titles=[f"torsion {j}" for j in range(k)])
    for j in range(k):
        for name, data, color in dists:
            # filled KDE line, not go.Violin: with orientation='h' and differing trace
            # names plotly gives each violin its own y-category, so violinmode='overlay'
            # never engages and the two distributions stack instead of superimposing
            x_v, y_v = lightweight_one_sided_violin(
                data[:, j], n_kde, bandwidth_factor=bw_factor,
                data_min=-180.0, data_max=180.0)
            fig.add_scatter(x=x_v, y=y_v, mode="lines", fill="toself",
                            fillcolor=color, line=dict(color=color, width=1.2),
                            name=name, legendgroup=name, showlegend=(j == 0),
                            row=1, col=j + 1)
        fig.update_xaxes(range=[-185, 185], dtick=90, row=1, col=j + 1)
        fig.update_yaxes(showticklabels=False, row=1, col=j + 1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=280, width=300 * k,
                      margin=dict(l=30, r=20, t=40, b=30))
    return fig


def wrap_state(x):
    """Wrap to the state space (-1, 1]. Period 2, NOT 2*pi -- see build_conformer_buffer."""
    return (x + 1.0) % 2.0 - 1.0


class TerminalBuffers:
    """Terminal states for backward training, split by role.

    Two buffers rather than one, because the two jobs are different and mixing them
    hides which is failing:

    ``reference`` -- the ANCHOR. States we believe in: an exact Boltzmann draw where
        the quadrature is affordable (k <= 3), else the local optima from
        build_conformer_buffer. Backward training on an exact draw is the strongest
        available statement that the machinery works, because the data *are* the target
        by construction -- if TB can't fit that, nothing downstream will.

    ``prior`` -- COVERAGE. Draws from a dumb prior. Quality is not the point and most
        states are bad; the requirement is only that the support contains every mode.
        Uniform on the torus is the correct dumb prior for a torsion when nothing
        better is loaded, since it is the maximum-entropy distribution on the space.

    Everything is wrapped on the way in regardless of source. Old buffer files predate
    the period-2 fix and hold states out to +/-565 degrees, which would be fed to
    get_traj_bwd as terminal states outside the sampler's domain.
    """

    def __init__(self, reference=None, prior=None, prior_frac: float = 0.5):
        self.reference = None if reference is None else wrap_state(reference)
        self.prior = None if prior is None else wrap_state(prior)
        self.prior_frac = 0.0 if self.prior is None else float(prior_frac)
        if self.reference is None:
            self.prior_frac = 1.0 if self.prior is not None else 0.0

    def __bool__(self):
        return self.reference is not None or self.prior is not None

    def describe(self) -> str:
        parts = []
        for name in ("reference", "prior"):
            b = getattr(self, name)
            parts.append(f"{name} {'--' if b is None else len(b)}")
        return f"terminal buffers: {', '.join(parts)}   prior_frac {self.prior_frac:.2f}"

    def draw(self, n: int, device):
        """``n`` terminal states, split between the two buffers by ``prior_frac``."""
        n_prior = int(round(n * self.prior_frac))
        n_ref = n - n_prior
        out = []
        if n_ref and self.reference is not None:
            out.append(self.reference[torch.randint(len(self.reference), (n_ref,))])
        if n_prior and self.prior is not None:
            out.append(self.prior[torch.randint(len(self.prior), (n_prior,))])
        return torch.cat(out).to(device)


def build_terminal_buffers(energy, train_c, eval_c, dim) -> TerminalBuffers:
    """Assemble the reference and prior buffers, reporting what actually got used.

    The fallbacks are deliberate but they must be *loud*: a silent fallback to uniform
    is how a run ends up training against a buffer nobody chose.
    """
    ref = None
    source = getattr(train_c, "reference_source", "auto")
    path = getattr(train_c, "buffer_path", None)
    path = Path(path) if path else None

    if source in ("auto", "exact"):
        draw = boltzmann_reference(energy, eval_c.brute_force_grid,
                                   n=int(getattr(train_c, "reference_size", 8192)))
        if draw is not None:
            ref = torch.as_tensor(draw, dtype=torch.float64)
            print(f"reference buffer: {len(ref)} EXACT Boltzmann states by grid inversion")
        elif source == "exact":
            raise SystemExit("reference_source='exact' needs brute_force_grid > 0 and k <= 3")

    if ref is None and path is not None and path.exists():
        blob = torch.load(path, weights_only=False)
        key = "modes" if source in ("auto", "modes") and "modes" in blob else "states"
        ref = torch.as_tensor(blob[key], dtype=torch.float64)
        print(f"reference buffer: {len(ref)} local optima ('{key}') from {path}")
    elif ref is None and path is not None:
        print(f"reference buffer: {path} not found")

    prior = None
    p_path = getattr(train_c, "prior_states_path", None)
    n_prior = int(getattr(train_c, "prior_size", 20000))
    if p_path and Path(p_path).exists():
        blob = torch.load(Path(p_path), weights_only=False)
        prior = torch.as_tensor(blob["states"] if isinstance(blob, dict) else blob,
                                dtype=torch.float64)
        print(f"prior buffer: {len(prior)} states from {p_path}")
    elif n_prior > 0:
        prior = torch.rand(n_prior, dim, dtype=torch.float64) * 2 - 1
        print(f"prior buffer: {n_prior} UNIFORM-on-torus states "
              f"(no prior_states_path given; this is the max-entropy dumb prior)")

    return TerminalBuffers(ref, prior, getattr(train_c, "prior_frac", 0.5))


@torch.no_grad()
def evaluate(gfn, energy, discretizer, batch_size, device, log_temp, refs):
    """On-policy eval pass: no exploration, no gradients.

    Separate from the training step on purpose. The fitted log Z reflects whatever
    distribution the trajectories came from, so reading calibration off a training
    batch conflates the model with the sampling scheme. Everything here is measured
    from the policy as it would actually be used.
    """
    init = torch.zeros(batch_size, energy.data_ndim, device=device)
    states, log_pf, log_pb, log_flow, gauss = gfn.get_traj_fwd(
        init, discretizer, torch.zeros(batch_size, device=device),
        condition=False, mol_batch=None, return_gauss_params=True)
    final = states[:, -1]
    log_reward = -energy.energy(final, None, log_temp)
    log_z = log_flow[:, 0]
    _, residual = tb_loss(log_z, log_pf, log_pb, log_reward)

    out = {
        "eval/log_z": log_z.mean().item(),
        "eval/mean_log_reward": log_reward.mean().item(),
        "eval/residual_mean": residual.mean().item(),
        "eval/residual_std": residual.std().item(),
        # log Z_est - <log R> is an entropy-like quantity; on-policy it cannot exceed
        # the state space's maximum entropy. Headroom < 0 means the reported density
        # is not normalised, which no amount of further training will fix.
        "eval/implied_entropy": (log_z.mean() - log_reward.mean()).item(),
        "dist/eval_log_reward": safe_histogram(log_reward.cpu().numpy()),
        "dist/energy": safe_histogram((-log_reward).cpu().numpy()),
    }
    for j in range(energy.data_ndim):
        out[f"dist/torsion_{j}"] = safe_histogram(180.0 * final[:, j].cpu().numpy())

    # SDE parameters: gauss_params are already per-step means over the batch, so these
    # are the trajectory-mean drift and log-variance of each Gaussian kernel. logvars_f
    # against logvars_b is the readable one -- the forward policy inflating away from
    # the fixed backward kernel is the variance walk that precedes a blow-up.
    for key, val in gauss.items():
        out[f"sde/{key}_mean"] = val.mean().item()
        out[f"sde/{key}_std"] = val.std().item()
    out["dist/sde_logvars_f"] = safe_histogram(gauss["logvars_f"].cpu().numpy())
    # THE guard for a periodic state space: the reported transition density is a plain
    # Gaussian, which is only the right density while a step cannot reach around the
    # wrap. Past ~0.25 the missing wrapped images are O(1) per step and log Z inflates
    # while every self-consistency metric keeps improving.
    sigma_step = (0.5 * gauss["logvars_f"].exp()).sqrt().mean().item()
    out["sde/sigma_step"] = sigma_step
    out["sde/sigma_over_halfperiod"] = sigma_step / 1.0

    # quick_tb_stats: the same control-metric family train.py reads. It wants
    # trajectory-summed [B] (flow_parity_plot sums internally, this one does not), and
    # condition_id=None means the per-condition entries collapse onto the pooled ones,
    # so only the pooled numbers carry information here.
    # NB the summed form reintroduces the subtract-last cancellation the training loss
    # avoids -- harmless at d=2 (log P ~ 1e3 against an O(1) residual) but worth
    # revisiting before this is used at the dimensions the full DoF set would reach.
    out.update({f"tb/{k}": v for k, v in quick_tb_stats(
        log_pf.sum(-1), log_pb.sum(-1), log_z, log_reward).items()})

    if refs is not None:
        gt_log_z, gt_e_log_r, _, h_max = refs
        out["eval/dlog_z"] = out["eval/log_z"] - gt_log_z
        # Z-free: survives past the point where exact log Z does not
        out["eval/dlog_reward"] = out["eval/mean_log_reward"] - gt_e_log_r
        out["eval/entropy_headroom"] = h_max - out["eval/implied_entropy"]
    return out, final, (log_reward, log_z, log_pb, log_pf)


class ConformerModeller:
    """Minimal Modeller surface so train.py's LR and batch machinery runs unmodified.

    ``LRController`` reaches into its owner for exactly ``step_ind``, ``args``,
    ``phase``, ``optimizers`` and ``lr_ctrl``, and ``increment_batch_size`` for the
    batch-growth state plus a step-time window. Providing that interface lets both be
    used as-is rather than reimplemented, so their semantics stay identical to the
    crystal runs and any fix there lands here too.

    ``phase`` is constant and ``protocol`` is None: v0.1 has a single stage, so the
    per-stage warmup rearm and the knee's stage pinning are inert rather than absent.
    """

    def __init__(self, args, gfn, device):
        self.args = args
        self.gfn_model = gfn
        self.device = device
        self.step_ind = 0
        self.phase = 1                  # single stage; LRController keys its ramp off this
        self.protocol = None
        self.lr_ctrl = None             # LRController builds and owns this dict
        self.batch_size = int(args.training.batch_size)
        self.args.max_batch_size = int(args.training.max_batch_size)
        self.batch_size_ever_oomed = False
        self.batch_size_cooldown_until = 0
        self.batch_size_last_grow = 0
        self.batch_size_saturated_stage = None
        self.batch_size_pinned_at = 0
        self._rung_throughput = None
        self._recent_step_times = deque(maxlen=50)
        self.last_grad_norm_pre_clip = float('nan')
        self.grad_nonfinite = 0
        self._throughput = {'samples': 0, 'seconds': 0.0}
        self.init_schedulers_optimizers()
        self.lr_controller = LRController(self)

    def init_schedulers_optimizers(self):
        """Mirrors train.py: policy optimizers per mode, flow on its own LR.

        The warmup division is the same -- optimizers are constructed at
        lr / lr_warmup_ratio because the first train_step runs before the controller's
        first tick, so they must not start cold-Adam at the full operating LR.
        """
        a = self.args
        init_flow_lr = a.lr_flow
        init_policy_lrs = {'fwd': a.lr_policy / a.lr_warmup_ratio,
                           'bwd': a.lr_back / a.lr_warmup_ratio,
                           'replay': a.lr_replay / a.lr_warmup_ratio,
                           'fused': a.lr_fused / a.lr_warmup_ratio}
        wd = a.weight_decay if getattr(a, 'use_weight_decay', False) else 0

        def policy_params(g):
            return [{'params': g.t_model.parameters()},
                    {'params': g.s_model.parameters()},
                    {'params': g.forward_policy.parameters()},
                    {'params': g.backward_policy.parameters()}]

        self.optimizers = {}
        for mode in ('fwd', 'bwd', 'replay'):
            self.optimizers[mode] = torch.optim.Adam(
                policy_params(self.gfn_model), init_policy_lrs[mode], weight_decay=wd)
        self.optimizers['fused'] = torch.optim.Adam(
            policy_params(self.gfn_model)
            + [{'params': self.gfn_model.flow_model.parameters(), 'lr': init_flow_lr}],
            init_policy_lrs['fused'], weight_decay=wd)
        self.optimizers['flow'] = torch.optim.Adam(
            self.gfn_model.flow_model.parameters(), init_flow_lr, weight_decay=wd)

    def step_lr_schedule(self):
        return self.lr_controller.step()

    increment_batch_size = Modeller.increment_batch_size

    def fast_metrics(self):
        """train.py's ``ten_step_reporting``, restricted to what v0.1 has.

        These are cheap scalars whose *shape over time* is the diagnosis: the LR
        ramp/hold/decay, the batch ladder, the grad-norm envelope. Reporting them
        only at ``report_every`` would sample the schedule at 1/250 resolution --
        coarser than the schedule itself moves, so a ramp reads as a step function
        and a single-report LR cut is invisible.
        """
        m = {f'lr/{name}': opt.param_groups[0]['lr']
             for name, opt in self.optimizers.items()}
        # the flow group rides at the END of the fused optimizer on its own LR;
        # the standalone 'flow' entry above is the one v0.1 actually steps
        m['lr/fused_flow'] = self.optimizers['fused'].param_groups[-1]['lr']

        m['train/batch_size'] = self.batch_size
        m['train/grad_norm_pre_clip'] = self.last_grad_norm_pre_clip
        # count since the last call, not a rate -- drained here
        m['train/grad_nonfinite'] = self.grad_nonfinite
        self.grad_nonfinite = 0
        if self._recent_step_times:
            m['train/step_time'] = float(np.median(self._recent_step_times))
        # samples/sec over the window, not batch/step_time: batch_size is a moving
        # denominator once growth is on, so the two stop agreeing
        if self._throughput['seconds'] > 0:
            m['train/samples_per_sec'] = (self._throughput['samples']
                                          / self._throughput['seconds'])
        self._throughput = {'samples': 0, 'seconds': 0.0}

        m.update(self.lr_controller.report())
        return m


def tb_loss(log_z, log_pf, log_pb, log_reward):
    """Trajectory-balance residual, differenced per step before summing.

    ``log_pf``/``log_pb`` arrive as ``[B, T]``. Summing each and subtracting would form
    an O(1) result from two O(d*T) quantities; differencing first keeps everything O(1).
    """
    delta = (log_pf.double() - log_pb.double()).sum(-1)
    residual = log_z.double() + delta - log_reward.double()
    return residual.pow(2).mean(), residual


def main():
    args = load_config()
    # flatten_wandb_params mutates args in place (it merges the flattened keys back
    # into __dict__), so grab the nested sections *after* it runs -- otherwise the
    # section handles are fine but the ordering reads as though they might not be.
    wandb_config = flatten_wandb_params(args)
    with wandb.init(project=WANDB_PROJECT,
                    config=wandb_config,
                    name=args.run_name,
                    tags=[args.tag],
                    mode=getattr(args, "wandb_mode", "online")):
        run(args)


def run(args):
    prob, mdl = args.problem, args.model
    train_c, eval_c = args.training, args.eval

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device = torch.device(getattr(args, "device", "cpu"))
    torch.set_num_threads(getattr(args, "num_threads", 2))

    energy = ConformerTorsions(smiles=prob.smiles, device=str(device),
                               log_temperature=prob.log_temperature,
                               epsilon=prob.epsilon,
                               min_separation=prob.min_separation,
                               scale_14=prob.scale_14,
                               lj_k_factor=prob.lj_k_factor,
                               include_trivial_rotations=prob.include_trivial_rotations)
    print(energy.describe())
    dim = energy.data_ndim

    t0 = time.time()
    refs = exact_references(energy, eval_c.brute_force_grid)
    gt_log_z = refs[0] if refs else None
    if refs:
        gt_log_z, gt_e_log_r, gt_h, h_max = refs
        print(f"exact references (grid {eval_c.brute_force_grid}^{dim}, {time.time() - t0:.1f} s): "
              f"log Z {gt_log_z:.4f}   E_p[log R] {gt_e_log_r:.4f}   "
              f"H[p] {gt_h:.4f} of max {h_max:.4f}")
    elif dim > 3:
        print(f"k={dim}: too many torsions for brute-force references; calibration this "
              f"run is relative only (see the validation ladder)")

    gfn = build_gfn(dim, mdl, device)
    mod = ConformerModeller(args, gfn, device)   # owns optimizers, LR controller, batch size
    discretizer = lambda bsz: uniform_discretizer(bsz, args.integrator.T)

    buffers = (build_terminal_buffers(energy, train_c, eval_c, dim)
               if float(getattr(train_c, "bwd_frac", 0.0)) > 0 else TerminalBuffers())
    print(buffers.describe() if buffers else "no terminal buffers; forward-only")

    log_temp = torch.tensor(prob.log_temperature, device=device)
    # figure overlay: the exact Boltzmann draw where we have it, since that is the true
    # target. The local-optima buffer is a set of MINIMA -- overlaying against it would
    # make a correctly-dispersed sampler look over-dispersed.
    reference_states = None
    if refs is not None:
        reference_states = boltzmann_reference(energy, eval_c.brute_force_grid)
    elif buffers.reference is not None:
        reference_states = buffers.reference[
            torch.randperm(len(buffers.reference))[:4096]].cpu().numpy()
    history, t_start, bwd_seen = [], time.time(), 0.0

    for step in range(train_c.steps):
        mod.step_ind = step
        step_t0 = time.time()
        if step % 10 == 0:                       # same cadence train.py uses
            mod.step_lr_schedule()
        if getattr(train_c, "grow_batch_size", False):
            mod.increment_batch_size()
        batch_size = mod.batch_size
        for opt in mod.optimizers.values():
            opt.zero_grad(set_to_none=True)
        # bwd_frac is the target share of steps trained backward. Drawing it per step
        # (rather than strict alternation) keeps the realised share equal to the knob at
        # any value, not just 0.5 -- train/bwd_fraction reports what actually happened.
        use_bwd = bool(buffers) and np.random.rand() < float(getattr(train_c, "bwd_frac", 0.0))

        if use_bwd:
            terminal = buffers.draw(batch_size, device)
            states, log_pf, log_pb, log_flow = gfn.get_traj_bwd(
                terminal, discretizer, condition=False, mol_batch=None)
            final = terminal
        else:
            init = torch.zeros(batch_size, dim, device=device)
            # exploration_std is per-sample ([B]), not a scalar -- fwd_get_logvars
            # broadcasts it against the per-dimension log-variance head
            expl = torch.full((batch_size,), train_c.exploration_std, device=device)
            states, log_pf, log_pb, log_flow = gfn.get_traj_fwd(
                init, discretizer, expl, condition=False, mol_batch=None)
            final = states[:, -1]

        log_reward = -energy.energy(final, None, log_temp)
        log_z = log_flow[:, 0]          # same source train.py uses: log_Z_learned = log_flow[:, 0]
        loss, residual = tb_loss(log_z, log_pf, log_pb, log_reward)
        loss.backward()
        # clip_grad_norm_ returns the norm BEFORE clipping -- the only free read of
        # how hard the clip is biting, and the thing to look at first when the LR
        # schedule and the loss disagree
        gnorm = torch.nn.utils.clip_grad_norm_(gfn.parameters(), args.gradient_norm_clip)
        mod.last_grad_norm_pre_clip = float(gnorm)
        mod.grad_nonfinite += int(not np.isfinite(mod.last_grad_norm_pre_clip))
        # turn-taking: the direction's own policy optimizer, plus the flow (Z) optimizer,
        # exactly as train.py's non-fused branch does
        mod.optimizers["bwd" if use_bwd else "fwd"].step()
        mod.optimizers["flow"].step()
        step_dt = time.time() - step_t0
        mod._recent_step_times.append(step_dt)
        mod._throughput['samples'] += batch_size
        mod._throughput['seconds'] += step_dt

        bwd_seen += float(use_bwd)
        reporting = step % eval_c.report_every == 0 or step == train_c.steps - 1
        # cheap scalars at schedule resolution; merged into the report dict below
        # when the two cadences land on the same step, so wandb sees one commit
        fast = mod.fast_metrics() if (step % 10 == 0 or reporting) else {}
        if reporting:
            with torch.no_grad():
                metrics = {
                    "train/loss": loss.item(),
                    # residual_mean is structurally pinned near 0: log Z is free to
                    # absorb it, so it carries almost no information. residual_std is
                    # the convergence signal.
                    "train/residual_mean": residual.mean().item(),
                    "train/residual_std": residual.std().item(),
                    "train/log_z": log_z.mean().item(),
                    "train/mean_log_reward": log_reward.mean().item(),
                    "train/bwd_fraction": bwd_seen / max(step + 1, 1),
                    "train/steps_per_second": (step + 1) / (time.time() - t_start),
                    "train/wall_hours": (time.time() - t_start) / 3600.0,
                    "train/direction": float(use_bwd),
                    "dist/residual": safe_histogram(residual.cpu().numpy()),
                }
                metrics.update(fast)
                ev, final, parity_in = evaluate(
                    gfn, energy, discretizer, batch_size, device, log_temp, refs)
                metrics.update(ev)

                figs_period = getattr(eval_c, "figs_period", 0)
                if figs_period and step % figs_period == 0:
                    lr_e, lz_e, lpb_e, lpf_e = parity_in
                    parity, _ = flow_parity_plot(lr_e, lz_e, lpb_e, lpf_e)
                    metrics["fig/tb_parity"] = wandb.Plotly(parity)
                    metrics["fig/latents"] = wandb.Plotly(
                        torsion_latent_figure(final.cpu().numpy(), reference_states))
                wandb.log(metrics, step=step, commit=True)

                rec = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                rec["step"] = step
                rec["direction"] = "bwd" if use_bwd else "fwd"
                history.append(rec)

                dz = f"  dlogZ {ev['eval/dlog_z']:+7.3f}" if "eval/dlog_z" in ev else ""
                dr = f"  dlogR {ev['eval/dlog_reward']:+6.3f}" if "eval/dlog_reward" in ev else ""
                hh = (f"  Hgap {ev['eval/entropy_headroom']:+6.2f}"
                      if "eval/entropy_headroom" in ev else "")
                print(f"  {step:>6}  loss {metrics['train/loss']:9.3f}  "
                      f"resid {metrics['train/residual_mean']:+7.3f} +/- "
                      f"{metrics['train/residual_std']:5.3f}  "
                      f"logZ {ev['eval/log_z']:8.3f}{dz}{dr}{hh}  "
                      f"[{'bwd' if use_bwd else 'fwd'}]")
        elif fast:
            # the loss rides along at the fast cadence too: it is already computed,
            # and at 1/250 it is too sparse to tell a spike from a trend
            with torch.no_grad():
                fast["train/loss"] = loss.item()
                fast["train/residual_std"] = residual.std().item()
            wandb.log(fast, step=step, commit=True)

    summary = dict(run_name=args.run_name, tag=args.tag, smiles=prob.smiles, dim=dim,
                   gt_log_z=gt_log_z,
                   gt_e_log_reward=refs[1] if refs else None,
                   gt_entropy=refs[2] if refs else None,
                   max_entropy=refs[3] if refs else None,
                   wall_seconds=time.time() - t_start, history=history)
    Path(eval_c.out).write_text(json.dumps(summary, indent=2))
    for k in ("gt_log_z", "gt_e_log_reward", "gt_entropy", "max_entropy", "wall_seconds"):
        wandb.run.summary[k] = summary[k]
    print(f"\nwrote {eval_c.out}")


if __name__ == "__main__":
    main()
