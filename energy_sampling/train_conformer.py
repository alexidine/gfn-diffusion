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

*The state layout comes from the energy.* ``ConformerTorsions.periodic_dims`` declares
which dims wrap, and is handed to ``GFN(angular_mask=...)``. At `torsion` and `dihedral`
that is every dim; from `flex` up the r and theta blocks are linear and bounded. This
replaces the old ``TorsionGFN`` subclass, which overrode ``get_periodic_dimensions``
wholesale. Note the subclass could not simply be DELETED: the base method's non-crystal
branch writes ``[False] * dim``, which hands a torsion state zero wrapped dims silently,
and a 2-periodic reward with no wrap has no finite log Z at all. The sin/cos policy
encoding and post-step wrapping already existed for aunit periodicity and are reused.

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
from grad_clip_guard import GradClipGuard
from train import Modeller, safe_histogram
from eval.evaluations import flow_parity_plot
from utils import (dict2namespace, load_yaml, preflight_config, quick_tb_stats,
                   resolve_derived_config, uniform_discretizer)

WANDB_PROJECT = "GFN Conformers"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "conformer_dev.yaml"
#: WHERE A BARE ``eval.out`` LANDS. conformer_dev.yaml names an output *file*,
#: not a path, and ``Path(name).write_text`` resolved it against the CWD --
#: always ``energy_sampling/``. Resolving here leaves the config owning the NAME
#: and this module owning the DIRECTORY, so the user-owned yaml needs no edit.
RESULTS_DIR = Path(__file__).resolve().parent / "energies" / "results"


def resolve_out(out) -> Path:
    """Bare filename -> ``energies/results/``; an explicit path is left alone."""
    p = Path(out)
    if p.is_absolute() or p.parent != Path("."):
        return p
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / p.name


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


def build_gfn(dim: int, mdl, device, angular_mask) -> GFN:
    """Construct the policy against the layout the ENERGY declares.

    ``angular_mask`` comes from ``ConformerTorsions.periodic_dims`` and is required, not
    optional: the base class's fallback for a non-crystal state is ``[False] * dim``,
    which is a silently unnormalizable target rather than a merely degraded layout.
    Passing it explicitly is what retired the ``TorsionGFN`` subclass.
    """
    if len(angular_mask) != dim:
        raise ValueError(f"angular_mask has {len(angular_mask)} entries for dim {dim}")
    gfn = GFN(
        dim=dim,
        angular_mask=angular_mask,
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

    THIS DOES NOT SCALE PAST `torsion`, IN THREE SEPARATE WAYS. Needs an abstraction; all
    three are live at `flex` and `full` today.

    1. WIDTH IS UNBOUNDED. One row of `k` panels at `width=300 * k` -- 9000 px at
       propanol/full (d=30), and worse on anything real. A grid helps but only to about
       15-18 panels; past that no per-dimension layout is readable.
    2. EVERY PANEL IS LABELLED "torsion j", WHICH IS FALSE ABOVE `dihedral`. At `full` the
       first n_r columns are BOND LENGTHS and the next n_th are ANGLES. The energy already
       knows -- `_free_block` is 0/1/2 per state column -- so the label is available and
       simply is not read.
    3. EVERY COLUMN IS SCALED BY 180 AND RANGED [-185, 185] AS DEGREES. That is meaningful
       only for the phi block. The r and theta columns are linear box coordinates, so the
       axis is wrong for them and a bond-length distribution is being drawn as though it
       were an angle. `periodic_dims` is the discriminator and is already passed to the
       policy.

    THE AGREED FRAMING (2026-08-20) IS SCALARS FIRST, FIGURE AS DRILL-DOWN. What you want
    to see depends on what the reference IS, and the code already knows -- TerminalBuffers
    carries `reference` and `prior` separately, and prior_frac == 1.0 means "no true target
    exists here". Three questions, three different scalars, and the directionality differs:

      * REFERENCE IS THE PRIOR -> coverage, and it is ONE-SIDED. What matters is prior mass
        the sampler has abandoned, not sampler mass the prior lacks -- concentrating is the
        point of training, so a symmetric divergence would penalise success.
      * REFERENCE IS THE TARGET -> agreement, symmetric. Per-column Wasserstein or KS.
      * WATCHING FOR PATHOLOGIES -> each named one is itself a scalar: variance explosion
        (per-column sd ratio, max over columns), mode collapse (entropy drop), wall-piling
        (mass within eps of +-1), wrap discontinuity (mass at both edges of a periodic
        column).

    So the distribution plot is not the primary diagnostic; it is what you open when a
    scalar trips, over only the columns that tripped -- which is a handful by construction
    and therefore dissolves the layout problem above.

    DIMENSION REDUCTION WAS CONSIDERED AND REJECTED. A projection over the latent box mixes
    incommensurable units -- r in angstrom, theta in radians, phi in radians ON A CIRCLE --
    so a PCA over it is a weighted sum of things sharing no metric, and the periodic columns
    do not embed in Euclidean space without picking a branch cut. The thermal widths could
    whiten it, but the projection would still cross the periodic/linear boundary.

    Options for the per-column layout if one is still wanted, none chosen:
      * GROUP BY DoF CLASS -- three panels (r, theta, phi) with every column of a class
        overlaid. Scales to any d; loses per-column identity.
      * WORST-K BY DIVERGENCE -- rank columns by 1-D Wasserstein against the reference and
        show the K worst plus a summary of the rest. Scales, and surfaces the failure
        rather than averaging it away.
      * DENSITY-DIFFERENCE HEATMAP -- columns on one axis, latent value on the other,
        colour = sampler density minus reference density. One panel at any d, and zero
        means agreement, so disagreement is what draws the eye.
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


def wrap_state(x, periodic_mask):
    """Wrap the PERIODIC columns to (-1, 1]. Period 2, NOT 2*pi.

    `periodic_mask` is required, not optional. From `flex` up the state carries linear
    blocks (r, theta), and wrapping one of those is not merely imprecise: a bond-length
    latent at 1.3 folds to -0.7 -- the opposite corner of the box -- with a perfectly
    plausible energy and no error anywhere. The three data-prep scripts keep their own
    unconditional copies of this, which is correct there only because they are pinned to
    level='torsion' where every column is periodic.
    """
    m = torch.as_tensor(periodic_mask, dtype=torch.bool, device=x.device)
    if m.numel() != x.shape[-1]:
        raise ValueError(f"periodic_mask has {m.numel()} entries for a {x.shape[-1]}-wide state")
    return torch.where(m, (x + 1.0) % 2.0 - 1.0, x)


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

    def __init__(self, reference=None, prior=None, prior_frac: float = 0.5,
                 periodic_mask=None):
        self._pmask = periodic_mask
        self.reference = None if reference is None else self._admit(reference, 'reference')
        self.prior = None if prior is None else self._admit(prior, 'prior')
        self.prior_frac = 0.0 if self.prior is None else float(prior_frac)
        if self.reference is None:
            self.prior_frac = 1.0 if self.prior is not None else 0.0

    def _admit(self, states, which: str):
        """Wrap the periodic columns, and REFUSE a linear column that is out of the box.

        A stored file written against a narrower level, or by a build script that wrapped
        every column, arrives here looking exactly like a valid one. The periodic columns
        are wrapped (a total map, so always safe); the linear ones cannot be repaired
        without guessing, so they raise.
        """
        if self._pmask is None:
            raise ValueError("TerminalBuffers needs the energy's periodic_dims")
        states = wrap_state(states, self._pmask)
        lin = ~torch.as_tensor(self._pmask, dtype=torch.bool, device=states.device)
        if lin.any():
            bad = (states[:, lin].abs() > 1.0 + 1e-9)
            if bad.any():
                worst = float(states[:, lin].abs().max())
                raise ValueError(
                    f"{which} buffer has {int(bad.any(dim=1).sum())} row(s) whose LINEAR "
                    f"columns leave the box (max |x| = {worst:.4g}). Wrapping them would "
                    f"fold a bond length or angle to the opposite corner silently, so this "
                    f"refuses instead. The file was almost certainly written at a "
                    f"different level, or by a script that wraps every column.")
        return states

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
            ref = torch.as_tensor(draw, dtype=torch.get_default_dtype())
            print(f"reference buffer: {len(ref)} EXACT Boltzmann states by grid inversion")
        elif source == "exact":
            raise SystemExit("reference_source='exact' needs brute_force_grid > 0 and k <= 3")

    if ref is None and path is not None and path.exists():
        blob = torch.load(path, weights_only=False)
        key = "modes" if source in ("auto", "modes") and "modes" in blob else "states"
        ref = torch.as_tensor(blob[key], dtype=torch.get_default_dtype())
        print(f"reference buffer: {len(ref)} local optima ('{key}') from {path}")
    elif ref is None and path is not None:
        # an explicitly-configured buffer that is not there is a config error, not a
        # reason to fall through to uniform. Printing it is how a run ends up training
        # against a buffer nobody chose.
        raise SystemExit(f"training.buffer_path was set to {path}, which does not exist")

    prior = None
    p_path = getattr(train_c, "prior_states_path", None)
    ip_path = getattr(train_c, "internal_prior_path", None)
    n_prior = int(getattr(train_c, "prior_size", 20000))
    if p_path and Path(p_path).exists():
        blob = torch.load(Path(p_path), weights_only=False)
        prior = torch.as_tensor(blob["states"] if isinstance(blob, dict) else blob,
                                dtype=torch.get_default_dtype())
        if prior.shape[1] != dim:
            raise SystemExit(
                f"prior_states_path {p_path} holds {prior.shape[1]}-wide states but this "
                f"problem is {dim}-wide (level {energy.level!r}). One file is one level.")
        print(f"prior buffer: {len(prior)} states from {p_path}")
    elif ip_path:
        # A FITTED prior over internal coordinates, drawn per-DoF in this energy's own
        # spec numbering (see ConformerTorsions.sample_prior_states). This is the thing
        # that makes `flex` and `full` trainable: uniform-on-box puts every bond length
        # and angle anywhere in its box independently, which for propanol is ~300 kcal/mol
        # of strain in every backward terminal.
        if not Path(ip_path).exists():
            raise SystemExit(f"training.internal_prior_path {ip_path} does not exist")
        if energy.collective:
            raise SystemExit(
                f"internal_prior_path needs a selection level; {energy.level!r} has "
                f"collective columns. Use prior_states_path with a file from "
                f"build_prior_states.py for the torsion route.")
        fitted = torch.load(Path(ip_path), weights_only=False)
        print(f"prior: fitted InternalPrior from {ip_path} ({fitted.n_fitted} molecules; "
              f"{len(fitted.bonds)} bond / {len(fitted.angles)} angle / "
              f"{len(fitted.torsions)} torsion types)")
        prior, _ = energy.sample_prior_states(
            fitted, n_prior, np.random.default_rng(int(getattr(train_c, "prior_seed", 0))))
        prior = prior.to(torch.get_default_dtype())
    elif n_prior > 0:
        prior = torch.rand(n_prior, dim, dtype=torch.get_default_dtype()) * 2 - 1
        print(f"prior buffer: {n_prior} UNIFORM-on-box states (no prior_states_path or "
              f"internal_prior_path; this is the max-entropy dumb prior, and from `flex` "
              f"up it is a very dumb one -- every bond length independently uniform)")

    return TerminalBuffers(ref, prior, getattr(train_c, "prior_frac", 0.5),
                           periodic_mask=energy.periodic_dims)


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
    ``phase``, ``optimizers`` and ``lr_ctrl``, and ``select_batch_size`` for the
    batch-sizer state plus a step-time window. Providing that interface lets both be
    used as-is rather than reimplemented, so their semantics stay identical to the
    crystal runs and any fix there lands here too.

    ``phase`` is constant and ``protocol`` is None: v0.1 has a single stage, so the
    per-stage clearing of the sizer's conclusion is inert rather than absent. No
    conformer config sets ``batch_util_target``, so the sizer holds the base batch
    (S3) and only the OOM/runaway bounds ever move it.
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
        # BOTH bounds are mirrored onto `args`, because train.py's batch machinery reads
        # them from there rather than from the modeller. `batch_size` is the CONFIGURED
        # base and must not be confused with `self.batch_size`, which select_batch_size
        # mutates -- the floor is what the run was designed around, not where it currently
        # sits, and sourcing it from the mutable field would let the domain drift upward
        # with every rung the sizer climbs.
        self.args.batch_size = int(args.training.batch_size)
        self.args.max_batch_size = int(args.training.max_batch_size)
        self.batch_size_cooldown_until = 0
        self.batch_size_last_grow = 0
        self.batch_sizer = None
        self._recent_step_times = deque(maxlen=50)
        self._recent_step_work = deque(maxlen=50)
        self.last_grad_norm_pre_clip = float('nan')
        self.grad_nonfinite = 0
        self._throughput = {'samples': 0, 'seconds': 0.0}
        # Same adaptive clip bar as the crystal route. v0.1 has one stage and no
        # protocol, so there is no transition to refresh on -- the tracker's own
        # rate limit is the only adaptation here.
        self.grad_guard = GradClipGuard.from_config(
            args.gradient_norm_clip, getattr(args, 'grad_clip_guard', None))
        self.grad_guard.announce()
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

    def _batch_floor(self) -> int:
        """The configured batch size, as the batch sizer's lower domain bound.

        MIRRORS train.py's Modeller._batch_floor deliberately rather than importing it:
        ConformerModeller is a duck-typed adapter, not a subclass, so every member
        train.py's machinery reaches for has to exist here by hand. That is exactly how
        this broke -- select_batch_size grew a _batch_floor() call and the adapter did not
        follow, so the conformer entry point raised AttributeError before step 0.

        If the two definitions ever need to differ, that is a signal the adapter should
        become a subclass instead.
        """
        return max(1, min(int(self.args.batch_size), int(self.args.max_batch_size)))


    def step_lr_schedule(self):
        return self.lr_controller.step()

    select_batch_size = Modeller.select_batch_size
    _conclude_batch_calibration = Modeller._conclude_batch_calibration
    _gpu_util_mean = Modeller._gpu_util_mean
    _now = Modeller._now

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
        # per-branch clip bar + firing rate; empty dict when the guard is off
        m.update(self.grad_guard.report())
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
    # float32 EVERYWHERE for the conformer track. Measured against float64 on identical
    # draws, the relative error is ~5e-7 on the potential, ~6e-7 on log J and ~1e-6 A on
    # the closure bond -- four orders below the quantities being reported, with nothing
    # compounding through the NeRF chain. The two places precision genuinely matters are
    # exempt: the TB residual accumulator, which stays float64 by its own explicit cast
    # (tb_loss does log_pf.double() - log_pb.double(), one scalar per trajectory and the
    # one place cancellation actually bites).
    #
    # THE BUFFERS ARE NOT EXEMPT, and an earlier version of this comment said they were.
    # They hold terminal STATES on [-1, 1] that are fed straight to the policy, so pinning
    # them to float64 does not preserve precision -- it raises
    # `expected mat1 and mat2 to have the same dtype` at the first backward step. Storage
    # precision is only free where the tensor never meets a parameter.
    torch.set_default_dtype(torch.float32)
    device = torch.device(getattr(args, "device", "cpu"))
    torch.set_num_threads(getattr(args, "num_threads", 2))

    # `level` is read WITHOUT a fallback and ConformerTorsions takes no **kwargs, so a
    # config that omits it fails here rather than silently running `torsion`
    energy = ConformerTorsions(smiles=prob.smiles, device=str(device),
                               level=prob.level,
                               log_temperature=prob.log_temperature,
                               epsilon=prob.epsilon,
                               min_separation=prob.min_separation,
                               scale_14=prob.scale_14,
                               lj_k_factor=prob.lj_k_factor,
                               include_trivial_rotations=prob.include_trivial_rotations,
                               **{k: getattr(prob, k) for k in
                                  ('delta_r_max', 'delta_theta_max', 'bounding_coeff',
                                   'r_floor', 'theta_floor') if hasattr(prob, k)})
    print(energy.describe())
    dim = energy.data_ndim
    # the resolved level and width go to wandb.summary, not just stdout: a config that
    # says one thing and a run that does another is the failure this guards
    wandb.run.summary.update({'problem/level': energy.level,
                              'problem/data_ndim': dim,
                              'problem/n_atoms': energy.spec.n_atoms,
                              'problem/n_free_r': int((energy._free_block == 0).sum()),
                              'problem/n_free_theta': int((energy._free_block == 1).sum()),
                              'problem/n_free_phi': int((energy._free_block == 2).sum()),
                              'problem/linearity_verified': energy.linearity_verified})

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

    gfn = build_gfn(dim, mdl, device, energy.periodic_dims)
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
            mod.select_batch_size()
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
        chan = "bwd" if use_bwd else "fwd"
        gnorm = torch.nn.utils.clip_grad_norm_(gfn.parameters(),
                                               mod.grad_guard.threshold(chan))
        mod.grad_guard.observe(chan, float(gnorm))
        mod.last_grad_norm_pre_clip = float(gnorm)
        mod.grad_nonfinite += int(not np.isfinite(mod.last_grad_norm_pre_clip))
        # turn-taking: the direction's own policy optimizer, plus the flow (Z) optimizer,
        # exactly as train.py's non-fused branch does
        mod.optimizers[chan].step()
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
    out_path = resolve_out(eval_c.out)
    out_path.write_text(json.dumps(summary, indent=2))
    for k in ("gt_log_z", "gt_e_log_reward", "gt_entropy", "max_entropy", "wall_seconds"):
        wandb.run.summary[k] = summary[k]
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    # GPU pre-flight, same as train.py's. train_conformer was listed in
    # gpu_guard.TRAIN_ENTRYPOINTS -- so OTHER runs correctly saw it as a tenant -- while
    # never checking itself, which is the asymmetry that lets a conformer run launch onto
    # a card a trainer already holds. That is the collision the guard exists to prevent.
    # Override with GFN_GPU_GUARD=0; see gpu_guard.py.
    # `Path` (imported above), NOT os.path -- this module does not import os, so an
    # os.path call here would raise NameError on the one line meant to prevent a
    # collision. Same shape as the sys.exit slip in train.py's guard block: py_compile
    # passes and the safety path is still dead. Compiling is not running.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpu_guard import require_free_gpu, GPUBusy

    try:
        require_free_gpu()
    except GPUBusy as _e:
        raise SystemExit(str(_e))

    main()
