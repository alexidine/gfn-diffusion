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

    python train_conformer.py --yaml configs/conformer_dev.yaml

Config is read by ``get_train_args`` -- the same loader train.py uses, so the
nested-Namespace convention and `auto` resolution carry over unchanged.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from mxtaltools.common.training_utils import flatten_wandb_params

from energies.conformer_torsions import ConformerTorsions
from models.gfn import GFN
from train import safe_histogram
from utils import get_train_args, uniform_discretizer

WANDB_PROJECT = "GFN Conformers"


class TorsionGFN(GFN):
    """GFN whose entire state lives on a torus.

    ``get_periodic_dimensions`` in the base class hard-codes the crystal state layout
    (6 box params, then centroids, then orientations) and raises on any other dimension.
    Torsions are the simpler case -- every dimension wraps -- so the mask is overridden
    rather than the base class being taught about a second layout.
    """

    def get_periodic_dimensions(self, device, do_periodic_angles: bool = True,
                                periodic_centroid_axes=None):
        self.periodic_centroid_axes = ()
        self.ang_mask = torch.ones(self.dim, dtype=torch.bool, device=device)
        self.ang_dim = self.dim
        self.lin_dim = 0
        self.expanded_dim = 2 * self.dim
        self.ang_idx = self.ang_mask.nonzero(as_tuple=False).flatten()
        self.lin_idx = (~self.ang_mask).nonzero(as_tuple=False).flatten()


def build_gfn(dim: int, mdl, device) -> TorsionGFN:
    return TorsionGFN(
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
    ax = torch.linspace(-np.pi, np.pi, grid + 1, dtype=energy.dtype)[:-1]
    pts = torch.cartesian_prod(*([ax] * k)).reshape(-1, k)
    log_r = torch.cat([-energy.energy(pts[i:i + 65536])
                       for i in range(0, len(pts), 65536)])
    log_z = (torch.logsumexp(log_r, 0) + k * np.log(2 * np.pi / grid)).item()
    w = torch.softmax(log_r, 0)
    e_log_r = (w * log_r).sum().item()
    return log_z, e_log_r, log_z - e_log_r, k * np.log(2 * np.pi)


@torch.no_grad()
def evaluate(gfn, energy, discretizer, batch_size, device, log_temp, refs):
    """On-policy eval pass: no exploration, no gradients.

    Separate from the training step on purpose. The fitted log Z reflects whatever
    distribution the trajectories came from, so reading calibration off a training
    batch conflates the model with the sampling scheme. Everything here is measured
    from the policy as it would actually be used.
    """
    init = torch.zeros(batch_size, energy.data_ndim, device=device)
    states, log_pf, log_pb, log_flow = gfn.get_traj_fwd(
        init, discretizer, torch.zeros(batch_size, device=device),
        condition=False, mol_batch=None)
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
    }
    for j in range(energy.data_ndim):
        out[f"dist/torsion_{j}"] = safe_histogram(np.rad2deg(final[:, j].cpu().numpy()))

    if refs is not None:
        gt_log_z, gt_e_log_r, _, h_max = refs
        out["eval/dlog_z"] = out["eval/log_z"] - gt_log_z
        # Z-free: survives past the point where exact log Z does not
        out["eval/dlog_reward"] = out["eval/mean_log_reward"] - gt_e_log_r
        out["eval/entropy_headroom"] = h_max - out["eval/implied_entropy"]
    return out, final


def tb_loss(log_z, log_pf, log_pb, log_reward):
    """Trajectory-balance residual, differenced per step before summing.

    ``log_pf``/``log_pb`` arrive as ``[B, T]``. Summing each and subtracting would form
    an O(1) result from two O(d*T) quantities; differencing first keeps everything O(1).
    """
    delta = (log_pf.double() - log_pb.double()).sum(-1)
    residual = log_z.double() + delta - log_reward.double()
    return residual.pow(2).mean(), residual


def main():
    args = get_train_args()          # same loader train.py uses: --yaml <path>
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
    optimizer = torch.optim.Adam([
        {"params": [p_ for n, p_ in gfn.named_parameters() if "flow_model" not in n],
         "lr": args.lr_policy},
        {"params": [p_ for n, p_ in gfn.named_parameters() if "flow_model" in n],
         "lr": args.lr_flow},
    ])
    discretizer = lambda bsz: uniform_discretizer(bsz, args.integrator.T)

    buffer = None
    buffer_path = Path(train_c.buffer_path) if train_c.buffer_path else None
    if buffer_path is not None and buffer_path.exists():
        blob = torch.load(buffer_path, weights_only=False)
        buffer = torch.as_tensor(blob["states"], dtype=torch.float64)
        print(f"buffer: {len(buffer)} states from {buffer_path}")
    elif buffer_path is not None:
        print(f"buffer {buffer_path} not found; forward-only")

    log_temp = torch.tensor(prob.log_temperature, device=device)
    history, t_start, bwd_seen = [], time.time(), 0.0

    for step in range(train_c.steps):
        optimizer.zero_grad(set_to_none=True)
        use_bwd = buffer is not None and (step % 2 == 1) and np.random.rand() < train_c.bwd_frac

        if use_bwd:
            idx = torch.randint(len(buffer), (train_c.batch_size,))
            terminal = buffer[idx].to(device)
            states, log_pf, log_pb, log_flow = gfn.get_traj_bwd(
                terminal, discretizer, condition=False, mol_batch=None)
            final = terminal
        else:
            init = torch.zeros(train_c.batch_size, dim, device=device)
            # exploration_std is per-sample ([B]), not a scalar -- fwd_get_logvars
            # broadcasts it against the per-dimension log-variance head
            expl = torch.full((train_c.batch_size,), train_c.exploration_std, device=device)
            states, log_pf, log_pb, log_flow = gfn.get_traj_fwd(
                init, discretizer, expl, condition=False, mol_batch=None)
            final = states[:, -1]

        log_reward = -energy.energy(final, None, log_temp)
        log_z = log_flow[:, 0]          # same source train.py uses: log_Z_learned = log_flow[:, 0]
        loss, residual = tb_loss(log_z, log_pf, log_pb, log_reward)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gfn.parameters(), args.gradient_norm_clip)
        optimizer.step()

        bwd_seen += float(use_bwd)
        if step % eval_c.report_every == 0 or step == train_c.steps - 1:
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
                    "dist/residual": safe_histogram(residual.cpu().numpy()),
                }
                ev, _ = evaluate(gfn, energy, discretizer, train_c.batch_size,
                                 device, log_temp, refs)
                metrics.update(ev)
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
