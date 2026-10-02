r"""Per-condition log Z check: importance sampling against exact quadrature and the tracker.

WHAT IT MEASURES. For each condition c of a conformer run, three numbers that claim to be
log Z(c), and one that is:

    IS       log Z_IS = logmeanexp_tau( log R(x) + log P_B(tau | x) - log P_F(tau) ),  N
             rollouts of the EMA policy (or draws from the fitted InternalPrior), with its
             delta-method standard error. CONSISTENT for the true constant WHATEVER THE
             PROPOSAL, provided it covers the target -- a bad policy costs variance, not
             correctness -- so it needs no training and cannot be confounded by
             convergence. That is why an UNTRAINED policy is a legitimate proposal, and
             why the tests use one.
    exact    brute_force_log_z's rule on the MEMBER's own chart, at level `full` only where
             d <= 6 (NH3 and H2CO in QM9). The one number here that is not an estimate.
    tracker  ConditionLogZTracker.ema_logw / ema_log_z_emp, read off the checkpoint.
    head     the learned log Z(c) head, log_flow[:, 0] of the same rollouts.

THE TRACKER IS NOT THE ANSWER, and is never reported as the validated value. `lookup` serves
ema_logw, an EMA of mean log w over the TRACKER'S OWN FEED. Fed by forward rollouts alone that
is the Jensen LOWER BOUND E[log w] <= log Z, with a gap that closes only at the TB fixed
point; a feed that also carries backward rows (update_log_z on a fused stage, the conformer
default) mixes in terms bounded from the other side, and is then a bound on neither.
ema_log_z_emp is a linear EMA of each call's logmeanexp; with one sample per condition per
call (M >= B) it equals ema_logw (ConditionLogZTracker.update's docstring). Measured on a
100-step CPU checkpoint of the six-molecule set under that mixed feed (level full):
ema_log_z_emp read 5.6 and 6.6 nats ABOVE the quadrature on NH3 and H2CO, ema_logw 10.2 and
3.8 below. So the table prints both beside IS as a gap, never as an error. The sample's own
mean log w is printed too: the forward-only value of what ema_logw tracks.

QUADRATURE NEEDS A CONVERGENCE CHECK, AND IT IS REQUIRED. `brute_force_log_z` is a
left-endpoint rule over [-1, 1]^k at ONE isotropic grid, and at d = 6 its convergence is not
monotone: NH3 read -10.556, -10.485, -10.523 at grids 15, 17, 19 against a converged
-10.4892, because the stiff phi column is under-resolved and its error follows where the
nodes fall on the peak, not the spacing (artifacts/conformer_prod_stage0_2026-09-25/
wp09_final/out/logz.json). So the rule here is brute_force_log_z's with a PER-BLOCK count
(r, theta, phi), evaluated on TWO grids; the value is used only when they agree within
`--quad-tol`, and their difference is added to the comparison bar.
  * The default pair is one grid and the SAME grid shifted by half a cell on every axis. On a
    smooth integrand that vanishes at the box edge the leading aliasing error of each axis
    flips sign under that shift, so the disagreement is about twice the error of either --
    measured, H2CO at (16, 8, 32) reads -11.1120 / -11.1136 around the converged -11.1131,
    and the isotropic grid 19 whose error is -0.034 disagrees with its shift by 0.068.
  * A refinement pair (`--quad-grids 12,8,48 16,12,64`) is accepted as well.
  * EVERY BLOCK MUST CHANGE ITS NODES between the two grids, and a pair where one does not is
    refused: that block's error is then identical in both, invisible to the comparison.
    H2CO at (12, 8, 48) and (12, 10, 64) agree to 6e-4 nats while both are 0.022 off,
    because the stiff C=O bond column (r block) was not refined.
The quadrature integrates the box; the reward is defined past it on the r and theta columns
behind the wall, and the IS target is that whole domain. The mass outside was measured at
2.6e-4 nats (NH3) and 2e-7 (H2CO) over lin columns out to 1.5 (same file); `--quad-lin-half`
extends the lin axes by whole cells to include it.

THE INTEGRAND IS CHECKED, NOT ASSUMED. Before any rollout, `integrand_identity`
scores random box states both ways -- the member's -energy (what the rule sums) and the
run's energy_function.log_reward on the carrier-placed rows of that condition's batch (what
the IS scores) -- and refuses a disagreement. Chart methods are called on the MEMBER; the
multi-molecule dispatcher refuses them.

THE ESS FLOOR, ON EVERY ROW. On a poor proposal the weights are heavy-tailed and the
delta-method SE is itself unreliable -- an untrained policy at `full` measured ESS fractions of
1e-4 to 1e-2 on NH3 and H2CO. A row below `--min-ess` or `--min-ess-frac`, or one that dropped
non-finite log w, is reported BELOW_ESS_FLOOR (or NONFINITE_ROWS), never PASS, however close its
estimate happens to land -- and that holds for a row WITHOUT an exact value too, which in QM9 is
every molecule but NH3 and H2CO: there the estimate IS the output, so an unfloored one would be
the production number read as clean. A row without an exact value whose estimate clears the
floor is ESTIMATE_ONLY. Table 2's gaps are left blank on a refused row.

COVERAGE IS WHAT THE FLOOR CANNOT SEE. ESS is a functional of the draws obtained, so a basin
the proposal never visits adds no large weight and no warning -- it biases log Z_IS low. The
fitted prior holds every improper row and every sibling offset tight about the reference, so
unreflected it proposes one side of each non-planar centre, while at level `full` an unlocked
target holds both parities, the mirror image at equal energy. Measured on NH3 (this script,
N = 20000, seed 0) before the prior reflected anything: 0.72 nats below the quadrature at
ESS/N 0.37, against ln 2 = 0.69 for the missing pyramid. Since 2026-09-28
`ConformerTorsions.sample_prior_states` flips each INVERTIBLE centre of
energies/invertible_centres.py (three-coordinate, free under the lock, with a substituent
offset whose sign can be negated as an exact inversion, planar or not) on half its draws and
`prior_log_prob` scores the mixture, which closes that gap. The comparison with quadrature
catches a remaining one; a PAIR DOES NOT -- two conditions whose proposals miss their mirrors
alike carry equal bias, and their difference reads clean. So under `--proposal prior` a row
whose member keeps a free, non-planar centre of that table the prior does not flip
(`prior_coverage_bias`: a four-coordinate one, as at every sp3 centre of an unlocked target,
or one whose lock table is unreadable) is KNOWN_BIASED, the centres named in its notes, never
ESTIMATE_ONLY; a pair on it is KNOWN_BIASED too, and a row with an exact value keeps the
comparison's verdict, which sees the gap. A centre the table leaves out
(energies/invertible_centres.py, NOT COVERED) is not named, and no prior estimate needs it
named: every case listed there is a ring atom, or the planar far atom of a locked double bond,
its own flip. A caged N has no second side; a ring N the prior leaves one-sided, as the root N
of C1COCCN1, has one, but `prior_log_prob` refuses every molecule with a ring block, so on such
a row `run_check` records 'IS not run' and there is no prior estimate to read low. The label
follows the TARGET: once a condition's energy carries the stereo lock (`stereo_coeff` > 0), a
centre its table names holds one parity in the target as well and drops out. The policy
proposal holds no row by construction and is not labelled; its coverage is still what the
floor cannot see.

PAIRS. `--pair A B` names two conditions that must share log Z, an enantiomer pair being the
case in view: their IS difference is compared with the combined SE under the same floor.

PARTITION IDENTITY (`--partition-identity`). Under a stereo lock, sum over the 2^n sign
assignments of Z(s) equals the unlocked Z on any one grid, up to the mass in the lock's band.
`partition_identity` computes that on one grid through a caller's energy builder. The CLI
flag REFUSES (exit 2) and names what is missing, rather than running nothing and printing a
table: while ConformerTorsions takes no `stereo_coeff`, because there is no lock; once it does,
because the lock's per-member sign override is not wired to `partition_identity` here yet.

THE CARD. `train.py::Modeller.__init__` opens a CUDA context on GPU 0 whenever
torch.cuda.is_available(), whatever the configured device. So a CPU check HIDES every GPU
(CUDA_VISIBLE_DEVICES=-1) before anything queries CUDA, and the compute guard then runs for
every device, as conformer_modeller's __main__ runs it: it skips itself when nothing is
visible, and judges the card when something is (a CUDA device, or a process that initialised
CUDA before the variable could hide it).

EXIT STATUS.
  0  some check PASSED, and every other verdict is PASS or ESTIMATE_ONLY;
  1  some verdict is neither: a FAIL, an unconverged quadrature, an estimate refused by the
     floors, or a KNOWN_BIASED one -- and an uncaught error, which prints its traceback;
  2  a refused request: `Refused` (temperature conditioning, a busy card, an unknown
     condition, a grid pair or grid size, an integrand mismatch, a tracker sized for other
     conditions, the prior's preconditions -- T = 1 and level `full` -- the partition-identity
     mode), or an argument argparse rejects;
  3  nothing was checked: every row ESTIMATE_ONLY and no pair named, the case of a ladder
     rung without NH3 or H2CO. The last line of the report says so too.

    python -m eval.conformer_logz_check --config <run.yaml> \
        --checkpoint <ckpt.pt> [--n 10000] [--pair A B] [--json out.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import textwrap
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

#: the width bar of an entry's `planar` mark, which `prior_held_parity_centres` reads: a planar
#: centre is its own flip, so not a parity to label. A WORKING ASSUMPTION defined, with its
#: measurements, in energies/invertible_centres.py; the prior's flip does not read it.
from energies.invertible_centres import MIRROR_WIDTHS

#: Per-block grid order. ConformerTorsions._free_block codes columns 0 r, 1 theta, 2 phi,
#: 3 transverse, 4 a bounded double-bond dihedral; a member with a 3 or a 4 is refused (no
#: block count for either, and no disc wall here).
BLOCKS = ('r', 'theta', 'phi')

#: The default two-grid pair: one grid and itself shifted by half a cell (module docstring).
DEFAULT_GRIDS = ('16,8,32', '16,8,32@0.5')

#: ConformerTorsions keywords the partition-identity mode needs before it can run.
STEREO_SEAM = ('stereo_coeff',)

#: a row with no exact value whose estimate cleared the floors: usable, and checked by nothing
ESTIMATE_ONLY = 'ESTIMATE_ONLY'

#: an estimate that cleared the floors from a proposal KNOWN not to reach part of the target
#: (module docstring, COVERAGE): it reads low, so it is not usable, and a pair on it is no check
KNOWN_BIASED = 'KNOWN_BIASED'

#: every verdict a row or a pair can take. PASS is a check that passed; ESTIMATE_ONLY is no
#: check; every other one fails the exit status
VERDICTS = ('PASS', ESTIMATE_ONLY, 'FAIL', KNOWN_BIASED, 'UNCONVERGED', 'BELOW_ESS_FLOOR',
            'INVALID_PROPOSAL', 'NONFINITE_ROWS', 'NOT_RUN')

#: exit statuses (module docstring)
EXIT_PASS, EXIT_FAIL, EXIT_REFUSED, EXIT_UNCHECKED = 0, 1, 2, 3


class Refused(RuntimeError):
    """A request this check will not answer as asked. main() prints it and exits 2."""


# ------------------------------------------------------------------ statistics

def is_summary(log_w) -> dict:
    """``log Z = logmeanexp(log w)`` with its delta-method SE, the Kish ESS and diagnostics.

    SE^2 = Var(w) / (k mean(w)^2) = 1/ESS - 1/k over the k FINITE rows: the same number as
    sd(w)/(mean(w) sqrt k) with the population variance, written through ESS so the prior path
    (which returns ESS, not weights) gets the identical formula. ess_frac is over all n.

    NON-FINITE ROWS ARE COUNTED, NOT HIDDEN. They are left out of the estimate, as the
    eval's pooled log_Z does, and the count rides in the summary so a verdict can refuse
    the row: a NaN log w is a crashed rollout or reward, and an estimate that silently
    dropped it would read as clean.
    """
    lw = torch.as_tensor(log_w, dtype=torch.float64).detach().cpu().flatten()
    n = int(lw.numel())
    finite = torch.isfinite(lw)
    n_bad = n - int(finite.sum())
    lw = lw[finite]
    out = dict(n=n, n_nonfinite=n_bad, log_z=float('nan'), se=float('nan'),
               ess=0.0, ess_frac=0.0, w_max_frac=float('nan'), mean_log_w=float('nan'))
    if lw.numel() < 2:
        return out
    m = lw.max()
    w = torch.exp(lw - m)
    s1, s2 = w.sum(), (w * w).sum()
    k = int(lw.numel())
    ess = float(s1 * s1 / s2)
    out.update(log_z=float(torch.log(s1 / k) + m), ess=ess, ess_frac=ess / n,
               se=math.sqrt(max(1.0 / ess - 1.0 / k, 0.0)),
               w_max_frac=float(w.max() / s1), mean_log_w=float(lw.mean()))
    return out


# ------------------------------------------------------------------ quadrature

@dataclass(frozen=True)
class GridSpec:
    """Per-block node counts ``(r, theta, phi)`` and a shift in cells (0 = left endpoints)."""
    counts: Tuple[int, int, int]
    shift: float = 0.0

    @classmethod
    def parse(cls, text: str) -> 'GridSpec':
        """``'16,8,32'`` or ``'16,8,32@0.5'``."""
        body, _, shift = str(text).partition('@')
        counts = tuple(int(v) for v in body.split(','))
        if len(counts) != len(BLOCKS) or min(counts) < 1:
            raise ValueError(f'grid {text!r}: need {len(BLOCKS)} positive counts (r, theta, phi)')
        s = float(shift) if shift else 0.0
        if not 0.0 <= s < 1.0:
            raise ValueError(f'grid {text!r}: the shift is a fraction of a cell in [0, 1)')
        return cls(counts, s)

    def __str__(self):
        return ','.join(map(str, self.counts)) + (f'@{self.shift:g}' if self.shift else '')


def _blocks_of(member) -> np.ndarray:
    blocks = np.asarray(member._free_block).reshape(-1)
    if (blocks > 2).any():
        raise NotImplementedError(
            'this member carries transverse columns (block 3) or bounded double-bond '
            'dihedrals (block 4, double_bond_box_deg): there is no block count for them '
            'here, and the box rule would ignore a transverse disc wall')
    return blocks


def _axes(member, spec: GridSpec, lin_half: float):
    axes, log_cell = [], 0.0
    for b in _blocks_of(member):
        g = spec.counts[int(b)]
        h = 2.0 / g
        log_cell += math.log(h)
        idx = torch.arange(g, dtype=torch.float64)
        if int(b) != 2 and lin_half > 1.0:
            # WHOLE CELLS past each edge, so the wide axis is a superset of the box axis and
            # the difference is outside mass alone, not a change of grid phase
            m = int(math.ceil((lin_half - 1.0) / h))
            idx = torch.arange(-m, g + m, dtype=torch.float64)
        axes.append(-1.0 + h * (idx + spec.shift))
    return axes, log_cell


@torch.no_grad()
def grid_log_z(member, spec: GridSpec, lin_half: float = 1.0, chunk: int = 1 << 16,
               max_points: float = 5e7) -> Tuple[float, int]:
    """``ConformerTorsions.brute_force_log_z``'s rule with a per-block node count.

    `spec.counts == (g, g, g)` with no shift and `lin_half` 1 reproduces brute_force_log_z at
    `grid=g` (tests/conformer/test_logz_check.py pins it). phi columns always cover one
    period; lin columns cover the box, or past it with `lin_half` > 1. Sums ``-energy``, the
    member's log reward at its own temperature, in float64 over chunks of `chunk` nodes.
    """
    axes, log_cell = _axes(member, spec, lin_half)
    sizes = [len(a) for a in axes]
    total = refuse_over_budget(member, spec, lin_half, max_points)
    axes = [a.to(dtype=member.dtype, device=member.device) for a in axes]
    acc = []
    for s in range(0, total, chunk):
        rem = torch.arange(s, min(s + chunk, total), device=member.device)
        cols = []
        for j in reversed(range(len(axes))):
            cols.append(axes[j][rem % sizes[j]])
            rem = rem // sizes[j]
        pts = torch.stack(cols[::-1], dim=1)
        acc.append(torch.logsumexp((-member.energy(pts)).double(), 0))
    return float(torch.logsumexp(torch.stack(acc), 0)) + log_cell, total


def refuse_over_budget(member, spec: GridSpec, lin_half: float, max_points: float) -> int:
    """The node count of `spec` on this member, refused over `max_points`."""
    total = int(np.prod([len(a) for a in _axes(member, spec, lin_half)[0]]))
    if total > max_points:
        raise Refused(f'grid {spec} is {total:,} nodes for this member, over the '
                      f'{max_points:.3g} budget (--quad-max-points)')
    return total


def refuse_unrefined(member, grid_a: GridSpec, grid_b: GridSpec) -> None:
    """Refuse a pair in which some block present in this member keeps the same nodes: that
    block's quadrature error is then common to both values and cannot show in their
    difference (module docstring, the H2CO case)."""
    blocks = sorted({int(b) for b in _blocks_of(member)})
    same = [BLOCKS[b] for b in blocks
            if (grid_a.counts[b], grid_a.shift) == (grid_b.counts[b], grid_b.shift)]
    if same:
        raise Refused(
            f'grids {grid_a} and {grid_b} put the same nodes on the {same} block(s); an '
            f'error in that block is common to both values and invisible to their difference. '
            f'Refine every block, or shift the second grid (e.g. {grid_a}@0.5)')


def two_grid_log_z(member, grid_a: GridSpec, grid_b: GridSpec, tol: float,
                   lin_half: float = 1.0, max_points: float = 5e7) -> dict:
    """Both grids, their difference and whether they agree within `tol` nats
    (`refuse_unrefined` first)."""
    refuse_unrefined(member, grid_a, grid_b)
    t0 = time.perf_counter()
    za, na = grid_log_z(member, grid_a, lin_half, max_points=max_points)
    zb, nb = grid_log_z(member, grid_b, lin_half, max_points=max_points)
    diff = abs(za - zb)
    return dict(log_z=za, log_z_b=zb, diff=diff, converged=bool(diff <= tol), tol=tol,
                grid_a=str(grid_a), grid_b=str(grid_b), points=na + nb, lin_half=lin_half,
                wall_s=time.perf_counter() - t0)


# ------------------------------------------------------------------ the run's objects

def _member(energy_function, ident: str):
    """The single-chart energy for `ident`: a set's member, or the energy itself."""
    members = getattr(energy_function, '_members', None)
    if members is None:
        return energy_function
    if ident not in members:
        raise KeyError(f'the energy holds no member {ident!r}')
    return members[ident]


def _place(energy_function, ident: str, x: torch.Tensor) -> torch.Tensor:
    """Member-width state -> the run's state width (carrier placement, or unchanged)."""
    carrier = getattr(energy_function, 'carrier', None)
    return x if carrier is None else carrier.to_carrier(ident, x)


def _condition_rows(modeller) -> Dict[str, Tuple[object, int]]:
    """``{identifier: (dataset, row)}`` over mol_dataset, then the held-out set if loaded."""
    out = {}
    for ds in (modeller.mol_dataset, getattr(modeller, 'test_mol_dataset', None)):
        idents = getattr(getattr(ds, 'batch', None), 'identifier', None) if ds else None
        for row, ident in enumerate(idents or []):
            out.setdefault(ident, (ds, row))
    return out


def _condition_batch(dataset, row: int, n: int, device):
    """`n` copies of one condition graph, the way the eval draws them."""
    mb = dataset.sample_graphs_at([row], repeats=n).to(device)
    # fwd_eval_sampling orients every eval batch before rolling out; mirrored, so the
    # rows scored here are the rows the eval scores
    mb.orient_molecule(mode='standard')
    return mb


@torch.no_grad()
def integrand_identity(energy_function, ident: str, dataset, row: int, n: int = 256,
                       seed: int = 0) -> float:
    """max |(-member.energy(x)) - energy_function.log_reward(place(x))| / (1 + |log R|).

    The first is what the quadrature sums, the second what the IS scores: the run's
    dispatcher on the carrier-placed rows of this condition's own batch, at the run's
    temperature. Uniform box states, so the wall and the stiff columns are exercised.
    """
    member = _member(energy_function, ident)
    g = torch.Generator().manual_seed(int(seed))
    x = (torch.rand(n, member.data_ndim, generator=g, dtype=torch.float64) * 2 - 1)
    x = x.to(dtype=member.dtype, device=member.device)
    a = -member.energy(x)
    mb = _condition_batch(dataset, row, n, member.device)
    log_t = torch.full((n,), float(energy_function.log_temperature), device=member.device)
    b = energy_function.log_reward(_place(energy_function, ident, x), mb, log_t)
    return float(((a.double() - b.double()).abs() / (1.0 + a.double().abs())).max())


def check_integrand(energy_function, ident: str, dataset, row: int, tol: float,
                    seed: int = 0) -> float:
    """`integrand_identity`, refused above `tol`: the quadrature would then integrate a
    different target from the one the IS scores, and their gap would read as an IS error."""
    rel = integrand_identity(energy_function, ident, dataset, row, seed=seed)
    if not rel <= tol:                      # a NaN refuses too
        raise Refused(f'{ident}: the quadrature integrand (-member.energy) and the IS reward '
                      f'(energy_function.log_reward on the carrier rows) differ by {rel:.3g} '
                      f'relative, over {tol:g}; they are not one target')
    return rel


def _seed_for(seed: int, ident: str) -> int:
    """Per-condition seed: reproducible whatever subset or order the conditions run in."""
    return (int(seed) * 1_000_003 + zlib.crc32(ident.encode())) % (2 ** 31)


@torch.no_grad()
def policy_rollouts(modeller, ident: str, dataset, row: int, n: int, batch: int,
                    seed: int) -> dict:
    """N forward rollouts of the EMA policy on one condition: log w, the head, the id.

    Through `eval/utils.py::sample_eval_fwd_trajs`, the function fwd_eval_sampling uses, so
    each log w is built as the eval's pooled log_Z builds it -- but on `integrator.T` steps,
    the training discretisation, where the eval uses `eval_T`. The policy learned its drift
    per step at the training dt, and a different step schedule is a different SDE
    (utils.py::get_discretizer), so integrator.T is the proposal the policy was trained as.
    The two are equal in every conformer config today; `run_check` records both and notes a
    difference.
    """
    from eval.utils import sample_eval_fwd_trajs
    from train import get_discretizer
    from utils import get_gfn_init_state

    ef, model = modeller.energy_function, modeller.ema_model
    disc = get_discretizer(modeller.args.integrator)
    torch.manual_seed(_seed_for(seed, ident))
    log_w, head, cids = [], [], []
    done = 0
    while done < n:
        b = min(batch, n - done)
        mb = _condition_batch(dataset, row, b, modeller.device)
        init = get_gfn_init_state(b, ef.data_ndim, modeller.device)
        temps = ef.temperature * torch.ones(b, dtype=torch.float32, device=modeller.device)
        out = sample_eval_fwd_trajs(init, model, disc, ef, mb, temperatures=temps)
        log_w.append(out['log_r'] + out['log_pbs'].sum(-1) - out['log_pfs'].sum(-1))
        head.append(out['log_flow'][:, 0])
        cids.append(out['condition_id'])
        done += b
    cid = torch.unique(torch.cat(cids))
    if cid.numel() != 1:
        raise RuntimeError(f'{ident}: rollouts carried condition_id {cid.tolist()}; one '
                           f'condition batch must map to one id')
    head = torch.cat(head).double()
    return dict(log_w=torch.cat(log_w), head=float(head.mean()),
                head_spread=float(head.max() - head.min()), condition_id=int(cid))


def prior_estimate(modeller, ident: str, n: int, seed: int) -> dict:
    """IS with the fitted InternalPrior as proposal (energies/prior_diagnostics.py::is_log_z).

    That function returns ESS rather than weights, so SE comes from `is_summary`'s identity
    SE^2 = 1/ESS - 1/n. Ring blocks have no prior density and raise NotImplementedError
    there; the row is then NOT_RUN with the reason. Level `full` only, and even there the
    estimate misses every parity the prior holds (`prior_coverage_bias`).
    """
    from energies.prior_diagnostics import is_log_z

    prior = getattr(modeller, 'internal_prior', None)
    if prior is None:
        raise Refused('--proposal prior needs energy_config.internal_prior_path')
    if abs(float(modeller.energy_function.temperature) - 1.0) > 1e-12:
        # is_log_z scores potential + Jacobian at T = 1, while the quadrature and the
        # policy path score at the run's T: a different target, not a noisier estimate
        raise Refused(f'--proposal prior scores at T = 1 (prior_diagnostics.is_log_z); '
                      f'this run samples at T = {modeller.energy_function.temperature:g}')
    member = _member(modeller.energy_function, ident)
    if member.level != 'full':
        # is_log_z weighs the prior's density over EVERY coordinate it draws, while a lower
        # level's state keeps its free ones and discards the rest: the weights then carry
        # the density of discarded draws. Measured at level dihedral against the grid (one
        # grid of 16, N = 20000, seed 0): H2CO 4.83 and NH3 5.52 nats low, at ESS 9 and 8
        raise Refused(f'--proposal prior needs level full: prior_diagnostics.is_log_z weighs '
                      f'every coordinate the prior draws, and this run is at level '
                      f'{member.level}, whose state discards the frozen ones')
    r = is_log_z(member, prior, n=n, seed=_seed_for(seed, ident))
    ess, k = float(r['ess']), int(r['n'])
    return dict(n=k, n_nonfinite=0, log_z=float(r['log_z']), ess=ess, ess_frac=ess / k,
                se=math.sqrt(max(1.0 / ess - 1.0 / k, 0.0)) if ess > 0 else float('nan'),
                w_max_frac=float(r['w_max_frac']), mean_log_w=float('nan'),
                clip_frac=float(r['clip_frac']))


def prior_held_parity_centres(member) -> List[int]:
    """Placement slots of the non-planar centres whose PARITY the fitted prior never flips.

    `ConformerTorsions.sample_prior_states` -- and `prior_log_prob`, which scores exactly its
    draws -- holds two kinds of phi row tight about the reference: an improper row about its
    ph0 at `improper_phi_sigma`, and each substituent of a group (`torsion_groups`) at its
    reference offset from the group's leader or ring frame, at the group's
    `sibling_jitter_sigma`. The flip takes an offset v to -v, |wrap(2 v)| away. The centres
    considered are energies/invertible_centres.py's `centre_table`: those whose offset can be
    negated as an exact inversion without turning their ring system. The prior flips every
    INVERTIBLE centre of that table (three-coordinate and free under the lock, planar or not)
    on half its draws. The centres returned are the rest, less the `planar` ones: a centre
    whose every flipped row sits within `MIRROR_WIDTHS` of its width of its flip holds v near
    0 or pi, its own flip, so not a parity. That leaves the centres the lock names, every
    four-coordinate one, and free ones whose lock table could not be read.
    """
    from energies.invertible_centres import centre_table
    return sorted(c.slot for c in centre_table(member) if not c.invertible and not c.planar)


def prior_coverage_bias(member) -> Optional[str]:
    """Why IS with the fitted prior as proposal reads LOW on this member, or None.

    The target at level `full` holds both parities of every non-planar centre nothing locks.
    The prior proposes both sides of every invertible centre of energies/invertible_centres.py
    and one side of the rest (`prior_held_parity_centres`). With the stereo lock on
    (`stereo_coeff` > 0) a centre the lock's table names holds one parity in the target too and
    drops out: one whose `Centre.lock` is LOCKED, the table holding a tetrahedral element keyed
    on it (`stereo.kind`, `stereo.key`, placement slots; energies/invertible_centres.py, LOCK
    AND SCOPE), the reading the prior's flip uses too. What stays is a free, non-planar centre
    of the table the prior does not flip: a four-coordinate one (every sp3 centre of an
    unlocked target: the prior flips three-coordinate centres only). The estimate omits whatever
    mass the target holds on its other side, and the label names each centre with its kind. A
    locked member whose table cannot be read here drops NOTHING, and `invertible_centres` names
    none of its centres: a renamed interface keeps the label rather than lifting it.
    """
    from energies.invertible_centres import FREE, LOCKED, centre_table

    by_slot = {c.slot: c for c in centre_table(member)}
    centres = [s for s in prior_held_parity_centres(member) if by_slot[s].lock != LOCKED]
    if not centres:
        return None
    target = 'the unlocked target holds both'
    if float(getattr(member, 'stereo_coeff', 0.0) or 0.0) > 0.0:
        target = ('the lock leaves them free' if all(by_slot[s].lock == FREE for s in centres)
                  else "the lock's table (stereo.key, stereo.kind) is unreadable here, so none "
                       "counts as locked")

    def kind(slot):
        c = by_slot[slot]
        if c.lock != FREE:
            return ' (lock state unread)'
        return f' ({c.n_bonded}-coordinate: the prior flips three-coordinate centres only)'

    from rdkit import Chem
    table, z = Chem.GetPeriodicTable(), np.asarray(member.spec.z)
    names = ', '.join(f'{table.GetElementSymbol(int(z[c]))}{c}{kind(c)}' for c in centres)
    return (f'{KNOWN_BIASED}: the fitted prior proposes one parity at {names} (placement '
            f'slots) and {target}, so the estimate reads LOW by the target\'s mass on each '
            f'unproposed side (ln 2 where the two sides hold equal mass)')


def tracker_reading(state: Optional[dict], condition_id: Optional[int]) -> dict:
    """One condition's tracker values from a ConditionLogZTracker.state_dict()."""
    if state is None or condition_id is None:
        return dict(ema_logw=float('nan'), ema_log_z_emp=float('nan'), visits=None,
                    trusted=None)
    cid = int(condition_id)
    if not 0 <= cid < int(state['library_size']):
        raise IndexError(f'condition_id {cid} outside the tracker library '
                         f'({state["library_size"]})')
    logw = float(state['ema_logw'][cid])
    visits = int(state['count'][cid])
    # the same mask ConditionLogZTracker.lookup applies
    return dict(ema_logw=logw, ema_log_z_emp=float(state['ema_log_z_emp'][cid]),
                visits=visits, trusted=bool(visits >= int(state['min_visits'])
                                            and math.isfinite(logw)))


# ------------------------------------------------------------------ verdicts

@dataclass(frozen=True)
class Floors:
    k_se: float = 3.0
    abs_tol: float = 0.0
    min_ess: float = 100.0
    min_ess_frac: float = 1e-3
    max_clip_frac: float = 1e-3


def _is_problem(est: Optional[dict], fl: Floors) -> Optional[str]:
    """Why an IS estimate cannot carry a verdict, or None."""
    if est is None:
        return 'NOT_RUN'
    if est.get('n_nonfinite', 0):
        return 'NONFINITE_ROWS'
    if not math.isfinite(est.get('log_z', float('nan'))):
        return 'NOT_RUN'
    if est.get('clip_frac', 0.0) > fl.max_clip_frac:
        # the box clamp puts finite mass ON the wall, which no density can express
        return 'INVALID_PROPOSAL'
    if est['ess'] < fl.min_ess or est['ess_frac'] < fl.min_ess_frac:
        return 'BELOW_ESS_FLOOR'
    return None


def exact_verdict(est: Optional[dict], quad: Optional[dict], fl: Floors) -> Tuple[str, float]:
    """(verdict, bar) for IS against quadrature. bar = max(k SE + |A - B|, abs_tol)."""
    problem = _is_problem(est, fl)
    if problem:
        return problem, float('nan')
    if not quad or not quad.get('converged'):
        return 'UNCONVERGED', float('nan')
    bar = max(fl.k_se * est['se'] + quad['diff'], fl.abs_tol)
    return ('PASS' if abs(est['log_z'] - quad['log_z']) <= bar else 'FAIL'), bar


def pair_verdict(a: dict, b: dict, fl: Floors) -> dict:
    """Two conditions that must share log Z: |IS_a - IS_b| against k sqrt(SE_a^2 + SE_b^2)."""
    ea, eb = a.get('is'), b.get('is')
    out = dict(a=a['condition'], b=b['condition'], delta=float('nan'), se=float('nan'),
               bar=float('nan'), z=float('nan'),
               tracker_delta=a['tracker']['ema_logw'] - b['tracker']['ema_logw'],
               exact_delta=((a['exact'] or {}).get('log_z', float('nan'))
                            - (b['exact'] or {}).get('log_z', float('nan'))))
    problem = _is_problem(ea, fl) or _is_problem(eb, fl)
    if problem:
        out['verdict'] = problem
        return out
    d, se = ea['log_z'] - eb['log_z'], math.hypot(ea['se'], eb['se'])
    bar = max(fl.k_se * se, fl.abs_tol)
    # the delta is shown, but a known-biased side never passes: two proposals that miss their
    # mirrors alike carry equal bias, and their difference reads clean (module docstring)
    biased = a.get('bias') or b.get('bias')
    out.update(delta=d, se=se, bar=bar, z=abs(d) / se if se > 0 else float('inf'),
               verdict=KNOWN_BIASED if biased else ('PASS' if abs(d) <= bar else 'FAIL'))
    return out


# ------------------------------------------------------------------ partition identity

def stereo_seam_missing() -> List[str]:
    """The `STEREO_SEAM` keywords ConformerTorsions does not take yet."""
    import inspect

    from energies.conformer_torsions import ConformerTorsions
    params = inspect.signature(ConformerTorsions.__init__).parameters
    return [k for k in STEREO_SEAM if k not in params]


def partition_identity(energy_for_signs: Callable, n_elements: int, spec: GridSpec,
                       lin_half: float = 1.0, max_points: float = 5e7) -> dict:
    """log sum_s Z(s) against log Z(unlocked), every term on the SAME grid.

    `energy_for_signs(None)` is the unlocked energy and `energy_for_signs(s)` the energy
    locked to the sign assignment `s` (a tuple of +-1, one per stereo element). On one grid
    the identity holds node by node wherever the locks partition the state space, so the
    residual measures the lock's band mass plus roundoff, not quadrature error.
    """
    if n_elements < 1:
        raise ValueError('a partition needs at least one stereo element')
    z_u, pts = grid_log_z(energy_for_signs(None), spec, lin_half, max_points=max_points)
    per = {}
    for signs in itertools.product((1, -1), repeat=int(n_elements)):
        per[signs] = grid_log_z(energy_for_signs(signs), spec, lin_half,
                                max_points=max_points)[0]
    z_sum = float(torch.logsumexp(torch.tensor(list(per.values()), dtype=torch.float64), 0))
    return dict(log_z_unlocked=z_u, log_z_sum=z_sum, residual=z_sum - z_u,
                per_assignment={''.join('+' if s > 0 else '-' for s in k): v
                                for k, v in per.items()},
                grid=str(spec), points=pts * (1 + len(per)))


# ------------------------------------------------------------------ the check

def gpu_preflight(device: str, config_path: Optional[str] = None) -> None:
    """The compute guard, for every device (module docstring, THE CARD).

    A CPU device HIDES every GPU first, because `train.py::Modeller.__init__` opens a context
    on GPU 0 whenever torch.cuda.is_available(), whatever the device; the variable is read
    once, at the runtime's first query, so it is set before that query. If the query still
    finds a card, CUDA was initialised in this process before the variable could hide it:
    the variable is put back and that card is judged like any launch's.
    """
    from gpu_guard import GPUBusy, require_free_gpu

    if not str(device).startswith('cuda'):
        before = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if torch.cuda.is_available():
            if before is None:
                del os.environ['CUDA_VISIBLE_DEVICES']
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = before
    try:
        require_free_gpu(config_path=config_path)       # skips itself when nothing is visible
    except GPUBusy as e:
        raise Refused(str(e))


def build_modeller(config: str, checkpoint: Optional[str] = None, device: str = 'cpu'):
    """A ConformerModeller from the run's own config, pointed at `checkpoint` read-only.

    Through the config loader the trainer uses (utils.get_train_args minus argparse). With a
    checkpoint the run name is suffixed and cfg:buffers.fresh_on_switch set, so load_full
    takes the other-run branch and does NOT read the buffer sidecar -- at the upper rungs a
    sidecar is gigabytes, and nothing here reads a buffer. checkpoint_read_only makes every
    save a no-op.
    """
    from conformer_modeller import ConformerModeller
    from utils import dict2namespace, load_yaml, preflight_config, resolve_derived_config

    cfg = load_yaml(config)
    cfg.update(device=device, buffer_device=device, compile_policy=False,
               checkpoint_read_only=True, continue_from_checkpoint=False,
               load_weights_only=False)
    if checkpoint:
        ck = Path(checkpoint).resolve()
        if not ck.exists():
            raise FileNotFoundError(ck)
        cfg.update(checkpoints_dir=ck.parent.as_posix(), checkpoint_name=ck.name,
                   run_name=f"{cfg.get('run_name', 'run')}_logz_check")
        cfg.setdefault('buffers', {})['fresh_on_switch'] = True
    else:
        cfg['checkpoint_name'] = None
    args = resolve_derived_config(preflight_config(dict2namespace(cfg)))
    return ConformerModeller(args=args)


def init_modeller(m, policy: bool = True):
    """train()'s init sequence, as far as the registry: the energy, the model (a checkpoint
    load when one is named), the condition sets, then init_identifiers, whose sorted
    registry over every dataset is what condition_id -- and so the tracker row -- is."""
    if bool(getattr(m.args, 'temperature_conditioning', False)):
        raise Refused('temperature_conditioning is on: log Z(c) is then a function of T and '
                      'this check integrates at one fixed temperature')
    m.init_energy_function()
    # the prior proposal needs no model; a named checkpoint still loads one, for the tracker
    if policy or getattr(m.args, 'checkpoint_name', None):
        m.init_gfn()
    m.init_mol_dataset()
    m.init_prior_dataset()
    m.init_identifiers()
    return m


def run_check(m, conditions: Optional[Sequence[str]] = None, proposal: str = 'policy',
              n: int = 10_000, batch: int = 4096, seed: int = 0,
              grids: Sequence[str] = DEFAULT_GRIDS, quad_tol: float = 0.02,
              lin_half: float = 1.0, max_quad_dim: int = 6, max_points: float = 5e7,
              pairs: Sequence[Tuple[str, str]] = (), floors: Floors = Floors(),
              identity_tol: float = 1e-4) -> dict:
    """Every number of the report for an initialised modeller (`init_modeller`)."""
    ef = m.energy_function
    rows_of = _condition_rows(m)
    conditions = list(conditions) if conditions else list(dict.fromkeys(rows_of))
    unknown = [c for c in conditions if c not in rows_of]
    for a, b in pairs:
        unknown += [c for c in (a, b) if c not in rows_of]
    if unknown:
        raise Refused(f'no condition row for {sorted(set(unknown))}; the run holds '
                      f'{list(rows_of)}')
    for a, b in pairs:
        conditions += [c for c in (a, b) if c not in conditions]
    ga, gb = (GridSpec.parse(g) for g in grids)
    identity = {}
    for ident in conditions:
        # the grid pair, its size and the integrand are judged BEFORE any rollout, not after
        # an hour of them
        member = _member(ef, ident)
        if int(member.data_ndim) > max_quad_dim:
            continue
        try:
            refuse_unrefined(member, ga, gb)
            for g in (ga, gb):
                refuse_over_budget(member, g, lin_half, max_points)
        except NotImplementedError:                 # transverse: no quadrature anyway
            continue
        ds, row = rows_of[ident]
        identity[ident] = check_integrand(ef, ident, ds, row, identity_tol, seed=seed)
    tracker = (m.condition_log_z.state_dict() if hasattr(m, 'condition_log_z') else None)
    if tracker is not None and int(tracker['library_size']) != int(ef.condition_library_size):
        # ConformerModeller.init_condition_log_z's refusal: a table sized for another
        # condition set indexes its rows against the wrong molecules
        raise Refused(f"the checkpoint's tracker holds {tracker['library_size']} "
                      f"conditions and this run's energy {ef.condition_library_size}")

    rows = []
    for ident in conditions:
        ds, row = rows_of[ident]
        member = _member(ef, ident)
        rec = dict(condition=ident, level=member.level, k=int(member.data_ndim), exact=None,
                   head=float('nan'), head_spread=float('nan'), condition_id=None,
                   identity=identity.get(ident), notes=[])
        t0 = time.perf_counter()
        try:
            if proposal == 'policy':
                r = policy_rollouts(m, ident, ds, row, n, batch, seed)
                est = is_summary(r['log_w'])
                rec.update(head=r['head'], head_spread=r['head_spread'],
                           condition_id=r['condition_id'])
            else:
                est = prior_estimate(m, ident, n, seed)
                mb = _condition_batch(ds, row, 2, m.device)
                rec['condition_id'] = int(ef.condition_samples(mb)[3][0])
        except NotImplementedError as e:            # the prior has no density here
            est = None
            rec['notes'].append(f'IS not run: {str(e).splitlines()[0]}')
        rec['is_wall_s'] = time.perf_counter() - t0
        rec['is'] = est
        rec['tracker'] = tracker_reading(tracker, rec['condition_id'])

        quad = rec['k'] <= max_quad_dim
        if not quad:
            rec['notes'].append(f'k = {rec["k"]} > {max_quad_dim}: no quadrature')
        else:
            try:
                rec['exact'] = two_grid_log_z(member, ga, gb, quad_tol, lin_half, max_points)
                if not rec['exact']['converged']:
                    # said even when the IS row fails first, so both defects are visible
                    rec['notes'].append(f"quadrature UNCONVERGED: |A-B| = "
                                        f"{rec['exact']['diff']:.3g} > {quad_tol:g} nats")
            except NotImplementedError as e:        # transverse columns
                quad = False
                rec['notes'].append(f'no quadrature: {str(e).splitlines()[0]}')
        # the floors judge EVERY row: without an exact value the estimate is the output
        rec['is_status'] = _is_problem(est, floors) or 'ok'
        # and a coverage gap the floors cannot see, where the proposal is known to have one
        rec['bias'] = (prior_coverage_bias(member) if proposal == 'prior' and est is not None
                       else None)
        if rec['bias']:
            rec['notes'].append(rec['bias'])
            if rec['is_status'] == 'ok':
                rec['is_status'] = KNOWN_BIASED
        if quad:
            rec['verdict'], rec['bar'] = exact_verdict(est, rec['exact'], floors)
        else:
            rec['verdict'] = ESTIMATE_ONLY if rec['is_status'] == 'ok' else rec['is_status']
            rec['bar'] = float('nan')
        rows.append(rec)

    by = {r['condition']: r for r in rows}
    pair_rows = [pair_verdict(by[a], by[b], floors) for a, b in pairs]
    integ = getattr(m.args, 'integrator', None)
    return dict(meta=dict(proposal=proposal, n=int(n), batch=int(batch), seed=int(seed),
                          steps=getattr(integ, 'T', None),
                          eval_T=getattr(m.args, 'eval_T', None),
                          grids=[str(ga), str(gb)], quad_tol=quad_tol, lin_half=lin_half,
                          max_quad_dim=max_quad_dim, floors=floors.__dict__,
                          temperature=float(ef.temperature),
                          checkpoint=getattr(m.args, 'checkpoint_name', None),
                          step=int(getattr(m, 'step_ind', 0) or 0),
                          stage=getattr(m, 'stage', None),
                          tracker=tracker is not None),
                conditions=rows, pairs=pair_rows)


# ------------------------------------------------------------------ report

def _f(v, fmt='{:+.4f}'):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '-'
    return fmt.format(v)


def _table(title: str, caption: str, header: Sequence[str], body: Iterable[Sequence[str]]):
    body = [list(map(str, r)) for r in body]
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(header)]
    line = lambda cells: '  '.join(c.rjust(w) if i else c.ljust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    return '\n'.join([title, textwrap.fill(caption, 110), line(header),
                      line(['-' * w for w in widths]), *map(line, body)])


def verdict_counts(result: dict) -> Dict[str, int]:
    """Verdicts by kind, over every row and every pair."""
    got = [r['verdict'] for r in result['conditions']] + [p['verdict'] for p in result['pairs']]
    return {v: got.count(v) for v in VERDICTS if got.count(v)}


def _gap(r: dict, value: float) -> float:
    """A number derived from the IS estimate, blanked (NaN) on a row the floors refused or
    one KNOWN_BIASED."""
    return value if r.get('is_status') == 'ok' else float('nan')


def format_report(result: dict) -> str:
    meta, rows = result['meta'], result['conditions']
    if meta['checkpoint']:
        who = f"checkpoint {meta['checkpoint']} at step {meta['step']} (stage {meta['stage']})"
    else:
        who = ('an UNTRAINED policy (no checkpoint)' if meta['proposal'] == 'policy'
               else 'no checkpoint')
    prop = ('EMA-policy rollouts' if meta['proposal'] == 'policy'
            else 'draws from the fitted InternalPrior')
    levels = sorted({r['level'] for r in rows})
    fl = meta['floors']
    shift = GridSpec.parse(meta['grids'][0]).shift
    rule = ('the left-endpoint rule' if not shift
            else f'the rule on nodes {shift:g} cell right of the left endpoints')
    out = [_table(
        'Table 1. Per-condition log Z: importance sampling against exact quadrature.',
        f"IS from N = {meta['n']} {prop} per condition, seed {meta['seed']}, {who}; "
        f"level {'/'.join(levels)}, T = {meta['temperature']:g} kcal/mol"
        + (f", {meta['steps']} rollout steps (integrator.T)" if meta['proposal'] == 'policy'
           else '') +
        f". Exact = {rule} on grid {meta['grids'][0]}, used only when grid "
        f"{meta['grids'][1]} agrees within {meta['quad_tol']:g} nats (|A-B|); a condition with "
        f"k > {meta['max_quad_dim']} free coordinates has none. EVERY row needs ESS >= "
        f"{fl['min_ess']:g} with ESS/N >= {fl['min_ess_frac']:g} and no non-finite log w, or "
        f"carries that floor as its verdict; a row without an exact value that clears them is "
        f"{ESTIMATE_ONLY}, or {KNOWN_BIASED} where the proposal is known to miss part of the "
        f"target (Notes name it). "
        f"PASS needs |IS - exact| <= max({fl['k_se']:g} SE + |A-B|, {fl['abs_tol']:g}); "
        f"compare IS - exact with the bar.",
        ['condition', 'level', 'k (free coords)', 'IS log Z (nats)', 'SE (nats)',
         'ESS (draws)', 'ESS/N',
         'non-finite log w (rows)', 'exact log Z (nats)', '|A-B| (nats)', 'IS - exact (nats)',
         'bar (nats)', 'verdict'],
        [[r['condition'], r['level'], r['k'], _f((r['is'] or {}).get('log_z')),
          _f((r['is'] or {}).get('se'), '{:.4f}'), _f((r['is'] or {}).get('ess'), '{:.1f}'),
          _f((r['is'] or {}).get('ess_frac'), '{:.2e}'),
          _f((r['is'] or {}).get('n_nonfinite'), '{:d}'), _f((r['exact'] or {}).get('log_z')),
          _f((r['exact'] or {}).get('diff'), '{:.1e}'),
          _f((r['is'] or {}).get('log_z', float('nan'))
             - (r['exact'] or {}).get('log_z', float('nan'))),
          _f(r['bar'], '{:.4f}'), r['verdict']] for r in rows])]
    out.append(_table(
        'Table 2. The same conditions against the tracker and the learned head.',
        f"Tracker = ConditionLogZTracker read from the checkpoint "
        f"({'present' if meta['tracker'] else 'ABSENT: no checkpoint or no tracker in it'}). "
        f"ema_logw is the mean log w of the tracker's own feed: a Jensen LOWER bound on log Z "
        f"from forward rollouts alone, a bound on neither side once backward rows feed it "
        f"too. ema_log_z_emp equals it at one sample per condition per update. Neither is a "
        f"validated log Z; IS - ema_logw is a gap, not an error. mean log w is this "
        f"sample's own forward E[log w]. head = log_flow[:, 0] of the same rollouts. IS "
        f"status is Table 1's floors, or {KNOWN_BIASED}; the two gaps are blank on a row "
        f"that is not ok. visits = the tracker's lifetime sample count, trusted = its lookup "
        f"mask. Same N, seed and step as Table 1.",
        ['condition', 'IS log Z (nats)', 'IS status', 'mean log w (nats)', 'ema_logw (nats)',
         'ema_log_z_emp (nats)', 'visits (samples)', 'trusted (lookup mask)',
         'IS - ema_logw (nats)', 'head (nats)',
         'head - IS (nats)'],
        [[r['condition'], _f((r['is'] or {}).get('log_z')), r['is_status'],
          _f((r['is'] or {}).get('mean_log_w')), _f(r['tracker']['ema_logw']),
          _f(r['tracker']['ema_log_z_emp']),
          '-' if r['tracker']['visits'] is None else r['tracker']['visits'],
          '-' if r['tracker']['trusted'] is None else r['tracker']['trusted'],
          _f(_gap(r, (r['is'] or {}).get('log_z', float('nan')) - r['tracker']['ema_logw'])),
          _f(r['head']),
          _f(_gap(r, r['head'] - (r['is'] or {}).get('log_z', float('nan'))))]
         for r in rows]))
    if result['pairs']:
        out.append(_table(
            'Table 3. Condition pairs that must share log Z.',
            f"delta = IS_a - IS_b from Table 1's estimates; SE = sqrt(SE_a^2 + SE_b^2); PASS "
            f"needs |delta| <= max({fl['k_se']:g} SE, {fl['abs_tol']:g}) with both rows above "
            f"the ESS floor; a pair on a {KNOWN_BIASED} row is {KNOWN_BIASED}, whatever its "
            f"delta. Tracker and exact deltas are shown for reference and carry no verdict.",
            ['pair (a | b)', 'delta IS (nats)', 'SE (nats)', '|delta|/SE', 'verdict',
             'delta ema_logw (nats)', 'delta exact (nats)'],
            [[f"{p['a']} | {p['b']}", _f(p['delta']), _f(p['se'], '{:.4f}'),
              _f(p['z'], '{:.2f}'), p['verdict'], _f(p['tracker_delta']),
              _f(p['exact_delta'])] for p in result['pairs']]))
    notes = [f"  {r['condition']}: {'; '.join(r['notes'])}" for r in rows if r['notes']]
    if (meta['proposal'] == 'policy' and meta.get('eval_T') is not None
            and meta['eval_T'] != meta['steps']):
        notes.append(f"  (all): rollouts on integrator.T = {meta['steps']} steps; the eval's "
                     f"pooled log_Z uses eval_T = {meta['eval_T']}, a different discretisation")
    if notes:
        out.append('Notes:\n' + '\n'.join(notes))
    counts = verdict_counts(result)
    tail = ', '.join(f'{k} {v}' for k, v in counts.items()) or 'none'
    if exit_status(result) == EXIT_UNCHECKED:
        tail += (' -- NOTHING WAS CHECKED: no row has an exact value and no pair was named; '
                 'the estimates are floored, not validated')
    out.append('verdicts: ' + tail)
    return '\n\n'.join(out)


def exit_status(result: dict) -> int:
    """EXIT_FAIL on any verdict but PASS and ESTIMATE_ONLY; else EXIT_PASS when some check
    passed, EXIT_UNCHECKED when none was formed (module docstring)."""
    counts = verdict_counts(result)
    if set(counts) - {'PASS', ESTIMATE_ONLY}:
        return EXIT_FAIL
    return EXIT_PASS if counts.get('PASS') else EXIT_UNCHECKED


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help="the run's own config YAML")
    ap.add_argument('--checkpoint', default=None,
                    help='checkpoint .pt; omitted = an untrained policy (no tracker)')
    ap.add_argument('--proposal', choices=('policy', 'prior'), default='policy')
    ap.add_argument('--conditions', nargs='*', default=None,
                    help='identifiers to check (default: every condition row)')
    ap.add_argument('--n', type=int, default=10_000, help='rollouts per condition')
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--quad-grids', nargs=2, default=list(DEFAULT_GRIDS), type=GridSpec.parse,
                    metavar=('A', 'B'), help="per-block counts 'r,theta,phi[@shift]'")
    ap.add_argument('--quad-tol', type=float, default=0.02)
    ap.add_argument('--quad-lin-half', type=float, default=1.0)
    ap.add_argument('--quad-max-points', type=float, default=5e7)
    ap.add_argument('--max-quad-dim', type=int, default=6)
    ap.add_argument('--k-se', type=float, default=3.0)
    ap.add_argument('--abs-tol', type=float, default=0.0)
    ap.add_argument('--min-ess', type=float, default=100.0)
    ap.add_argument('--min-ess-frac', type=float, default=1e-3)
    ap.add_argument('--pair', nargs=2, action='append', default=[], metavar=('A', 'B'),
                    help='two conditions that must share log Z (repeatable)')
    ap.add_argument('--partition-identity', action='store_true',
                    help='sum over stereo sign assignments == unlocked (needs stereo_coeff)')
    ap.add_argument('--json', default=None, help='also write every number here')
    a = ap.parse_args(argv)
    try:
        return _run(a)
    except Refused as e:
        print(f'refused: {e}', file=sys.stderr)
        return EXIT_REFUSED


def _run(a) -> int:
    if a.partition_identity:
        missing = stereo_seam_missing()
        if missing:
            raise Refused(f'--partition-identity: ConformerTorsions takes no {missing}, so '
                          f'there is no stereo lock to partition. The mode is inert until the '
                          f'lock lands; nothing was run.')
        # the lock exists, but its per-member sign override -- what partition_identity's
        # energy_for_signs needs -- is not wired here yet: refused, not run half-built
        raise Refused('--partition-identity: ConformerTorsions takes stereo_coeff, but its '
                      'sign override is not wired into partition_identity yet '
                      '(eval/conformer_logz_check.py); nothing was run.')
    gpu_preflight(a.device, a.config)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    # the route's dtype, set before the config is read, as conformer_modeller's __main__
    torch.set_default_dtype(torch.float32)

    m = init_modeller(build_modeller(a.config, a.checkpoint, a.device),
                      policy=a.proposal == 'policy')
    result = run_check(
        m, a.conditions, a.proposal, a.n, a.batch, a.seed, a.quad_grids, a.quad_tol,
        a.quad_lin_half, a.max_quad_dim, a.quad_max_points, [tuple(p) for p in a.pair],
        Floors(k_se=a.k_se, abs_tol=a.abs_tol, min_ess=a.min_ess, min_ess_frac=a.min_ess_frac))
    print(format_report(result), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(result), indent=1), encoding='utf-8')
    return exit_status(result)

if __name__ == '__main__':
    sys.exit(main())
