r"""Offline evaluation of a trained conformer GFN, per condition: QUALITY, COVERAGE, TB / log Z.

A cheap proxy for "is the sampler right", read off N forward draws per condition plus
sampler-independent references. It runs NO Markov chain and estimates NO thermal
population: it does not establish that the sampler's basin weights are Boltzmann weights.

    python -m eval.conformer_model_eval --config <run.yaml> --checkpoint <ckpt.pt> \
        [--n-per-condition 2048] [--refs <table.pt>] [--device cpu|cuda] [--seed 0] \
        [--out <dir>] [--quadrature-max-d 6] [bars ...]

THE MODEL. Built through the trainer's own loader (`conformer_logz_check.build_modeller` /
`init_modeller`: the run's config, ConformerModeller, `Checkpointer.load_full` with the
checkpoint read-only and no buffer sidecar), so the rollouts use the checkpoint's eval model
(`model_eval`: the EMA weights, or the live policy itself when `cfg:ema_decay` is null --
train.py's update_ema_model then sets ema_model = gfn_model -- and the report says which),
its ragged set head rebuilt from the stored stamp (`ConformerModeller.gfn_from_config`), its
P_B snapshot where it has one, and its per-condition log Z tracker. Nothing trains and nothing
is written to the checkpoint directory. The prior dataset, which the eval never reads, is
drawn at its minimum (`_small_prior_draw`); only its identifiers enter the registry.
A mismatch is REFUSED (exit 2), not repaired:
  * config vs checkpoint: `Checkpointer.assert_problem_match` (energy_config incl.
    stereo_coeff, the conditioning flags) and the stored set-head stamp against the config
    (`_check_reloaded_policy`). utils._NON_IDENTITY_ENERGY_CONFIG_KEYS exempts some
    energy_config keys from that identity, and the checkpoint records no value for them;
    those that reach the energy's members (`identity_exempt`: energy_clip, bounding_coeff
    and lj_coeff on this route) are scored at the CONFIG's values, which the report prints --
    a config that changed one loads without refusal, and log R, the quadrature and the
    tracker and head gaps are then against another target than the one trained;
  * conditions file vs checkpoint: the carrier width, periodic columns and r|theta|phi split
    (`_assert_checkpoint_layout`), the condition set -- its identifiers in mol_id order and
    each member's signature -- against the one the checkpoint's gfn_config['conformer'] stamp
    records (`ConformerModeller._assert_condition_set`, reached because the eval loads
    through `load_full`, a full resume), and the tracker's library size against the number
    of conditions;
  * conditions file vs config: the per-graph `ctree_stereo_coeff` against the energy's
    (`_refuse_stereo_mismatch`).
A checkpoint written before the stamp carried the condition set is warned about, not refused;
against such a checkpoint, two condition files of equal layout and count that name different
molecules are not told apart here. The conditions file's sha256 and identifiers are printed
and written to the JSON for comparison with the run's provenance.
The checkpoint's `gfn_config['device']` is replaced by `--device` before the model is built:
GFN places constant tensors on it at construction, so a GPU checkpoint cannot otherwise be
built on the CPU. It carries no parameter.

THE DRAWS. Per condition, N rollouts of the eval model's forward policy from that condition's graph,
on `cfg:eval_T` uniform steps (the discretizer the run's eval uses; a difference from
`cfg:integrator.T`, the training discretisation, is noted), seeded per condition. Each row
keeps its terminal state, log R (the run's one-pass energy), log P_F, log P_B and the flow
head's log Z(c).

QUALITY (sample and force field only, plus the floor):
  finite       fraction of rows with a finite baked energy (`conformer_energy`);
  in-box       fraction with every non-periodic column inside |x| <= 1
               (`geometry_stats`' all_in_range, the rule `ConformerModeller._in_box` applies);
  lock-active  fraction of rows whose stereo-lock term is > 0 (an element inside its band or
               past it), and `inverted`, rows with an element on the wrong side (s v < 0);
               n/a when the member has no locked element or stereo_coeff is 0;
  clash        fraction of rows whose deepest nonbonded overlap (sigma - r) / sigma exceeds
               `--clash-overlap` (default 0.5, i.e. r < sigma / 2), with
               `prior_smoke._worst_overlap` over the force field's own nonbonded pairs (at
               least `min_separation` bonds apart, so 1-4 pairs are in). sigma is 2^(-1/6) R*,
               where MMFF's buffered 14-7 pair term crosses zero (0.889 R*), so r < sigma is on
               the repulsive wall. The pair term is 57 epsilon at overlap 0.25 and 1174 epsilon
               at 0.5; over the MMFF pairs of CO, CC, CCO, CCCC, phenol and
               N-methylacetamide (epsilon 0.019 to 0.068 kcal/mol) that is 1.1 to 3.9 kcal/mol
               at 0.25, about 1 to 4 kT at T = 1 kcal/mol, and 22 to 80 kcal/mol at 0.5. The
               quantiles and the reference conformer's own overlap are printed beside it. n/a
               without nonbonded pairs;
  excess       (E - e_min) / T in kT over finite rows, E the baked energy (clip(U + lock) +
               wall at T = 1, kcal/mol), e_min the condition's floor: from `--refs`
               (build_conformer_references.py), else built here once with the reduced search
               `EVAL_SEARCH` (starts and steps printed) and cached. The floor is a multi-start
               local minimum, an upper bound, so every excess is a lower bound;
  T_eff/T      `thermal_stats`' 1 + 2 median(excess) / k. 2.0 is its reading for a harmonic
               well of k coordinates sampled at the target temperature (median excess k/2);
               a proxy, not a thermal-population check.

COVERAGE (against `prior_diagnostics.basin_reference`, from the same table): rotamer basins
are the product of per-group rotamer centres found by 1-D scans of the true energy; a basin
is ACCESSIBLE when its energy, with every other coordinate at the reference geometry, is
within 10 kT of the best. n_missed counts accessible basins no draw lands in; the worst
accessible basin's share is printed against the uniform share 1/n_accessible. Uniform is not
the target: this reads visitation, not weights. `basin_coupling` (debiased total correlation
and suppressed combinations) is printed against the target's own `target_coupling`; on a
coupled molecule the product basin set over-counts reachable conformers, so n_missed is a
lower bound on what is missing. Where the table SKIPPED the enumeration (more product basins
than its max_modes), the product basins exist but were not scored; the bar then reads the
MARGINAL coverage instead (`marginal_coverage`: each group's rotamer centres from the same
1-D scans, a centre missed when no draw's group label reaches it -- a weaker check, a lower
bound on missed basins, and labelled 'marginal' wherever it is printed); without a member to
scan it is UNAVAILABLE, never n/a. Those basins do not include the two sides of a non-planar
centre: `parity_coverage` reads, for each such centre the stereo lock leaves free (a
three-coordinate centre, or any centre when stereo_coeff is 0), the share of draws on each
side, and counts the centre MISSED when every draw sits on one side. The centres are the entries
of energies/invertible_centres.py's structural table, which the fitted prior's two-sided draw
reads too, that are not `planar` (`free_centres`): a centre with a substituent offset whose
sign can be negated as an exact inversion without turning the centre's ring system, whose
reference offset sits more than `MIRROR_WIDTHS` prior widths from its flip (a working
assumption; a planar centre is its own flip). No energy enters. The prior draws both sides of
every free THREE-coordinate centre of the table, planar or not; a free four-coordinate one
(stereo_coeff 0) it draws on one side, and this metric still counts it. A caged N, with no
exocyclic substituent, has no entry.
PARITY ROWS FROM BEFORE 2026-09-28 ARE NOT COMPARABLE WITH LATER ONES on ring molecules, on
crowded centres and, under the lock, at a locked double bond's key atom: a checkpoint that
passed the parity bar before can fail it after, on the same draws. The earlier table reflected
a ring atom's siblings about the group's first row, which at a ring atom whose first row is not
its ring child moves the ring child and opens the ring, so the side read inaccessible; the table
now pivots on the ring child, covers a ring atom at its ring's closure through the ring frame,
and leaves out a centre whose flip would turn its ring system (a caged N, but also the root N
of C1COCCN1). The earlier rule also counted a centre only when its reflected reference
geometry, every rotor held, scored within 10 kT, so a crowded acyclic centre -- the N of
CCCN(C)C scores 150 kT there while the target holds half its mass on each side -- could not be
missed; every non-planar centre in the table is required on both sides now. And the earlier
table read a double-bond element's key atom as locked; it is free now, so under the lock it is
required on both sides where the width bar reads it non-planar, as the root C of
C/C=C/C(=O)N(C)C at 1.08 widths.

TB / LOG Z (the forward log-weight log w = log R + log P_B - log P_F):
  mean, std (the within-condition spread `logw_std_within` reads), ESS/N (Kish), the gap
  log Z_IS - mean log w (>= 0 up to noise), and a Pareto-k tail diagnostic: a generalised
  Pareto fit (Zhang & Stephens 2009, with PSIS's weakly informative prior) to the largest
  M = ceil(min(0.2 N, 3 sqrt N)) weights. k < 0.5 finite variance, 0.5 to 0.7 usable, above 0.7
  the IS estimate is not reliable (Vehtari et al., PSIS).
  IS log Z = logmeanexp(log w) with a bootstrap SE (and the delta-method SE of
  `conformer_logz_check.is_summary` beside it); the tracker's ema_logw (what lookup serves)
  and ema_log_z_emp from the checkpoint, with |IS - tracker| -- a gap, never an error; exact
  quadrature where k <= `--quadrature-max-d` (conformer_logz_check's two-grid rule, required
  to converge, the integrand checked first; cached); conditions of one constitution: mirror
  images (and two identifiers of one isomer) must share log Z, diastereomers need not.
  conformer_logz_check's KNOWN_BIASED label belongs to its prior proposal; the policy
  proposal here carries none, and its coverage is what the ESS and Pareto k cannot see --
  which is why the coverage family is beside them.
  THE IS STATUS, ON EVERY ROW (`is_status`). conformer_logz_check's `_is_problem` floors
  (NONFINITE_ROWS, BELOW_ESS_FLOOR: ESS < min_ess or ESS/N < min_ess_frac), then
  PARETO_K_ABOVE_BAR (k-hat >= max_pareto_k: PSIS's line past which the estimate, and its
  bootstrap SE, are not reliable), else 'ok'. It holds for a row WITHOUT an exact value too,
  where the estimate is the output. On a row that is not 'ok' the IS log Z is printed with its
  status, and every difference taken from it -- log Z_IS - mean log w, IS - tracker, head -
  IS, IS - exact in the summary -- is left blank.

VERDICTS. Each condition is judged against explicit bars (options; the defaults are WORKING
ASSUMPTIONS, printed in the verdict table's caption): finite >= 1.0, in-box >= 0.99,
lock-active < 1e-3, clash <= 1e-3, missed basins <= 0, missed free centres <= 0, the ESS
floors (ESS >= 100 and ESS/N >= 1e-3, no non-finite log w), Pareto k < 0.7, |IS - exact| <=
max(3 SE + |A - B|, 0.1) (conformer_logz_check's `exact_verdict`, which applies the floors
first). A bar whose quantity does not exist for the molecule is n/a; one whose input is
missing is UNAVAILABLE and fails.

EXIT STATUS. 0 every bar passed or is n/a and every required pair agreed; 1 some bar or pair
failed (and an uncaught error, with its traceback); 2 refused (a mismatch above, a busy card,
an unknown condition, a quadrature grid refusal).

COST. A cost table closes the report: setup, references, sampling (P_F/P_B rollouts), energy
(log R), sample metrics, lock and clash, TB statistics, quadrature.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

import eval.conformer_logz_check as lzc

#: the floor search when no --refs table is given: build_conformer_references' search at a
#: reduced depth (its DEFAULT_SEARCH is 32 ETKDG seeds, 127 uniform, 128 prior, 150 steps)
EVAL_SEARCH = dict(n_seeds=8, n_uniform=55, n_prior=0, steps=150)

#: rows below which a condition's sample statistics are not computed (per_molecule_block)
N_MIN = 32

#: bar outcomes. PASS and N_A are the two that do not fail the exit status
PASS, FAIL, N_A, UNAVAILABLE = 'PASS', 'FAIL', 'n/a', 'UNAVAILABLE'

#: pair kinds that must share log Z (`stereo_pairs`); a diastereomer pair carries no requirement
MUST_AGREE = ('mirror', 'same isomer')

EXIT_PASS, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2

#: the IS status of a row that clears conformer_logz_check's floors but not the Pareto-k bar
PARETO_K_ABOVE_BAR = 'PARETO_K_ABOVE_BAR'

Refused = lzc.Refused


@dataclass(frozen=True)
class Bars:
    """The per-condition bars. Every default is a WORKING ASSUMPTION (scope: this eval;
    revisit when a run's measured distribution of a quantity shows where the bar sits)."""
    min_finite_frac: float = 1.0
    min_in_box_frac: float = 0.99
    max_lock_active_frac: float = 1e-3          # strict: lock-active < this
    clash_overlap: float = 0.5                  # the clash definition, fraction of sigma
    max_clash_frac: float = 1e-3
    max_n_missed: int = 0
    max_parity_missed: int = 0
    max_pareto_k: float = 0.7                   # strict: k < this
    k_se: float = 3.0
    abs_tol: float = 0.1                        # nats
    min_ess: float = 100.0
    min_ess_frac: float = 1e-3

    def floors(self) -> 'lzc.Floors':
        return lzc.Floors(k_se=self.k_se, abs_tol=self.abs_tol, min_ess=self.min_ess,
                          min_ess_frac=self.min_ess_frac)


class Cost:
    """Wall seconds per phase."""

    def __init__(self):
        self.s: Dict[str, float] = defaultdict(float)

    @contextmanager
    def __call__(self, phase: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.s[phase] += time.perf_counter() - t


# ------------------------------------------------------------------ importance-weight statistics

def gpd_fit(x) -> tuple:
    """``(k, sigma)`` of a generalised Pareto fit to exceedances ``x >= 0``.

    Zhang & Stephens (2009) posterior-mean estimator of the profile parameter
    b = -k / sigma, over m = 30 + floor(sqrt n) grid points, then k = mean log(1 - b x). The
    parameterisation is F(x) = 1 - (1 + k x / sigma)^(-1/k): k > 0 is a heavy (Pareto) tail
    with finite moments below order 1/k. Then PSIS's weakly informative prior shrinks k toward
    0.5 with the weight of 10 observations, (n k + 5) / (n + 10) (Vehtari et al.). The same
    computation as arviz's `_gpdfit`, which this venv does not have.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    n = int(x.size)
    if n < 5 or not np.isfinite(x).all() or x[-1] <= 0.0:
        raise ValueError(f'a GPD fit needs >= 5 finite exceedances with a positive maximum; '
                         f'got n = {n}')
    m = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1, dtype=np.float64) - 0.5))
    b /= 3.0 * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    loglik = n * (np.log(-(b / k)) - k - 1.0)
    with np.errstate(over='ignore'):            # an overflowed term is a weight of 0
        w = 1.0 / np.exp(loglik - loglik[:, None]).sum(axis=1)
    keep = w >= 10.0 * np.finfo(np.float64).eps
    w, b = w[keep], b[keep]
    w /= w.sum()
    b_post = float((b * w).sum())
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    k_post = (n * k_post + 10.0 * 0.5) / (n + 10.0)
    return k_post, sigma


def pareto_khat(log_w) -> dict:
    """PSIS's k-hat for the importance weights ``exp(log_w)`` (finite rows).

    The tail is the M = ceil(min(0.2 N, 3 sqrt N)) largest weights, taken as exceedances over
    the (M+1)-th largest. Weights are scaled by the largest first, so only ratios enter.
    `k` is NaN, with the reason, when the tail holds fewer than 5 positive exceedances; a
    sample with zero log-weight spread has no tail at all and is reported as such.
    """
    lw = torch.as_tensor(log_w, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    lw = lw[np.isfinite(lw)]
    n = int(lw.size)
    out = dict(k=float('nan'), sigma=float('nan'), m_tail=0, n=n, note='')
    if n < 10:
        out['note'] = f'{n} finite weights: too few for a tail fit'
        return out
    m = int(math.ceil(min(0.2 * n, 3.0 * math.sqrt(n))))
    s = np.sort(lw)
    w = np.exp(s - s[-1])
    tail = w[-m:] - w[-(m + 1)]
    out['m_tail'] = m
    if (tail > 0).sum() < 5:
        out['note'] = ('zero log-weight spread' if float(s[-1] - s[0]) == 0.0
                       else 'fewer than 5 distinct tail weights')
        if float(s[-1] - s[0]) == 0.0:
            out['k'] = float('-inf')            # equal weights: no tail, IS exact
        return out
    k, sigma = gpd_fit(tail)
    out.update(k=float(k), sigma=float(sigma))
    return out


def bootstrap_log_z_se(log_w, n_boot: int, seed: int, chunk: int = 64) -> float:
    """SE of logmeanexp(log w) over `n_boot` resamples of the finite rows."""
    lw = torch.as_tensor(log_w, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    lw = lw[np.isfinite(lw)]
    n = int(lw.size)
    if n < 2 or n_boot < 2:
        return float('nan')
    rng = np.random.default_rng(int(seed))
    w = np.exp(lw - lw.max())
    vals = []
    for s in range(0, int(n_boot), chunk):
        b = min(chunk, int(n_boot) - s)
        idx = rng.integers(0, n, size=(b, n))
        vals.append(np.log(w[idx].mean(axis=1)))
    return float(np.std(np.concatenate(vals), ddof=1))


def logw_block(log_w, n_boot: int, seed: int) -> dict:
    """Every statistic of one condition's forward log-weights."""
    est = lzc.is_summary(log_w)
    lw = torch.as_tensor(log_w, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    fin = lw[np.isfinite(lw)]
    est['std_log_w'] = float(fin.std()) if fin.size else float('nan')
    est['kl_gap'] = est['log_z'] - est['mean_log_w']
    est['se_delta'] = est['se']
    est['se_boot'] = bootstrap_log_z_se(lw, n_boot, seed)
    # THE SE THE BARS READ is the bootstrap one (the delta-method value rides beside it)
    est['se'] = est['se_boot']
    pk = pareto_khat(lw)
    est['pareto_k'], est['pareto_m_tail'], est['pareto_note'] = pk['k'], pk['m_tail'], pk['note']
    return est


def is_status(tb: Optional[dict], bars: 'Bars') -> str:
    """Whether a row's IS estimate can be read: conformer_logz_check's floors
    (`_is_problem`), then the Pareto-k bar, else 'ok' (module docstring, THE IS STATUS)."""
    problem = lzc._is_problem(tb, bars.floors())
    if problem:
        return problem
    k = tb.get('pareto_k')
    if _fin(k) and float(k) >= bars.max_pareto_k:
        return PARETO_K_ABOVE_BAR
    return 'ok'


def _gap(rec: dict, value) -> float:
    """A number derived from the IS estimate, NaN (printed blank) on a row whose IS status is
    not 'ok' -- conformer_logz_check's `_gap` rule, over this eval's status."""
    return float(value) if rec.get('is_status') == 'ok' and _fin(value) else float('nan')


# ------------------------------------------------------------------ quality helpers

def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


@torch.no_grad()
def stereo_lock_stats(member, x, chunk: int = 4096) -> dict:
    """Rows whose stereo-lock term is > 0 (``active``) and rows with an element on the wrong
    side of zero (``inverted``: s v < 0), on the member's own table (`StereoTable`)."""
    coeff = float(getattr(member, 'stereo_coeff', 0.0) or 0.0)
    table = getattr(member, 'stereo', None)
    n_el = 0 if table is None else int(table.n)
    out = dict(stereo_coeff=coeff, n_elements=n_el)
    if coeff <= 0.0 or n_el == 0:
        out['na'] = 'stereo_coeff 0' if coeff <= 0.0 else 'no locked element'
        return out
    xs = torch.as_tensor(x, dtype=member.dtype, device=member.device)
    n = int(xs.shape[0])
    lock, inv = [], []
    for a, b in _chunks(n, chunk):
        pos = member.build_positions(xs[a:b]).reshape(b - a, -1, 3)
        lock.append(table.lock_energy(pos, coeff).double().cpu())
        _, _, sign, _ = table.tensors(pos.device, pos.dtype)
        inv.append(((sign * table.values(pos)) < 0).any(-1).cpu())
    lock, inv = torch.cat(lock), torch.cat(inv)
    active = lock > 0
    out.update(n=n, n_active=int(active.sum()), active_frac=float(active.double().mean()),
               n_inverted=int(inv.sum()), inverted_frac=float(inv.double().mean()),
               lock_max=float(lock.max()) if n else float('nan'))
    return out


@torch.no_grad()
def clash_stats(member, x, overlap: float, chunk: int = 4096) -> dict:
    """Deepest nonbonded overlap per row as a fraction of the pair's sigma
    (`prior_smoke._worst_overlap`), its quantiles, and the fraction above `overlap`."""
    from energies.prior_smoke import _worst_overlap

    xs = torch.as_tensor(x, dtype=member.dtype, device=member.device)
    n = int(xs.shape[0])
    _, ff1 = member._batch(1)
    n_pairs = int(ff1.pair_index.shape[0])
    out = dict(n_pairs=n_pairs, overlap_bar=float(overlap))
    if n_pairs == 0:
        sep = getattr(member, '_ff_kwargs', {}).get('min_separation', '?')
        out['na'] = f'no nonbonded pair (min_separation {sep} bonds)'
        return out
    worst = []
    for a, b in _chunks(n, chunk):
        _, ff = member._batch(b - a)
        worst.append(_worst_overlap(member.build_positions(xs[a:b]), ff, b - a).double().cpu())
    worst = torch.cat(worst).numpy()
    x0 = torch.zeros(1, xs.shape[1], dtype=member.dtype, device=member.device)
    ref = float(_worst_overlap(member.build_positions(x0), ff1, 1)[0])
    out.update(n=n, frac=float((worst > overlap).mean()), n_clash=int((worst > overlap).sum()),
               p50=float(np.percentile(worst, 50)), p99=float(np.percentile(worst, 99)),
               max=float(worst.max()), reference=ref)
    return out


def excess_stats(energy, e_min: Optional[float], temperature: float) -> dict:
    """(E - e_min) / T over the finite rows: quantiles, and rows more than
    EMIN_RESCORE_TOL below the floor (a floor that is not the minimum)."""
    from build_conformer_references import EMIN_RESCORE_TOL

    e = torch.as_tensor(energy, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
    e = e[np.isfinite(e)]
    if e_min is None or not math.isfinite(float(e_min)) or e.size == 0:
        return dict(na='no floor' if e.size else 'no finite energy')
    u = (e - float(e_min)) / float(temperature)
    return dict(e_min=float(e_min), p10=float(np.percentile(u, 10)),
                p50=float(np.percentile(u, 50)), p90=float(np.percentile(u, 90)),
                p99=float(np.percentile(u, 99)), max=float(u.max()),
                n_below_floor=int((e < float(e_min) - EMIN_RESCORE_TOL).sum()))


@torch.no_grad()
def marginal_coverage(member, x) -> dict:
    """Per-group rotamer coverage: the fallback where the product basins were not enumerated.

    The groups and their centres are `rotamer_modes`' (the 1-D scans of the true energy that
    `basin_reference` takes its product of, each group's leader moved with its followers
    tracking and every other coordinate at the reference geometry). Every centre it returns
    has a scan density above `min_rel` (0.05) of its group's best, within ln 20 = 3.0 kT, so
    each is accessible in 1-D; `n_accessible` counts them all. A centre is MISSED when no
    draw's group label (`rotamer_group_labels`) is that centre. Every product basin through a
    missed centre is missed too, so n_missed here is a lower bound on the missed basins, and
    visiting every centre does not show that any combination was visited.
    """
    from energies.prior_diagnostics import rotamer_group_labels, rotamer_modes

    groups = rotamer_modes(member)
    if not groups:
        return dict(kind='marginal', na='no rotor group')
    xs = torch.as_tensor(x, dtype=member.dtype, device=member.device)
    dof = np.concatenate([t.detach().double().cpu().numpy().reshape(xs.shape[0], -1)
                          for t in member.dof_from_state(xs)], axis=1)
    lab = rotamer_group_labels(groups, dof, int(member.n_r + member.n_th))
    n = int(lab.shape[0])
    counts = [np.bincount(lab[:, g], minlength=len(c)) for g, (_, c) in enumerate(groups)]
    missed = int(sum(int((c == 0).sum()) for c in counts))
    worst = min(float(c.min()) / n * len(c) for c in counts) if n else float('nan')
    return dict(kind='marginal', n_groups=len(groups), n_accessible=int(sum(map(len, counts))),
                n_missed=missed, n_visited=int(sum(map(len, counts))) - missed,
                worst_over_uniform=worst,
                per_group=[dict(rows=[int(r) for r in rows], n_centres=len(c),
                                missed=int((cnt == 0).sum()))
                           for (rows, c), cnt in zip(groups, counts)])


def coverage_block(phys_row: Mapping, ref: Mapping, member=None, x=None) -> dict:
    """The coverage readings of one `per_molecule_block` row, with the basin table's counts.

    `why` (no reference entry) leaves `n_missed` absent, which the coverage bar reads as
    UNAVAILABLE. A table that SKIPPED the enumeration (above its max_modes) gives the
    `marginal_coverage` of `member`'s draws `x` (kind 'marginal'), or, without them, `why`
    naming the skip: the basins exist and were not scored, which is UNAVAILABLE, not n/a.
    """
    br = (ref or {}).get('basin_ref')
    cov = {}
    if br is None:
        cov['why'] = 'no reference entry for this condition'
        return cov
    if 'skipped' in br:
        skipped = f"product enumeration skipped ({br['skipped']})"
        if member is None or x is None:
            return dict(why=f'{skipped}; no draws to read marginally')
        cov = marginal_coverage(member, x)
        cov['skipped'] = skipped
        return cov
    p = phys_row
    n_acc = int(np.asarray(br['accessible'], dtype=bool).sum())
    cov.update(kind='joint', n_groups=len(br['groups']), n_modes=len(br['combos']),
               n_accessible=n_acc,
               n_missed=p.get('cover/n_missed'), worst_frac=p.get('cover/worst_frac'),
               occupancy_entropy=p.get('cover/occupancy_entropy'),
               tc_debiased=p.get('cover/coupling_tc_debiased'),
               n_suppressed=p.get('cover/coupling_n_suppressed'),
               target_tc=(ref or {}).get('target_tc'),
               nonthermal_worst_basin=p.get('cover/nonthermal_worst_basin_frac'))
    if _fin(cov['n_missed']):
        cov['n_visited'] = n_acc - int(cov['n_missed'])
    if _fin(cov['worst_frac']) and n_acc:
        cov['worst_over_uniform'] = float(cov['worst_frac']) * n_acc
    return cov


@torch.no_grad()
def parity_coverage(member, x) -> dict:
    """Which side of each FREE non-planar centre the draws sit on.

    The centres are energies/invertible_centres.py's `free_centres`: the entries of the table
    the fitted prior's flip reads too -- each centre with a substituent offset whose sign can
    be negated as an exact inversion without turning the centre's ring system, taken from the
    chart's structure -- that the lock leaves free and that are not `planar` (reference offset
    within `MIRROR_WIDTHS` prior widths of its flip, a working assumption: a planar centre is
    its own flip). The target holds both sides of every one of them, so each must be two-sided
    (`n_accessible` = `n_centres`): a centre is MISSED when every draw sits on one side. A
    centre the stereo lock names (a tetrahedral element of its table, `stereo.kind` and
    `stereo.key`, when `stereo_coeff` > 0) is not free: the target holds one side there. A draw
    is on the REFERENCE side of a centre when the triple product of three of its neighbours
    about it has the sign it has at the reference (`reference_side`, on the draw's built
    positions). Rotamer basins do not include these sides: a molecule with no rotor (NH3) has
    one basin and full basin coverage whatever its draws. Each row carries `kind` (the flip:
    root, sibling or frame) and `n_bonded`; the prior flips the three-coordinate centres,
    planar ones too, and a four-coordinate one appears only when `stereo_coeff` is 0. At a
    collective level (`torsion`) no row moves alone, the target holds the reference's side of
    every centre, and the metric is n/a.
    """
    from energies.invertible_centres import free_centres, reference_side

    centres = free_centres(member)
    if not centres:
        if getattr(member, 'collective', False):
            return dict(na=f'collective level {member.level!r}: no row moves a centre alone, '
                           f'so the target holds the reference side of each', n_centres=0)
        return dict(na='no non-planar centre with a flip that the lock leaves free', n_centres=0)
    xs = torch.as_tensor(x, dtype=member.dtype, device=member.device)
    pos = member.build_positions(xs).reshape(xs.shape[0], -1, 3)
    rows = []
    for c in centres:
        frac = float(reference_side(pos, c).mean())
        rows.append(dict(centre=c.slot, name=c.name, kind=c.kind, n_bonded=int(c.n_bonded),
                         ref_side_frac=frac, missed=bool(frac in (0.0, 1.0))))
    return dict(n_centres=len(rows), n_accessible=len(rows),
                n_missed=sum(r['missed'] for r in rows),
                minority_frac=min(min(r['ref_side_frac'], 1.0 - r['ref_side_frac'])
                                  for r in rows),
                missed_names=[r['name'] for r in rows if r['missed']], centres=rows)


# ------------------------------------------------------------------ stereoisomer pairs

def stereo_pairs(smiles_of: Mapping[str, str]) -> List[dict]:
    """Pairs of conditions of one constitution (`build_conformer_set.constitution_key`),
    classified by `build_conformer_conditions.smiles_identity`: 'mirror' when one is the
    other with every tetrahedral centre inverted (`encoder_probe.mirror_smiles`), 'same
    isomer' when both name one stereoisomer, 'diastereomer' otherwise.

    A stereo nitrogen's tag is read as part of the identity (`lock_nitrogen=True`): an
    identifier carries one only when its set was built under `lock_stereo_nitrogen`, where
    two invertomers are two isomers, and an identifier without one reads the same either way.
    """
    from build_conformer_conditions import smiles_identity
    from build_conformer_set import constitution_key
    from models.encoder_probe import mirror_smiles

    groups = defaultdict(list)
    for ident, smi in smiles_of.items():
        groups[constitution_key(smi)].append(ident)
    out = []
    for key, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        idt = {i: smiles_identity(smiles_of[i], True) for i in ids}
        mir = {i: smiles_identity(mirror_smiles(smiles_of[i]), True) for i in ids}
        for a, b in itertools.combinations(sorted(ids), 2):
            if idt[a] == idt[b]:
                kind = 'same isomer'
            elif mir[a] == idt[b]:
                kind = 'mirror'
            else:
                kind = 'diastereomer'
            out.append(dict(a=a, b=b, kind=kind, constitution=key))
    return out


# ------------------------------------------------------------------ the model

def build_eval_modeller(config: str, checkpoint: Optional[str], device: str = 'cpu'):
    """`conformer_logz_check.build_modeller` (read-only, no buffer sidecar), with the
    checkpoint's `gfn_config['device']` replaced by this eval's device (module docstring)."""
    m = lzc.build_modeller(config, checkpoint, device)
    ck = m.checkpointer
    stored_fn = ck._gfn_config_from

    def on_eval_device(blob):
        cfg = stored_fn(blob)
        m._eval_device_swap = (cfg.get('device'), str(m.device))
        cfg['device'] = str(m.device)
        return cfg

    ck._gfn_config_from = on_eval_device
    return m


def _small_prior_draw(m):
    """Make this instance's `init_prior_dataset` draw a CARRIER prior at its minimum.

    The eval never reads the prior dataset; `init_identifiers` reads its identifiers, which
    enter the registry that condition_id (and so the tracker row) indexes.
    `_draw_carrier_prior` gives every member max(ceil(n / members), 2) rows whatever n, so at
    n = 1 each member still draws and the registry is the trainer's, while the draw and its
    scoring no longer grow with cfg:energy_config.prior_sample_size. A file prior
    (cfg:prior_path) and the single-molecule route keep the trainer's call unchanged. The
    config value is restored after the call; the checkpoint's problem identity was already
    checked in init_gfn, which runs first.
    """
    orig = m.init_prior_dataset

    def init_prior_dataset():
        ec = m.args.energy_config
        if (not getattr(m.energy_function, 'is_carrier', False)
                or getattr(m.args, 'prior_path', None)):
            return orig()
        had = hasattr(ec, 'prior_sample_size')
        old = getattr(ec, 'prior_sample_size', None)
        ec.prior_sample_size = 1
        m._eval_prior_rows = (old, 1)
        try:
            return orig()
        finally:
            if had:
                ec.prior_sample_size = old
            else:
                delattr(ec, 'prior_sample_size')

    m.init_prior_dataset = init_prior_dataset
    return m


def init_eval_modeller(m):
    """`conformer_logz_check.init_modeller`, with every load-time mismatch turned into a
    `Refused` naming it, then the tracker checks."""
    if getattr(m.args, 'energy_function', None) != 'conformer_torsions':
        raise Refused(f"energy_function {getattr(m.args, 'energy_function', None)!r}: this "
                      f"eval reads the conformer route only")
    _small_prior_draw(m)
    try:
        lzc.init_modeller(m)
    except SystemExit as e:                      # _refuse_stereo_mismatch and kin
        raise Refused(f'the conditions file does not match this run: {e}') from None
    except (ValueError, NotImplementedError) as e:
        # assert_problem_match, _assert_checkpoint_layout, _assert_condition_set (a full
        # resume onto another condition set), _check_reloaded_policy
        raise Refused(f'the checkpoint does not match this config or conditions file '
                      f'({type(e).__name__}): {e}') from None
    ef = m.energy_function
    named = getattr(m.args, 'checkpoint_name', None)
    tracker = getattr(m, 'condition_log_z', None)
    if named and tracker is None:
        raise Refused('the checkpoint carries no condition_log_z tracker')
    if tracker is not None and int(tracker.library_size) != int(ef.condition_library_size):
        raise Refused(f"the checkpoint's tracker holds {int(tracker.library_size)} conditions "
                      f"and this conditions file gives {int(ef.condition_library_size)}: its "
                      f"rows would index other molecules")
    return m


def _members(ef, idents) -> Dict[str, object]:
    members = getattr(ef, '_members', None)
    return dict(members) if members is not None else {i: ef for i in idents}


# ------------------------------------------------------------------ references

def _energy_kwargs(m) -> dict:
    return dict(vars(m.args.energy_config))


def load_or_build_references(m, refs_path: Optional[str], cache_dir: Path,
                             identifiers: Optional[Sequence[str]] = None, log=print):
    """``(ReferenceTable, info)``: `--refs` loaded and checked, else a table built with
    `EVAL_SEARCH` into the cache and reused while its stamp matches and it holds an entry,
    or a recorded failure, for every identifier asked for. `identifiers` (default: every
    condition of the file) restricts the build and the re-score to those conditions; the
    build resumes from the per-molecule parts `build_references` keeps beside a subset table.
    Every entry asked for has its floor state re-scored through the run's member
    (`ReferenceTable.verify`)."""
    from build_conformer_references import (StaleReferencesError, build_references,
                                            load_references, member_kwargs)

    cond = Path(m.args.molecules_path)
    ekw = _energy_kwargs(m)
    # the table covers cfg:molecules_path, i.e. mol_dataset; a held-out condition has no entry
    in_file = set(getattr(getattr(m.mol_dataset, 'batch', None), 'identifier', None) or [])
    want = (None if identifiers is None
            else [i for i in dict.fromkeys(identifiers) if not in_file or i in in_file])
    if refs_path:
        try:
            table = load_references(refs_path, conditions_path=cond, energy_kwargs=ekw,
                                    require_complete=False)
        except StaleReferencesError as e:
            raise Refused(f'--refs {refs_path}: {e}') from None
        info = dict(source='table', path=str(refs_path))
    else:
        cache = Path(cache_dir) / 'references.eval.pt'
        table = None
        if cache.exists():
            try:
                table = load_references(cache, conditions_path=cond, energy_kwargs=ekw,
                                        require_complete=False)
                have = {k: table.stamp['search'].get(k) for k in EVAL_SEARCH}
                need = [i for i in (want if want is not None
                                    else table.stamp['conditions']['identifiers'])
                        if i not in table and i not in table.failures]
                if have != EVAL_SEARCH:
                    log(f'references: cache {cache} was built with search {have}, not '
                        f'{EVAL_SEARCH}; rebuilding')
                    table = None
                elif need:
                    log(f'references: cache {cache} has no entry for {len(need)} condition(s) '
                        f'asked for, e.g. {need[0]!r}; building them')
                    table = None
            except StaleReferencesError as e:
                log(f'references: cache {cache} does not describe this run ({str(e)[:200]}); '
                    f'rebuilding')
                table = None
        if table is None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            res = build_references(cond, cache, member_kwargs(ekw), search=EVAL_SEARCH,
                                   internal_prior_path=None, workers=0, identifiers=want,
                                   log=log)
            if res['failures']:
                log(f'references: {len(res["failures"])} molecule(s) failed: '
                    f'{dict(list(res["failures"].items())[:3])}')
            table = load_references(cache, conditions_path=cond, energy_kwargs=ekw,
                                    require_complete=False)
        info = dict(source='eval_search', path=str(cache))
    search = table.stamp.get('search', {})
    info.update(n_starts=int(sum(search.get(k, 0) for k in ('n_seeds', 'n_uniform', 'n_prior'))
                             + 1),
                steps=search.get('steps'),
                missing=[i for i in table.missing if want is None or i in want],
                failures={i: e for i, e in table.failures.items() if want is None or i in want},
                format=table.stamp.get('format'))
    idents = [i for i in table.identifiers if want is None or i in want]
    members = {i: mb for i, mb in _members(m.energy_function, idents).items()
               if i in table and i in idents}
    try:
        info['verify_worst_kcal'] = table.verify(members, allow_missing=True)
    except StaleReferencesError as e:
        raise Refused(f'references: {e}') from None
    return table, info


# ------------------------------------------------------------------ quadrature (cached)

def _quad_key(m, ident: str, grids, quad_tol, lin_half, max_points) -> str:
    from build_conformer_references import defining_energy, file_sha256, member_kwargs
    ef = m.energy_function
    member = lzc._member(ef, ident)
    blob = dict(conditions=file_sha256(m.args.molecules_path), ident=ident,
                smiles=getattr(member, 'smiles', None),
                energy=defining_energy(member_kwargs(_energy_kwargs(m))),
                temperature=float(ef.temperature), grids=[str(g) for g in grids],
                quad_tol=float(quad_tol), lin_half=float(lin_half), max_points=float(max_points))
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def cached_quadrature(m, ident, grids, quad_tol, lin_half, max_points, cache_file: Path):
    """`conformer_logz_check.two_grid_log_z` on the member, memoised in `cache_file` by the
    conditions file, the member-defining energy, T and the grid arguments."""
    key = _quad_key(m, ident, grids, quad_tol, lin_half, max_points)
    store = {}
    if cache_file.exists():
        try:
            store = json.loads(cache_file.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            store = {}
    if key in store:
        return dict(store[key], cached=True)
    member = lzc._member(m.energy_function, ident)
    res = lzc.two_grid_log_z(member, grids[0], grids[1], quad_tol, lin_half, max_points)
    store[key] = res
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(cache_file.name + '.tmp')
    tmp.write_text(json.dumps(store, indent=1), encoding='utf-8')
    os.replace(tmp, cache_file)
    return dict(res, cached=False)


# ------------------------------------------------------------------ the draws

@torch.no_grad()
def rollout_condition(m, ident: str, dataset, row: int, n: int, batch: int, seed: int,
                      steps: Optional[int] = None, cost: Optional[Cost] = None) -> dict:
    """N forward rollouts of `ema_model` (the eval model) for one condition, as
    `eval/utils.py::sample_eval_fwd_trajs` makes them (condition_samples, get_traj_fwd,
    log_reward with return_exp), timed in two parts, keeping every row's terminal state."""
    from utils import get_gfn_init_state, uniform_discretizer

    cost = cost or Cost()
    ef, model = m.energy_function, m.ema_model
    T = int(steps if steps is not None else m.args.eval_T)
    disc = lambda bsz: uniform_discretizer(bsz, T)
    torch.manual_seed(lzc._seed_for(seed, ident))
    parts = defaultdict(list)
    done = 0
    while done < n:
        b = min(batch, n - done)
        with cost('sampling'):
            mb = lzc._condition_batch(dataset, row, b, m.device)
            init = get_gfn_init_state(b, ef.data_ndim, m.device)
            temps = ef.temperature * torch.ones(b, dtype=torch.float32, device=m.device)
            mb, log_t, cond, cid = ef.condition_samples(mb, sg_inds=None, z_primes=None,
                                                        temperature=temps)
            states, log_pfs, log_pbs, log_flow, _ = model.get_traj_fwd(
                init, disc, None, cond.to(model.device), mb, return_gauss_params=True)
        with cost('energy'):
            log_r, sb = ef.log_reward(states[:, -1], mol_batch=mb, log_temperature=log_t,
                                      return_exp=True)
        cpu = lambda t: t.detach().cpu()
        parts['x'].append(cpu(states[:, -1]))
        parts['log_r'].append(cpu(log_r).double())
        parts['log_pf'].append(cpu(log_pfs.sum(-1)).double())
        parts['log_pb'].append(cpu(log_pbs.sum(-1)).double())
        parts['head'].append(cpu(log_flow[:, 0]).double())
        parts['energy'].append(cpu(sb.conformer_energy).reshape(-1).double())
        parts['cid'].append(cpu(cid).reshape(-1))
        done += b
    out = {k: torch.cat(v) for k, v in parts.items()}
    cids = torch.unique(out.pop('cid'))
    if cids.numel() != 1:
        raise RuntimeError(f'{ident}: rollouts carried condition_id {cids.tolist()}')
    out['condition_id'] = int(cids)
    out['log_w'] = out['log_r'] + out['log_pb'] - out['log_pf']
    out['steps'] = T
    return out


# ------------------------------------------------------------------ verdicts

def _fin(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))


def _bar(value, ok, bar: str, na: Optional[str] = None, note: str = '') -> dict:
    if na:
        return dict(status=N_A, value=value, bar=bar, note=na)
    if not _fin(value):
        return dict(status=UNAVAILABLE, value=value, bar=bar, note=note or 'not computed')
    return dict(status=PASS if ok(float(value)) else FAIL, value=float(value), bar=bar,
                note=note)


def judge(rec: dict, bars: Bars) -> Dict[str, dict]:
    """Every bar for one condition row. A section absent from `rec` leaves its bars
    UNAVAILABLE."""
    p, lock, clash = rec.get('phys', {}), rec.get('lock', {}), rec.get('clash', {})
    v = {}
    v['finite'] = _bar(p.get('phys/finite_frac'), lambda x: x >= bars.min_finite_frac,
                       f'>= {bars.min_finite_frac:g}')
    v['in_box'] = _bar(p.get('geom/all_in_range'), lambda x: x >= bars.min_in_box_frac,
                       f'>= {bars.min_in_box_frac:g}',
                       na='no free r/theta column' if p.get('geom/all_in_range_frozen') else None)
    v['lock'] = _bar(lock.get('active_frac'), lambda x: x < bars.max_lock_active_frac,
                     f'< {bars.max_lock_active_frac:g}', na=lock.get('na'))
    v['clash'] = _bar(clash.get('frac'), lambda x: x <= bars.max_clash_frac,
                      f'<= {bars.max_clash_frac:g} (overlap > {bars.clash_overlap:g} sigma)',
                      na=clash.get('na'))
    cov = rec.get('coverage', {})
    v['coverage'] = _bar(cov.get('n_missed'), lambda x: x <= bars.max_n_missed,
                         f"n_missed <= {bars.max_n_missed}"
                         + (' (marginal: rotamer centres)' if cov.get('kind') == 'marginal'
                            else ''),
                         na=cov.get('na'), note=cov.get('why', '') or cov.get('skipped', ''))
    par = rec.get('parity', {})
    v['parity'] = _bar(par.get('n_missed'), lambda x: x <= bars.max_parity_missed,
                       f'n_missed <= {bars.max_parity_missed}', na=par.get('na'),
                       note=', '.join(par.get('missed_names', [])))
    tb = rec.get('tb')
    fl = bars.floors()
    floor_bar = (f'ESS >= {bars.min_ess:g}, ESS/N >= {bars.min_ess_frac:g}, no non-finite '
                 f'log w')
    if not tb:
        v['ess_floor'] = dict(status=UNAVAILABLE, value=None, bar=floor_bar, note='no draws')
    else:
        problem = lzc._is_problem(tb, fl)
        v['ess_floor'] = dict(status=FAIL if problem else PASS, value=tb.get('ess'),
                              bar=floor_bar, note=problem or '')
    tb = tb or {}
    k = tb.get('pareto_k')
    if k == float('-inf'):
        v['pareto_k'] = dict(status=PASS, value=k, bar=f'< {bars.max_pareto_k:g}',
                             note=tb.get('pareto_note', ''))
    else:
        v['pareto_k'] = _bar(k, lambda x: x < bars.max_pareto_k, f'< {bars.max_pareto_k:g}',
                             note=tb.get('pareto_note', ''))
    ex = rec.get('exact')
    if ex is None:
        v['exact_logz'] = dict(status=N_A, value=None, bar='-', bar_value=float('nan'),
                               note=rec.get('exact_na', ''))
    else:
        # conformer_logz_check's rule: its floors first, then UNCONVERGED, then the bar
        verdict, bar = lzc.exact_verdict(tb or None, ex, fl)
        d = tb.get('log_z', float('nan')) - ex.get('log_z', float('nan'))
        v['exact_logz'] = dict(status=PASS if verdict == 'PASS' else FAIL,
                               value=abs(d) if math.isfinite(d) else None,
                               bar=(f'<= {bar:.4g}' if _fin(bar) else '-'), bar_value=bar,
                               note=verdict)
    return v


def overall(verdicts: Mapping[str, dict]) -> str:
    bad = [k for k, v in verdicts.items() if v['status'] in (FAIL, UNAVAILABLE)]
    return PASS if not bad else FAIL + ':' + ','.join(bad)


# ------------------------------------------------------------------ the evaluation

def evaluate(m, *, n: int = 2048, batch: int = 2048, seed: int = 0,
             conditions: Optional[Sequence[str]] = None, refs=None, refs_info=None,
             bars: Bars = Bars(), quadrature_max_d: int = 6,
             grids: Sequence[str] = lzc.DEFAULT_GRIDS, quad_tol: float = 0.02,
             lin_half: float = 1.0, max_points: float = 5e7, n_boot: int = 200,
             cache_dir: Optional[Path] = None, steps: Optional[int] = None,
             cost: Optional[Cost] = None) -> dict:
    """Every number of the report for an initialised modeller (`init_eval_modeller`).

    `refs` is a `ReferenceTable` or a plain ``{identifier: {e_min, basin_ref, target_tc}}``
    (tests); None makes the floor and coverage readings UNAVAILABLE.
    """
    from energies.conformer_eval_metrics import per_molecule_block, split_by_member

    cost = cost or Cost()
    ef = m.energy_function
    rows_of = lzc._condition_rows(m)
    idents = list(conditions) if conditions else list(dict.fromkeys(rows_of))
    unknown = [c for c in idents if c not in rows_of]
    if unknown:
        raise Refused(f'no condition row for {unknown}; the run holds {list(rows_of)}')
    eval_refs = (refs.eval_refs() if hasattr(refs, 'eval_refs') else dict(refs or {}))
    ga, gb = (g if isinstance(g, lzc.GridSpec) else lzc.GridSpec.parse(g) for g in grids)
    cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / \
        'conformer_model_eval_cache'
    fl = bars.floors()

    # ---- quadrature preconditions BEFORE any rollout: the grid pair, its size, the integrand
    quad_plan, quad_na = {}, {}
    with cost('quadrature'):
        for ident in idents:
            member = lzc._member(ef, ident)
            k = int(member.data_ndim)
            if k > quadrature_max_d:
                quad_na[ident] = f'k = {k} > {quadrature_max_d}'
                continue
            try:
                lzc.refuse_unrefined(member, ga, gb)
                for g in (ga, gb):
                    lzc.refuse_over_budget(member, g, lin_half, max_points)
            except NotImplementedError as e:          # transverse columns
                quad_na[ident] = str(e).splitlines()[0]
                continue
            ds, row = rows_of[ident]
            quad_plan[ident] = lzc.check_integrand(ef, ident, ds, row, 1e-4, seed=seed)

    tracker = (m.condition_log_z.state_dict() if hasattr(m, 'condition_log_z') else None)
    members, states, energies, draws = {}, {}, {}, {}
    for ident in idents:
        ds, row = rows_of[ident]
        d = rollout_condition(m, ident, ds, row, n, batch, seed, steps=steps, cost=cost)
        with cost('sample metrics'):
            mb_all = ds.sample_graphs_at([row], repeats=n)
            mem, xs, es, _ = split_by_member(ef, d['x'], mb_all, d['energy'])
            if len(mem) != 1:
                raise RuntimeError(f'{ident}: the draws resolved to {list(mem)}')
            (key,) = mem
            members[ident], states[ident], energies[ident] = mem[key], xs[key], es[key]
        draws[ident] = d

    with cost('sample metrics'):
        phys = per_molecule_block(members, states, energies, eval_refs, n_min=min(N_MIN, n),
                                  nonthermal_entropy_per_dim=getattr(
                                      m.args, 'nonthermal_entropy_per_dim', None))

    smiles_of = {i: str(getattr(members[i], 'smiles', i)) for i in idents}
    out_rows = []
    for ident in idents:
        member, d = members[ident], draws[ident]
        ref = eval_refs.get(ident) or {}
        rec = dict(condition=ident, smiles=smiles_of[ident], level=getattr(member, 'level', None),
                   k=int(member.ndim), n=int(n), condition_id=d['condition_id'],
                   phys=phys[ident], notes=[])
        with cost('lock and clash'):
            rec['lock'] = stereo_lock_stats(member, states[ident])
            rec['clash'] = clash_stats(member, states[ident], bars.clash_overlap)
        rec['excess'] = excess_stats(energies[ident], ref.get('e_min'),
                                     float(member.temperature))

        with cost('sample metrics'):
            rec['coverage'] = coverage_block(phys[ident], ref, member, states[ident])
            rec['parity'] = parity_coverage(member, states[ident])

        # ---- TB / log Z
        with cost('TB statistics'):
            rec['tb'] = logw_block(d['log_w'], n_boot, lzc._seed_for(seed, ident) + 1)
            rec['tb']['head'] = float(d['head'].mean())
            rec['tb']['head_spread'] = float(d['head'].max() - d['head'].min())
        rec['tracker'] = lzc.tracker_reading(tracker, d['condition_id'])
        rec['exact'], rec['exact_check'] = None, None
        if ident in quad_plan:
            with cost('quadrature'):
                rec['exact'] = cached_quadrature(m, ident, (ga, gb), quad_tol, lin_half,
                                                 max_points, cache_dir / 'quadrature.json')
            rec['exact']['identity'] = quad_plan[ident]
            if not rec['exact']['converged']:
                rec['notes'].append(f"quadrature UNCONVERGED: |A-B| = {rec['exact']['diff']:.3g}"
                                    f" > {quad_tol:g} nats")
        else:
            rec['exact_na'] = quad_na.get(ident, '')
        rec['is_status'] = is_status(rec['tb'], bars)
        rec['verdicts'] = judge(rec, bars)
        rec['overall'] = overall(rec['verdicts'])
        if rec['exact'] is not None:
            ev = rec['verdicts']['exact_logz']
            rec['exact_check'] = dict(verdict=ev['note'], bar=ev['bar_value'])
        if rec['is_status'] != 'ok':
            rec['notes'].append(f"IS status {rec['is_status']} (ESS {rec['tb']['ess']:.1f}, "
                                f"Pareto k {rec['tb']['pareto_k']:.3g}): its log Z is printed "
                                f"with the status and every gap taken from it is blank")
        if rec['excess'].get('n_below_floor'):
            rec['notes'].append(f"{rec['excess']['n_below_floor']} row(s) below the floor e_min: "
                                f"the floor is not this condition's minimum")
        out_rows.append(rec)

    # ---- pairs of one constitution
    by = {r['condition']: r for r in out_rows}
    pairs = []
    for pr in stereo_pairs(smiles_of):
        a, b = by[pr['a']], by[pr['b']]
        rows = [dict(condition=r['condition'], tracker=r['tracker'], exact=r['exact'],
                     bias=None, **{'is': r['tb']}) for r in (a, b)]
        pv = lzc.pair_verdict(rows[0], rows[1], fl)
        # past the floors, a side whose k-hat is above the bar has no reliable SE either
        k_side = [r['condition'] for r in (a, b) if r['is_status'] == PARETO_K_ABOVE_BAR]
        if k_side and pv['verdict'] in ('PASS', 'FAIL'):
            pv['verdict'] = PARETO_K_ABOVE_BAR
        pv.update(kind=pr['kind'], constitution=pr['constitution'],
                  required=pr['kind'] in MUST_AGREE)
        pairs.append(pv)

    integ = getattr(m.args, 'integrator', None)
    return dict(
        meta=dict(n=int(n), batch=int(batch), seed=int(seed), steps=int(draws[idents[0]]['steps']),
                  train_steps=getattr(integ, 'T', None), eval_T=getattr(m.args, 'eval_T', None),
                  temperature=float(ef.temperature), device=str(m.device),
                  energy_clip=getattr(ef, 'energy_clip', None),
                  identity_exempt=identity_exempt(m),
                  ema_decay=getattr(m.args, 'ema_decay', None),
                  prior_rows=getattr(m, '_eval_prior_rows', None),
                  checkpoint=getattr(m.args, 'checkpoint_name', None),
                  step=int(getattr(m, 'step_ind', 0) or 0), stage=getattr(m, 'stage', None),
                  pb_frozen=bool(getattr(m.ema_model, 'pb_frozen', False)),
                  device_swap=getattr(m, '_eval_device_swap', None),
                  tracker=tracker is not None,
                  conditions_file=str(getattr(m.args, 'molecules_path', None)),
                  conditions_sha256=_sha_or_none(getattr(m.args, 'molecules_path', None)),
                  identifiers=idents, refs=refs_info or {},
                  grids=[str(ga), str(gb)], quad_tol=quad_tol, lin_half=lin_half,
                  quadrature_max_d=int(quadrature_max_d), n_boot=int(n_boot),
                  nonthermal_entropy_per_dim=getattr(m.args, 'nonthermal_entropy_per_dim', None),
                  bars=asdict(bars), cache_dir=str(cache_dir)),
        conditions=out_rows, pairs=pairs, cost=dict(cost.s))


def identity_exempt(m) -> Dict[str, object]:
    """The energy_config keys `utils._NON_IDENTITY_ENERGY_CONFIG_KEYS` exempts from the
    checkpoint's problem identity that are ConformerTorsions arguments, with the value the
    members are built with -- the config's, or the signature default where the config omits
    the key (`member_kwargs`' filter): scored here, and recorded nowhere in the checkpoint."""
    import inspect

    from build_conformer_references import _ct_parameters, member_kwargs
    from utils import _NON_IDENTITY_ENERGY_CONFIG_KEYS

    mk, params = member_kwargs(_energy_kwargs(m)), _ct_parameters()
    return {k: (mk[k] if k in mk else params[k].default)
            for k in _NON_IDENTITY_ENERGY_CONFIG_KEYS
            if k in params and (k in mk or params[k].default is not inspect.Parameter.empty)}


def _sha_or_none(path) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    from build_conformer_references import file_sha256
    return file_sha256(path)


def exit_status(result: dict) -> int:
    bad = [r for r in result['conditions'] if r['overall'] != PASS]
    bad += [p for p in result['pairs'] if p['required'] and p['verdict'] != 'PASS']
    return EXIT_FAIL if bad else EXIT_PASS


# ------------------------------------------------------------------ report

_f = lzc._f


def _g(v, fmt='{:.3g}'):
    return _f(v, fmt)


def _median_worst(rows, get, worst='max'):
    """``(median, worst value, who)`` over the rows with a finite value. `who` names the
    condition at the worst value, or says it is a tie: 'all tied (n)' when every row has it,
    else the tied conditions ('A, B +2 more')."""
    vals = [(r['condition'], get(r)) for r in rows]
    vals = [(c, float(v)) for c, v in vals if _fin(v)]
    if not vals:
        return None, None, None
    arr = np.asarray([v for _, v in vals])
    wv = float(arr.max() if worst == 'max' else arr.min())
    at = [c for c, v in vals if v == wv]
    if len(at) == 1:
        who = at[0]
    elif len(at) == len(vals):
        who = f'all tied ({len(at)})'
    else:
        who = 'tie: ' + ', '.join(at[:3]) + (f' +{len(at) - 3} more' if len(at) > 3 else '')
    return float(np.median(arr)), wv, who


def _policy_words(meta) -> str:
    """What drew the rollouts: the checkpoint's eval model, EMA or live."""
    if not meta['checkpoint']:
        return 'rollouts of the untrained ema_model as built'
    d = meta.get('ema_decay')
    if d is None:
        return ("rollouts of the checkpoint's model_eval, which is the live policy (ema_decay "
                "null: train.py's update_ema_model sets ema_model = gfn_model)")
    return f"rollouts of the checkpoint's model_eval (EMA weights, ema_decay {d:g})"


def format_report(res: dict) -> str:
    meta, rows = res['meta'], res['conditions']
    T = lzc._table
    b = meta['bars']
    who = (f"checkpoint {Path(str(meta['checkpoint'])).name} at step {meta['step']} (stage "
           f"{meta['stage']})" if meta['checkpoint'] else 'an UNTRAINED policy (no checkpoint)')
    levels = '/'.join(sorted({str(r['level']) for r in rows}))
    base = (f"{who}; N = {meta['n']} {_policy_words(meta)} per condition, seed {meta['seed']}, "
            f"{meta['steps']} uniform steps, level {levels}, T = {meta['temperature']:g} kcal/mol, "
            f"{len(rows)} condition(s)")
    rf = meta['refs']
    if rf.get('source') == 'table':
        floor = (f"floor e_min and basin table from --refs {rf['path']} ({rf.get('n_starts')} "
                 f"starts x {rf.get('steps')} steps per molecule)")
    elif rf.get('source') == 'eval_search':
        floor = (f"floor e_min and basin table built here with {rf.get('n_starts')} starts x "
                 f"{rf.get('steps')} steps per molecule (EVAL_SEARCH), cached at {rf['path']}")
    else:
        floor = 'NO reference table: floor and coverage readings are unavailable'
    status_rule = (f"IS status = conformer_logz_check's floors (ESS >= {b['min_ess']:g} and "
                   f"ESS/N >= {b['min_ess_frac']:g}, no non-finite log w), then "
                   f"{PARETO_K_ABOVE_BAR} at Pareto k >= {b['max_pareto_k']:g}, else ok")
    out = []
    out.append(T(
        'Table 1a. QUALITY: validity of the draws, per condition.',
        f"{base}. finite = baked energy finite; in-box = every non-periodic column within "
        f"|x| <= 1; lock-active = stereo-lock term > 0 (bar < {b['max_lock_active_frac']:g}), "
        f"inverted = an element on the wrong side; clash = deepest nonbonded overlap "
        f"(sigma - r)/sigma above {b['clash_overlap']:g} (bar <= {b['max_clash_frac']:g}), with "
        f"that overlap's p99 over the draws and at the reference conformer; clip-active = raw "
        f"U above energy_clip. Fractions are over the N draws. '-' = not applicable (no locked "
        f"element / no nonbonded pair).",
        ['condition', 'k (free coords)', 'finite (frac)', 'in-box (frac)', 'clip-active (frac)',
         'locked elements (count)', 'lock-active (frac)', 'inverted (frac)',
         'nonbonded pairs (count)', 'clash (frac)', 'overlap p99 (frac of sigma)',
         'overlap at ref (frac of sigma)'],
        [[r['condition'], r['k'], _g(r['phys'].get('phys/finite_frac'), '{:.4f}'),
          _g(r['phys'].get('geom/all_in_range'), '{:.4f}'),
          _g(r['phys'].get('E/clip_active_frac'), '{:.4f}'),
          r['lock']['n_elements'] if r['lock'].get('stereo_coeff', 0) > 0 else '0 (off)',
          _g(r['lock'].get('active_frac'), '{:.2e}'), _g(r['lock'].get('inverted_frac'), '{:.2e}'),
          r['clash']['n_pairs'], _g(r['clash'].get('frac'), '{:.2e}'),
          _g(r['clash'].get('p99'), '{:.3f}'), _g(r['clash'].get('reference'), '{:.3f}')]
         for r in rows]))
    out.append(T(
        'Table 1b. QUALITY: energy against the floor, per condition.',
        f"{base}; {floor}. excess = (E - e_min)/T in kT over finite rows, E = clip(U + lock) + "
        f"wall at T = 1 (kcal/mol); the floor is an upper bound on the minimum, so each excess is "
        f"a lower bound. T_eff/T = 1 + 2 median(excess)/k (thermal_stats), a ratio: 2.0 is its "
        f"reading for a harmonic well sampled at the target T; it is a proxy, not a population "
        f"check. within k/2 = fraction with excess <= k/2 (0.5 in that well). nonthermal = "
        f"fraction with excess above u* = s k, s = {meta['nonthermal_entropy_per_dim']} (the "
        f"run's nonthermal_entropy_per_dim).",
        ['condition', 'e_min (kcal/mol)', 'excess p50 (kT)', 'excess p90 (kT)',
         'excess p99 (kT)', 'excess max (kT)', 'below floor (rows)', 'T_eff/T (ratio)',
         'within k/2 (frac)', 'nonthermal (frac)'],
        [[r['condition'], _g(r['excess'].get('e_min'), '{:.3f}'),
          _g(r['excess'].get('p50'), '{:.2f}'),
          _g(r['excess'].get('p90'), '{:.2f}'), _g(r['excess'].get('p99'), '{:.2f}'),
          _g(r['excess'].get('max'), '{:.2f}'), _g(r['excess'].get('n_below_floor'), '{:d}'),
          _g(r['phys'].get('E/T_eff_over_T'), '{:.3f}'),
          _g(r['phys'].get('E/frac_within_equipartition'), '{:.3f}'),
          _g(r['phys'].get('thermal/nonthermal_frac'), '{:.4f}')] for r in rows]))
    out.append(T(
        'Table 2. COVERAGE of the target\'s rotamer basins, per condition.',
        f"{base}; {floor}. kind joint: basins = products of per-group rotamer centres from 1-D "
        f"scans of the true energy (basin_reference); accessible = within 10 kT of the best "
        f"basin with every other coordinate at the reference geometry; visited/missed count "
        f"accessible basins with and without a draw (bar: missed <= {b['max_n_missed']}). kind "
        f"marginal (the table skipped the product enumeration): the same counts over each "
        f"group's rotamer centres, every one within 3 kT of its group's best in 1-D; a lower "
        f"bound on missed basins. worst/uniform = the least-visited accessible basin's (or "
        f"centre's, within its group) share of the draws over the uniform share (1 = uniform, "
        f"0 = missed): visitation, not Boltzmann weight. occupancy entropy = Shannon entropy of "
        f"the accessible basins' shares over its maximum (1 = uniform, 0 = one basin). TC = "
        f"debiased total correlation of the group labels (nats) against the target's own; "
        f"suppressed = combinations the marginals predict at >= 10 draws that have none. "
        f"nonthermal worst basin = over occupied basins, the largest fraction of a basin's draws "
        f"with excess above u* (basin_nonthermal). '-' = unavailable (one group, one accessible "
        f"or one occupied basin, or kind marginal). free centres = non-planar centres the "
        f"stereo lock leaves free with a substituent offset to flip (parity_coverage), each "
        f"required on both sides; missed = those whose other side holds no draw (bar: <= "
        f"{b['max_parity_missed']}); minority side = the smaller side's share of the draws, "
        f"worst centre (the target's is 0.5 where the sides are mirror images).",
        ['condition', 'kind', 'rotor groups (count)', 'basins (count)', 'accessible (count)',
         'visited (count)', 'missed (count)', 'worst/uniform (ratio)',
         'occupancy entropy (0-1)', 'TC debiased (nats)', 'target TC (nats)',
         'suppressed (combos)', 'nonthermal worst basin (frac)', 'free centres (count)',
         'centres missed (count)', 'minority side (frac)'],
        [[r['condition'], r['coverage'].get('kind', '-'),
          _g(r['coverage'].get('n_groups'), '{:d}'),
          _g(r['coverage'].get('n_modes'), '{:d}'), _g(r['coverage'].get('n_accessible'), '{:d}'),
          _g(r['coverage'].get('n_visited'), '{:d}'), _g(r['coverage'].get('n_missed'), '{:d}'),
          _g(r['coverage'].get('worst_over_uniform'), '{:.3f}'),
          _g(r['coverage'].get('occupancy_entropy'), '{:.3f}'),
          _g(r['coverage'].get('tc_debiased'), '{:.4f}'),
          _g(r['coverage'].get('target_tc'), '{:.4f}'),
          _g(r['coverage'].get('n_suppressed'), '{:d}'),
          _g(r['coverage'].get('nonthermal_worst_basin'), '{:.4f}'),
          r['parity'].get('n_centres', 0),
          (f"{r['parity']['n_missed']} ({','.join(r['parity']['missed_names'])})"
           if r['parity'].get('n_missed') else _g(r['parity'].get('n_missed'), '{:d}')),
          _g(r['parity'].get('minority_frac'), '{:.3f}')] for r in rows]))
    out.append(T(
        'Table 3a. TB: the forward log-weights, per condition.',
        f"{base}. log w = log R + log P_B - log P_F (P_B "
        + ('the frozen snapshot restored from the checkpoint' if meta['pb_frozen']
           else "the eval model's own backward policy (no snapshot)") + "). "
        f"std = the within-condition spread; gap = log Z_IS - mean log w (>= 0 up to noise; 0 at "
        f"a perfect sampler), blank where the IS status is not ok; ESS/N Kish; max share = the "
        f"largest weight over the sum; Pareto k = GPD fit to the largest ceil(min(0.2N, 3 sqrt "
        f"N)) weights (bar < {b['max_pareto_k']:g}; < 0.5 finite variance). {status_rule}.",
        ['condition', 'mean log w (nats)', 'std log w (nats)', 'gap (nats)', 'ESS/N (frac)',
         'max share (frac)', 'Pareto k (shape)', 'tail (weights)', 'non-finite log w (rows)',
         'IS status'],
        [[r['condition'], _f(r['tb'].get('mean_log_w')), _g(r['tb'].get('std_log_w'), '{:.4f}'),
          _g(_gap(r, r['tb'].get('kl_gap')), '{:.4f}'), _g(r['tb'].get('ess_frac'), '{:.3e}'),
          _g(r['tb'].get('w_max_frac'), '{:.3f}'), _g(r['tb'].get('pareto_k'), '{:.3f}'),
          r['tb'].get('pareto_m_tail'), r['tb'].get('n_nonfinite'), r['is_status']]
         for r in rows]))
    out.append(T(
        'Table 3b. TB: IS log Z(c) against exact quadrature.',
        f"{base}. IS = logmeanexp(log w), printed on every row with its status; SE = bootstrap "
        f"over {meta['n_boot']} resamples (the delta-method SE beside it), neither describing "
        f"the error of a row whose IS status is not ok. exact = conformer_logz_check's two-grid "
        f"quadrature, grid {meta['grids'][0]} checked by {meta['grids'][1]} within "
        f"{meta['quad_tol']:g} nats (|A-B|), only where k <= {meta['quadrature_max_d']}. check "
        f"= conformer_logz_check's exact_verdict: the floor's name when a floor fails, "
        f"UNCONVERGED, else PASS when |IS - exact| <= bar = max({b['k_se']:g} SE + |A-B|, "
        f"{b['abs_tol']:g}) and FAIL otherwise. {status_rule}.",
        ['condition', 'IS log Z (nats)', 'IS status', 'SE boot (nats)', 'SE delta (nats)',
         'ESS (draws)', 'exact log Z (nats)', '|A-B| (nats)', 'IS - exact (nats)', 'bar (nats)',
         'check'],
        [[r['condition'], _f(r['tb'].get('log_z')), r['is_status'],
          _g(r['tb'].get('se_boot'), '{:.4f}'),
          _g(r['tb'].get('se_delta'), '{:.4f}'), _g(r['tb'].get('ess'), '{:.1f}'),
          _f((r['exact'] or {}).get('log_z')), _g((r['exact'] or {}).get('diff'), '{:.1e}'),
          _f(r['tb']['log_z'] - (r['exact'] or {}).get('log_z', float('nan'))),
          _g((r['exact_check'] or {}).get('bar'), '{:.4f}'),
          (r['exact_check'] or {}).get('verdict', '-')] for r in rows]))
    out.append(T(
        'Table 3c. TB: IS log Z(c) against the tracker and the flow head.',
        f"{base}. ema_logw is what the checkpoint's tracker serves TB (a Jensen lower bound "
        f"under a forward-only feed, a bound on neither side once backward rows feed it); "
        f"ema_log_z_emp is its logmeanexp EMA; neither is a validated log Z, so IS - tracker is a "
        f"gap, not an error. visits = the tracker's lifetime sample count, trusted = its lookup "
        f"mask. head = the learned log Z(c), log_flow[:, 0] averaged over the draws. The three "
        f"gaps are blank on a row whose IS status is not ok (Table 3b): there the IS log Z is not "
        f"a reference to measure against. {status_rule}.",
        ['condition', 'IS log Z (nats)', 'IS status', 'ema_logw (nats)',
         'IS - ema_logw (nats)', 'ema_log_z_emp (nats)', 'IS - emp (nats)', 'visits (samples)',
         'trusted', 'head (nats)', 'head - IS (nats)'],
        [[r['condition'], _f(r['tb'].get('log_z')), r['is_status'],
          _f(r['tracker']['ema_logw']), _f(_gap(r, r['tb']['log_z'] - r['tracker']['ema_logw'])),
          _f(r['tracker']['ema_log_z_emp']),
          _f(_gap(r, r['tb']['log_z'] - r['tracker']['ema_log_z_emp'])),
          '-' if r['tracker']['visits'] is None else r['tracker']['visits'],
          '-' if r['tracker']['trusted'] is None else r['tracker']['trusted'],
          _f(r['tb'].get('head')), _f(_gap(r, r['tb'].get('head') - r['tb']['log_z']))]
         for r in rows]))
    if res['pairs']:
        out.append(T(
            'Table 4. Conditions of one constitution.',
            f"delta = IS_a - IS_b; SE = sqrt(SE_a^2 + SE_b^2) (bootstrap); a mirror pair and two "
            f"identifiers of one isomer must share log Z: PASS needs |delta| <= max({b['k_se']:g} "
            f"SE, {b['abs_tol']:g}) with both rows above the ESS floors, and a side at "
            f"{PARETO_K_ABOVE_BAR} makes the pair {PARETO_K_ABOVE_BAR}. Diastereomers have "
            f"different targets and carry no requirement; their delta is shown.",
            ['pair (a | b)', 'kind', 'delta IS (nats)', 'SE (nats)', '|delta|/SE (ratio)',
             'verdict', 'required', 'delta ema_logw (nats)', 'delta exact (nats)'],
            [[f"{p['a']} | {p['b']}", p['kind'], _f(p['delta']), _g(p['se'], '{:.4f}'),
              _g(p['z'], '{:.2f}'), p['verdict'], 'yes' if p['required'] else 'no',
              _f(p['tracker_delta']), _f(p['exact_delta'])] for p in res['pairs']]))
    else:
        out.append('Table 4. Conditions of one constitution: none in this set (no stereoisomer '
                   'or mirror pair to compare).')
    names = ['finite', 'in_box', 'lock', 'clash', 'coverage', 'parity', 'ess_floor',
             'pareto_k', 'exact_logz']
    out.append(T(
        'Table 5. Verdicts per condition against the bars.',
        f"Bars (WORKING ASSUMPTIONS, set by options): finite >= {b['min_finite_frac']:g}; in-box "
        f">= {b['min_in_box_frac']:g}; lock-active < {b['max_lock_active_frac']:g}; clash <= "
        f"{b['max_clash_frac']:g} at overlap {b['clash_overlap']:g} sigma; missed basins <= "
        f"{b['max_n_missed']} (joint, or marginal where Table 2 says so); missed free centres <= "
        f"{b['max_parity_missed']}; ess_floor = ESS >= {b['min_ess']:g}, ESS/N >= "
        f"{b['min_ess_frac']:g} and no non-finite log w; Pareto k < {b['max_pareto_k']:g}; "
        f"|IS - exact| <= max({b['k_se']:g} SE + |A-B|, {b['abs_tol']:g}) after the floors. n/a "
        f"= the quantity does not exist for the molecule; UNAVAILABLE = its input is missing, "
        f"and fails.",
        ['condition'] + names + ['overall'],
        [[r['condition']] + [r['verdicts'][k]['status'] for k in names] + [r['overall']]
         for r in rows]))
    summ = [
        ('finite', 'frac', lambda r: r['phys'].get('phys/finite_frac'), 'min'),
        ('in-box', 'frac', lambda r: r['phys'].get('geom/all_in_range'), 'min'),
        ('lock-active', 'frac', lambda r: r['lock'].get('active_frac'), 'max'),
        ('clash', 'frac', lambda r: r['clash'].get('frac'), 'max'),
        ('excess p50', 'kT', lambda r: r['excess'].get('p50'), 'max'),
        ('T_eff/T', 'ratio', lambda r: r['phys'].get('E/T_eff_over_T'), 'max'),
        ('missed basins', 'count', lambda r: r['coverage'].get('n_missed'), 'max'),
        ('worst basin / uniform', 'ratio',
         lambda r: r['coverage'].get('worst_over_uniform'), 'min'),
        ('missed free centres', 'count', lambda r: r['parity'].get('n_missed'), 'max'),
        ('std log w', 'nats', lambda r: r['tb'].get('std_log_w'), 'max'),
        ('ESS/N', 'frac', lambda r: r['tb'].get('ess_frac'), 'min'),
        ('Pareto k', 'shape', lambda r: r['tb'].get('pareto_k'), 'max'),
        ('|IS - exact|', 'nats', lambda r: (_gap(r, abs(r['tb']['log_z'] - r['exact']['log_z']))
                                            if r['exact'] else None), 'max'),
        ('|IS - ema_logw|', 'nats',
         lambda r: _gap(r, abs(r['tb']['log_z'] - r['tracker']['ema_logw'])), 'max'),
    ]
    body = []
    for name, unit, get, worst in summ:
        med, wv, wc = _median_worst(rows, get, worst)
        body.append([f'{name} ({unit})', f'{worst} is worst', _g(med, '{:.4g}'), _g(wv, '{:.4g}'),
                     wc or '-'])
    out.append(T(
        'Table 6. Median and worst condition per quantity.',
        f"Over the {len(rows)} condition(s) of Tables 1-3; the worst is the minimum or the "
        f"maximum as stated, and 'all tied' or 'tie' when more than one condition has it; "
        f"conditions without the quantity are left out, and the two IS gaps are left out on "
        f"rows whose IS status is not ok.",
        ['quantity (unit)', 'direction', 'median', 'worst', 'worst condition'], body))
    c = res['cost']
    tot = sum(c.values()) or 1.0
    order = sorted(c.items(), key=lambda kv: -kv[1])
    out.append(T(
        'Table 7. Cost by phase.',
        f"Wall seconds on device {meta['device']}, {len(rows)} condition(s) x N = {meta['n']}. "
        f"setup = config, energy, checkpoint load and condition sets; references = the floor and "
        f"basin table (load and re-score, or build); sampling = P_F/P_B rollouts; energy = log R "
        f"of the terminals; sample metrics = the per-molecule block (energy components, geometry, "
        f"basin coverage and coupling) and the parity check; quadrature includes the integrand "
        f"check and reads the cache when the grid was computed before. Dominant phase: "
        f"{order[0][0]}.",
        ['phase', 'wall (s)', 'share (frac of total)'],
        [[k, f'{v:.1f}', f'{v / tot:.1%}'] for k, v in order] + [['total', f'{tot:.1f}', '']]))
    notes = [f"  {r['condition']}: {'; '.join(r['notes'])}" for r in rows if r['notes']]
    if meta.get('eval_T') is not None and meta.get('train_steps') is not None \
            and meta['eval_T'] != meta['train_steps']:
        notes.append(f"  (all): rollouts on eval_T = {meta['eval_T']} steps; the policy trained "
                     f"on integrator.T = {meta['train_steps']}")
    if meta.get('device_swap') and meta['device_swap'][0] != meta['device_swap'][1]:
        notes.append(f"  (all): checkpoint gfn_config device {meta['device_swap'][0]!r} built on "
                     f"{meta['device_swap'][1]!r}")
    ex = meta.get('identity_exempt') or {}
    if ex:
        notes.append(
            "  (all): energy_config keys the checkpoint's problem identity exempts "
            "(utils._NON_IDENTITY_ENERGY_CONFIG_KEYS) and the members read: "
            + ', '.join(f'{k} {v}' for k, v in ex.items())
            + ". These are the config's values; the checkpoint records none of them, so a config "
              "that changed one loads without refusal, and log R, the quadrature and the "
              "tracker and head gaps then describe another target than the one trained")
    if meta.get('prior_rows'):
        notes.append(f"  (all): the prior dataset (read by nothing here; its identifiers enter "
                     f"the condition registry) was drawn at prior_sample_size "
                     f"{meta['prior_rows'][1]} in place of the config's {meta['prior_rows'][0]}, "
                     f"at least 2 rows per member")
    rf_missing = meta['refs'].get('missing')
    if rf_missing:
        notes.append(f"  (all): no reference entry for {rf_missing}")
    notes.append(f"  (all): conditions file {meta['conditions_file']} sha256 "
                 f"{(meta['conditions_sha256'] or '?')[:16]}...; a checkpoint whose stamp names "
                 f"its condition set was held to this file's identifiers and member signatures "
                 f"at load (ConformerModeller._assert_condition_set); one written before that "
                 f"stamp records no identifiers, so its identity beyond layout, count and "
                 f"stereo coefficient is this file's")
    out.append('Notes:\n' + '\n'.join(notes))
    n_fail = sum(1 for r in rows if r['overall'] != PASS)
    n_pair = sum(1 for p in res['pairs'] if p['required'] and p['verdict'] != 'PASS')
    out.append(f"verdict: {len(rows) - n_fail} of {len(rows)} condition(s) pass every bar"
               + (f"; {n_pair} required pair(s) fail" if n_pair else '')
               + (f"; failing: {', '.join(r['condition'] for r in rows if r['overall'] != PASS)}"
                  if n_fail else ''))
    return '\n\n'.join(out)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return _jsonable(o.tolist())
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ('inf' if o > 0 else '-inf')
    return o


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help="the run's own config YAML")
    ap.add_argument('--checkpoint', required=True, help='checkpoint .pt (read-only)')
    ap.add_argument('--n-per-condition', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=2048, help='rollouts per forward pass')
    ap.add_argument('--refs', default=None,
                    help='reference table from build_conformer_references.py (default: build '
                         'one with EVAL_SEARCH into the cache)')
    ap.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='directory for report.txt and eval.json')
    ap.add_argument('--cache-dir', default=None,
                    help='references and quadrature cache (default: <out>/cache)')
    ap.add_argument('--conditions', nargs='*', default=None,
                    help='identifiers to evaluate (default: every condition row)')
    ap.add_argument('--quadrature-max-d', type=int, default=6)
    ap.add_argument('--quad-grids', nargs=2, default=list(lzc.DEFAULT_GRIDS),
                    type=lzc.GridSpec.parse, metavar=('A', 'B'))
    ap.add_argument('--quad-tol', type=float, default=0.02)
    ap.add_argument('--n-boot', type=int, default=200)
    ap.add_argument('--threads', type=int, default=4, help='torch CPU threads')
    d = Bars()
    for f, v in asdict(d).items():
        ap.add_argument('--' + f.replace('_', '-'), type=type(v), default=v,
                        help=f'bar (working assumption), default {v}')
    a = ap.parse_args(argv)
    try:
        return _run(a)
    except Refused as e:
        print(f'refused: {e}', file=sys.stderr)
        return EXIT_REFUSED


def _run(a) -> int:
    lzc.gpu_preflight(a.device, a.config)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    torch.set_default_dtype(torch.float32)          # the route's dtype, before the config
    if a.device == 'cpu':
        torch.set_num_threads(int(a.threads))
    out = Path(a.out) if a.out else None
    cache_dir = Path(a.cache_dir) if a.cache_dir else (
        out / 'cache' if out else Path(tempfile.gettempdir()) / 'conformer_model_eval_cache')
    bars = Bars(**{f: getattr(a, f) for f in asdict(Bars())})
    cost = Cost()
    with cost('setup'):
        m = init_eval_modeller(build_eval_modeller(a.config, a.checkpoint, a.device))
    if a.conditions:
        rows_of = lzc._condition_rows(m)
        unknown = [c for c in a.conditions if c not in rows_of]
        if unknown:
            raise Refused(f'no condition row for {unknown}; the run holds {list(rows_of)}')
    with cost('references'):
        table, info = load_or_build_references(m, a.refs, cache_dir, identifiers=a.conditions)
    res = evaluate(m, n=a.n_per_condition, batch=a.batch, seed=a.seed,
                   conditions=a.conditions, refs=table, refs_info=info, bars=bars,
                   quadrature_max_d=a.quadrature_max_d, grids=a.quad_grids,
                   quad_tol=a.quad_tol, n_boot=a.n_boot, cache_dir=cache_dir, cost=cost)
    res['exit_status'] = exit_status(res)
    text = format_report(res)
    print(text, flush=True)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.txt').write_text(text + '\n', encoding='utf-8')
        (out / 'eval.json').write_text(json.dumps(_jsonable(res), indent=1), encoding='utf-8')
        print(f'wrote {out / "report.txt"} and {out / "eval.json"}')
    return res['exit_status']


if __name__ == '__main__':
    sys.exit(main())
