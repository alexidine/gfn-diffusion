"""
CPU tests for the `latent_gaussian` toy energy and its interaction with D33
(dead latent rows held out of the SDE). Companion to test_dead_latent_rows.py
(layer invariants, bitwise) and test_dead_latent_rows_deep.py (statistical).

WHAT ONLY THIS FILE COVERS. The other two suites use a synthetic gaussian target
built inside the test. This one drives the REAL MolecularCrystal energy object
through instantiate_crystals / latent_to_cell_params / analyze, on a real
molecule, at real space groups -- the path a training run takes. That is where
the config-shaped failures live: a `c` on the wrong row, a jacobian that should
not be there, a reduction penalty contaminating an analytic target.

WHY IT MATTERS THAT IT IS ANALYTIC. Every assertion below compares against a
closed form, not against another run:

    rows HELD:  log Z = (n_live/2) log(2 pi T) + n_live log w
    rows LIVE:  log Z = <above> + n_dead * log(2 + sqrt(pi/k))

The second term is the fictitious volume D33 removes. It is NOT n_dead*log 2 --
the box wall is soft, so the reachable volume per live-but-dead row is
2 + sqrt(pi/k), which at k = 1 is 3.77 against a box of 2.

    python test_latent_gaussian.py
"""
import math
import os
import sys

import torch

CPU = torch.device('cpu')

_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(_here), os.path.join(os.path.dirname(_here), '..', 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402
from energy_sampling.energies.molecular_crystal import MolecularCrystal  # noqa: E402
from energy_sampling.models.gfn import GFN  # noqa: E402
from energy_sampling.models.dead_latent_rows import resolve_dead_rows  # noqa: E402

DATASET = os.path.abspath(os.path.join(_here, '..', '..', 'mxtaltools',
                                       'mini_datasets', 'mini_new_csd.pt'))
T = 1.0
WIDTH = 0.1
MODE = 0.5
K = 1.0          # bounding_coeff
TRAJ = 10        # integrator.T, matching the battery arms
DIM = 12
SGS = (2, 14, 19, 4, 1)


# ------------------------------------------------------------------ helpers
def dead(sg):
    return tuple(resolve_dead_rows(int(sg), is_crystal=True, max_z_prime=1))


def target_c(sg):
    c = [MODE] * DIM
    for r in dead(sg):
        c[r] = 0.0
    return c


def analytic(sg, hold, k=K, width=WIDTH, temperature=T):
    n_dead = len(dead(sg))
    n_live = DIM - n_dead
    z = (n_live / 2) * math.log(2 * math.pi * temperature) + n_live * math.log(width)
    if not hold:
        z += n_dead * math.log(2.0 + math.sqrt(math.pi / k))
    return z


_MOL = None


def mol_batch(sg, n):
    """One real molecule replicated. Nothing builds a cell, so identity is irrelevant."""
    global _MOL
    if _MOL is None:
        data = torch.load(DATASET, weights_only=False)
        cands = [e for e in data if int(e.z_prime) == 1 and bool(e.is_well_defined)
                 and not bool(e.cocrystal)]
        cands.sort(key=lambda e: int(e.num_nodes))
        _MOL = cands[0]
    b = collate_data_list([_MOL.clone() for _ in range(n)])
    b.reset_sg_info(int(sg))
    return b


def energy_fn(sg, k=K, width=WIDTH):
    return MolecularCrystal(
        device=CPU, energy_function='latent_gaussian',
        space_groups=[int(sg)], z_primes=(1,),
        temperature=T, bounding_coeff=k, reduction_coeff=1.0,
        reward_range=None, analyze_kwargs={'c': target_c(sg), 'width': width},
        internal_oom_recovery=False, host_gas_phase_reference=False)


def log_T_of(n):
    return torch.full((n,), math.log10(T))


def build_gfn(ef, hold, sg, seed=0):
    torch.manual_seed(seed)
    return GFN(dim=ef.data_ndim, s_emb_dim=64, conditions_dim=0, harmonics_dim=16,
               t_dim=16, t_hidden_dim=32, s_hidden_dim=32, s_layers=2,
               policy_hidden_dim=32, policy_layers=2, flow_hidden_dim=32, flow_layers=2,
               conditional=False, learn_pb=True, learned_variance=True,
               t_scale=0.05, log_var_range=6.0, pb_var_range=6.0, clipping=True,
               gfn_clip=200.0, device=CPU, max_z_prime=1,
               do_periodic_angles=ef.is_crystal,
               periodic_centroids=False, periodic_centroid_axes=None,
               hold_dead_latent_rows=bool(hold),
               dead_latent_rows=(dead(sg) if hold else None),
               dplr_rank=0, pb_exact_reversal=True)


class Fail(Exception):
    pass


_RESULTS = []


def check(name, ok, detail=''):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")


# ------------------------------------------------------------------- tests
def test_flags_and_computes():
    """
    The two-flag split is the whole design: is_crystal True (crystal layout, dead
    rows) AND latent_energy True (analytic reward, no physical terms). Either one
    wrong silently changes the target.
    """
    print("\n1. energy-object flags")
    for sg in SGS:
        ef = energy_fn(sg)
        ok = (ef.is_crystal and ef.latent_energy and not ef.computes_require_cluster
              and ef.data_ndim == DIM and ef.energy_clip is None)
        check(f"sg{sg} flags", ok,
              f"is_crystal={ef.is_crystal} latent_energy={ef.latent_energy} "
              f"cluster={ef.computes_require_cluster} clip={ef.energy_clip}")


def test_no_jacobian_no_reduction():
    """
    Both must be STRUCTURALLY absent for a latent-scored problem. The jacobian is a
    change of measure to physical space and would make the target gaussian * |J|,
    which has no closed form in box coordinates; the reduction penalty would
    contaminate it by ~1 nat on P-1. Structural, not a config knob, so no config can
    switch either back on by accident.
    """
    print("\n2. jacobian and reduction are structurally absent")
    for sg in SGS:
        ef = energy_fn(sg)
        mb = mol_batch(sg, 8)
        x = torch.full((8, DIM), MODE)
        for r in dead(sg):
            x[:, r] = 0.0
        crystal = ef.instantiate_crystals(x, mb)
        out = crystal.analyze(ef.computes, cutoff=10, supercell_size=10,
                              std_orientation=False, predictor=None,
                              c=target_c(sg), width=WIDTH)
        for kk in out:
            crystal.add_graph_attr(out[kk], kk)
        _, ens = ef.generator_energy(crystal, torch.ones(8), raw_latents=x)
        jac_keys = [kk for kk in ens if 'jacobian' in kk]
        check(f"sg{sg} no jacobian key", not jac_keys, f"found {jac_keys}")
        check(f"sg{sg} no reduction key", 'reduction_energy' not in ens,
              f"keys={sorted(ens)}")


def test_dead_rows_do_not_move_the_gaussian():
    """
    THE MECHANISM BEHIND THE WHOLE DELTA PREDICTION, checked directly.

    Perturb only the dead rows. The gaussian term must not move at all (the crystal
    build discards those rows, so latent_params reads back the canonical 0.0), while
    the TOTAL energy must move by exactly the bounding term the perturbation creates.
    If the first half failed, the energy would be secretly 12-dimensional; if the
    second half failed, the rows-live arm's fictitious volume would not be
    log(2 + sqrt(pi/k)) and the A/B would be measuring something else.
    """
    print("\n3. dead-row perturbation moves ONLY the bounding term")
    for sg in SGS:
        d = dead(sg)
        if not d:
            check(f"sg{sg} (no dead rows -- vacuous)", True, "")
            continue
        n = 16
        ef = energy_fn(sg)
        g = torch.Generator().manual_seed(3)
        base = MODE + WIDTH * torch.randn(n, DIM, generator=g)
        for r in d:
            base[:, r] = 0.0
        pert = base.clone()
        # deliberately OUTSIDE the box so the bounding term is nonzero and measurable
        offs = torch.tensor([1.7, -2.3, 0.6][:len(d)])
        for i, r in enumerate(d):
            pert[:, r] = offs[i]

        mb = mol_batch(sg, n)
        e_base = ef.energy(base.clone(), mb, log_T_of(n)).reshape(-1)
        e_pert = ef.energy(pert.clone(), mb, log_T_of(n)).reshape(-1)

        # the gaussian term alone, read off the batch attribute
        def gauss_only(x):
            crystal = ef.instantiate_crystals(x.clone(), mol_batch(sg, n))
            out = crystal.analyze(ef.computes, cutoff=10, supercell_size=10,
                                 std_orientation=False, predictor=None,
                                 c=target_c(sg), width=WIDTH)
            return out['latent_gaussian']

        g_base, g_pert = gauss_only(base), gauss_only(pert)
        dg = (g_pert - g_base).abs().max().item()
        check(f"sg{sg} gaussian term invariant", dg < 1e-4, f"max |dE_gauss| = {dg:.2e}")

        # and the total must move by exactly k * sum relu(|x|-1)^2 over the dead rows
        expect = K * sum(max(abs(float(o)) - 1.0, 0.0) ** 2 for o in offs)
        got = (e_pert - e_base).mean().item()
        check(f"sg{sg} total moves by the bounding term", abs(got - expect) < 2e-3,
              f"measured {got:.6f}  predicted {expect:.6f}")


def test_analytic_log_z():
    """
    Importance sampling with an ANALYTIC proposal against the real energy object.
    On live rows the proposal IS the target, so Var(log w) ~ 0 there and a few
    thousand draws pin log Z to ~0.01 nats. Deliberately does NOT use the policy:
    an untrained P_F on a sigma-0.1 target has enormous weight variance, which would
    measure convergence rather than correctness (feedback: never certify log Z from
    a trained comparison).
    """
    print("\n4. analytic log Z, real energy, both arms")
    n_draw, batch = 20000, 500
    for sg in SGS:
        d = dead(sg)
        ef = energy_fn(sg)
        for hold in (True, False):
            live_dead = () if hold else d
            g = torch.Generator().manual_seed(11)
            mean = torch.full((DIM,), MODE)
            std = torch.full((DIM,), WIDTH * math.sqrt(T))
            for r in live_dead:
                mean[r], std[r] = 0.0, 1.2
            held = [r for r in d if hold]

            logw = []
            done = 0
            while done < n_draw:
                nb = min(batch, n_draw - done)
                x = mean + std * torch.randn(nb, DIM, generator=g)
                for r in held:
                    x[:, r] = 0.0
                lq = torch.zeros(nb)
                for r in range(DIM):
                    if r in held:
                        continue
                    lq = lq - 0.5 * ((x[:, r] - mean[r]) / std[r]) ** 2 \
                         - math.log(std[r]) - 0.5 * math.log(2 * math.pi)
                lr = -ef.energy(x, mol_batch(sg, nb), log_T_of(nb)).reshape(-1)
                logw.append(lr - lq)
                done += nb
            logw = torch.cat(logw)
            z = torch.logsumexp(logw, 0).item() - math.log(logw.numel())
            want = analytic(sg, hold)
            check(f"sg{sg} {'HELD' if hold else 'LIVE'} log Z", abs(z - want) < 0.05,
                  f"measured {z:+.4f}  analytic {want:+.4f}  err {z - want:+.4f}  "
                  f"Var(log w) {logw.var().item():.2e}")


def test_bounding_coeff_dial():
    """
    The rows-live volume is a closed function of a CONFIG KNOB, so sweeping it tests
    the same claim along an axis the space group cannot reach -- and refutes the
    n_dead*log2 model, which would be flat here.
    """
    print("\n5. bounding_coeff dial (sg 19, n_dead = 3)")
    sg, n_draw, batch = 19, 20000, 500
    d = dead(sg)
    for k in (0.5, 2.0, 10.0):
        ef = energy_fn(sg, k=k)
        g = torch.Generator().manual_seed(11)
        mean = torch.full((DIM,), MODE)
        std = torch.full((DIM,), WIDTH * math.sqrt(T))
        for r in d:
            mean[r], std[r] = 0.0, 1.2
        logw, done = [], 0
        while done < n_draw:
            nb = min(batch, n_draw - done)
            x = mean + std * torch.randn(nb, DIM, generator=g)
            lq = torch.zeros(nb)
            for r in range(DIM):
                lq = lq - 0.5 * ((x[:, r] - mean[r]) / std[r]) ** 2 \
                     - math.log(std[r]) - 0.5 * math.log(2 * math.pi)
            logw.append(-ef.energy(x, mol_batch(sg, nb), log_T_of(nb)).reshape(-1) - lq)
            done += nb
        logw = torch.cat(logw)
        z = torch.logsumexp(logw, 0).item() - math.log(logw.numel())
        want = analytic(sg, hold=False, k=k)
        flat = analytic(sg, hold=True) + len(d) * math.log(2)
        check(f"k={k} rows-live log Z", abs(z - want) < 0.06,
              f"measured {z:+.4f}  soft-wall {want:+.4f}  err {z - want:+.4f}  "
              f"| log2 model {flat:+.4f} off by {z - flat:+.4f}")


def test_gfn_pins_dead_rows_through_real_rollouts():
    """
    Drive the REAL GFN over the REAL energy and assert the dead rows sit at the
    canonical value at EVERY timestep of both directions. The other suites check this
    on a synthetic target; here the terminal actually goes through
    latent_to_cell_params, which is what has to agree.
    """
    print("\n6. dead rows pinned at every step, fwd and bwd, real energy")
    from energy_sampling.utils import get_discretizer
    from types import SimpleNamespace
    n = 24
    for sg in SGS:
        d = dead(sg)
        ef = energy_fn(sg)
        gfn = build_gfn(ef, hold=True, sg=sg)
        disc = get_discretizer(SimpleNamespace(T=TRAJ))
        init = torch.zeros(n, DIM)
        states_f, *_ = _roll_fwd(gfn, init, disc)
        term = MODE + WIDTH * torch.randn(n, DIM, generator=torch.Generator().manual_seed(4))
        for r in d:
            term[:, r] = 0.0
        states_b, *_ = _roll_bwd(gfn, term, disc)
        if not d:
            check(f"sg{sg} (no dead rows -- vacuous)", True, "")
            continue
        wf = states_f[..., list(d)].abs().max().item()
        wb = states_b[..., list(d)].abs().max().item()
        check(f"sg{sg} fwd states pinned", wf < 1e-6, f"max |dead| = {wf:.2e}")
        check(f"sg{sg} bwd states pinned", wb < 1e-6, f"max |dead| = {wb:.2e}")


def _roll_fwd(gfn, init, disc):
    out = gfn.get_traj_fwd(init, disc, None, False, None)
    return _unpack(out, batch=init.shape[0])


def _roll_bwd(gfn, term, disc):
    out = gfn.get_traj_bwd(term, disc, False, None)
    return _unpack(out, batch=term.shape[0])


def _unpack(out, batch=None, dim=DIM, traj=TRAJ):
    """
    Pick the states tensor by SHAPE, [B, T+1, D], not by position -- and insist there
    is EXACTLY ONE match. A future signature change then surfaces as a failure here
    rather than silently handing back the wrong tensor, which would make every
    downstream assertion vacuous.
    """
    hits = [i for i, o in enumerate(out)
            if torch.is_tensor(o) and o.dim() == 3 and o.shape[-1] == dim
            and o.shape[1] == traj + 1
            and (batch is None or o.shape[0] == batch)]
    if len(hits) != 1:
        raise Fail(f'expected exactly one [B, {traj + 1}, {dim}] states tensor in a '
                   f'rollout output of len {len(out)}, found {len(hits)}')
    i = hits[0]
    return (out[i],) + tuple(x for j, x in enumerate(out) if j != i)


def test_logprobs_ignore_dead_perturbation():
    """
    Score one fixed trajectory, then move the dead rows of every intermediate state
    and re-score. log_pf/log_pb must not budge: they are restricted to live dims. A
    non-zero difference means a dead dim is still contributing to the TB loss, which
    is the defect D33 exists to remove.
    """
    print("\n7. log_pf / log_pb ignore dead-row perturbation")
    from energy_sampling.utils import get_discretizer
    from types import SimpleNamespace
    n = 16
    for sg in SGS:
        d = dead(sg)
        if not d:
            check(f"sg{sg} (no dead rows -- vacuous)", True, "")
            continue
        ef = energy_fn(sg)
        gfn = build_gfn(ef, hold=True, sg=sg)
        disc = get_discretizer(SimpleNamespace(T=TRAJ))
        term = MODE + WIDTH * torch.randn(n, DIM, generator=torch.Generator().manual_seed(6))
        for r in d:
            term[:, r] = 0.0
        s0, *rest0 = _roll_bwd(gfn, term, disc)
        traj = s0.clone()
        traj[..., list(d)] += 3.7        # move them far, and out of the box
        out = gfn.get_traj_replay(traj, disc, False, None)
        s1, *rest1 = _unpack(out, batch=n)
        out_ref = gfn.get_traj_replay(s0.clone(), disc, False, None)
        s2, *rest2 = _unpack(out_ref, batch=n)
        worst, compared = 0.0, 0
        for a, b in zip(rest1, rest2):
            if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape \
                    and a.dtype.is_floating_point:
                worst = max(worst, (a - b).abs().max().item())
                compared += 1
        # A comparison loop that finds NOTHING leaves worst at 0.0 and this assertion
        # would pass by vacuity -- the failure mode that makes a green suite worthless.
        # Require that real tensors were actually compared.
        check(f"sg{sg} replay log-probs invariant", worst < 1e-5 and compared >= 2,
              f"max |delta| = {worst:.2e} over {compared} tensors"
              + ("  <-- VACUOUS, nothing compared" if compared < 2 else ""))


def main():
    print(f"latent_gaussian CPU suite   T={T} width={WIDTH} mode={MODE} k={K}")
    print(f"fictitious volume per live-but-dead row = "
          f"{math.log(2 + math.sqrt(math.pi / K)):+.4f}  (log 2 = {math.log(2):+.4f})")
    for fn in (test_flags_and_computes,
               test_no_jacobian_no_reduction,
               test_dead_rows_do_not_move_the_gaussian,
               test_analytic_log_z,
               test_bounding_coeff_dial,
               test_gfn_pins_dead_rows_through_real_rollouts,
               test_logprobs_ignore_dead_perturbation):
        try:
            fn()
        except Exception as e:  # a broken test must not hide the passing ones
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            _RESULTS.append((fn.__name__, False, f'{type(e).__name__}: {e}'))

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 76)
    print(f"{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed")
    if n_fail:
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  FAIL {name}  {detail}")
    print("PASS" if n_fail == 0 else "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
