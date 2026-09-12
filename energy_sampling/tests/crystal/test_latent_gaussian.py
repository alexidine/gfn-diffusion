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

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
# `_here` itself is in the list, and it is what makes the standalone invocation in
# the docstring work. `energy_sampling.energies` imports `profiling` by its bare
# name, so energy_sampling must be ON the path, not merely reachable as a package
# from its parent. Under pytest that comes free from pytest.ini's `pythonpath = . ..`,
# which is why the omission was invisible: the file ran under pytest and died on
# `ModuleNotFoundError: No module named 'profiling'` the moment anyone ran it the
# way its own docstring says to.
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), '..', 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402
from energy_sampling.energies.molecular_crystal import MolecularCrystal  # noqa: E402
from energy_sampling.models.gfn import GFN  # noqa: E402
from energy_sampling.models.dead_latent_rows import (  # noqa: E402
    free_centroid_rows, resolve_dead_rows)

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


def free(sg):
    """Dead rows that are FREE AXES (centroid rows) rather than clobbered angles."""
    return tuple(r for r in free_centroid_rows(int(sg), 1) if r in dead(sg))


def clobbered(sg):
    """Dead rows enforce_crystal_system overwrites -- the ones the gaussian is blind to."""
    return tuple(r for r in dead(sg) if r not in free(sg))


def analytic(sg, hold, k=K, width=WIDTH, temperature=T):
    """The closed form. See configs/gauss_aug12/spec.py for the derivation.

    THE TWO KINDS OF DEAD ROW DIVERGE ONCE THEY ARE LIVE. A clobbered angle
    round-trips to the canonical 0.0, so only the soft wall sees it. A free axis
    does not -- `latent_harmonic_en` reads `gauge_fix_free_axes=False` on purpose --
    so it is an ordinary gaussian dimension with no fictitious volume at all."""
    n_live = DIM - len(dead(sg))
    per_dim = 0.5 * math.log(2 * math.pi * temperature) + math.log(width)
    z = n_live * per_dim
    if not hold:
        z += len(clobbered(sg)) * math.log(2.0 + math.sqrt(math.pi / k))
        z += len(free(sg)) * per_dim
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
    """Record a check and print it. Does NOT raise, in either mode.

    That is not the same as being unable to fail. `_RESULTS` is read twice: by
    `main()` below, for the standalone run, and by the repo conftest's
    `pytest_pyfunc_call` wrapper, which fails the test that appended a False
    entry. Not raising here is what lets every check in a test run and print
    before the verdict is taken, so a partial failure stays readable."""
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")


# ------------------------------------------------------------------- tests
def test_the_registry_entry_puts_its_zeros_on_the_real_dead_rows():
    """`configs/problems.yaml`'s `latent_gaussian` entry carries the target's `c`
    as a literal list, and the zeros in it must land on the rows
    `resolve_dead_rows` actually reports for the space group the same entry
    declares. A literal cannot re-derive itself, and this is the one assertion
    about that entry that needs the resolver -- test_problems.py checks the rest
    from the YAML alone, without paying for torch.

    WHY A MISPLACED ZERO IS NOT COSMETIC: the crystal build clobbers dead rows
    back to 0.0, so a live row given c=0 is a dimension whose target the policy
    can reach but is scored away from, and a dead row given c=MODE is a
    dimension chasing a value the build discards. Both stay finite and plausible
    and both change log Z. `MODE = 0.5` exists precisely so the two are
    distinguishable at all.

    ASSERTS RATHER THAN `check()`: the reporting helper this module uses records
    into `_RESULTS`, which only `main()` reads, so a `check` that fails is
    invisible under pytest."""
    import yaml
    reg = yaml.safe_load(
        open(os.path.join(_here, 'configs', 'problems.yaml'), encoding='utf-8'))
    entry = reg['problems']['latent_gaussian']
    sgs = entry['space_groups']
    assert len(sgs) == 1, sgs
    c = entry['analyze_kwargs']['c']
    assert c == target_c(sgs[0]), (
        f"registry c={c}\nresolver c={target_c(sgs[0])} "
        f"(sg {sgs[0]}, dead rows {dead(sgs[0])})")
    assert entry['analyze_kwargs']['width'] == WIDTH
    print(f"\n0. registry c matches sg{sgs[0]} dead rows {dead(sgs[0])}")


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

    Perturb only the dead rows and watch which terms move.

      * a CLOBBERED ANGLE must move the gaussian by NOTHING -- the crystal build
        discards it, so latent_params reads back the canonical 0.0 -- and must move
        the TOTAL by exactly the bounding term the perturbation creates. If the
        first half failed the energy would be secretly 12-dimensional; if the second
        failed, that arm's fictitious volume would not be log(2 + sqrt(pi/k)) and
        the A/B would be measuring something else.
      * a FREE AXIS moves BOTH. `latent_harmonic_en` reads
        `latent_params(gauge_fix_free_axes=False)` on purpose, so nothing pins the
        row and the gaussian sees it exactly as emitted.

    THE SECOND CASE USED TO BE ASSERTED AS THE FIRST, which is how sg 4 and sg 1
    came to be scored against a closed form that gave every dead row the soft-wall
    term. It read as an invariance failure in the energy; it was the prediction
    that was wrong. See configs/gauss_aug12/spec.py.
    """
    print("\n3. dead-row perturbation: angles move bounding only, free axes move both")
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
        free_rows = free(sg)
        # THE OFFSET IS CHOSEN BY ROW KIND, and it has to be.
        #   clobbered angle -> OUTSIDE the box, so the bounding term is nonzero and
        #     measurable. Its own value never reaches the gaussian anyway.
        #   free axis -> INSIDE the box, because here the gaussian IS the prediction
        #     and it is computed from what the build reads back, not from what was
        #     emitted. A centroid row pushed to 1.7 reads back 1.0, so an out-of-box
        #     offset would test the build's clamp/wrap rule rather than this
        #     mechanism. That rule is real and is NOT characterised here.
        outside, inside = [1.7, -2.3, 3.1], [0.6, -0.4, 0.8]
        offs, n_out, n_in = {}, 0, 0
        for r in d:
            if r in free_rows:
                offs[r], n_in = inside[n_in], n_in + 1
            else:
                offs[r], n_out = outside[n_out], n_out + 1
        for r, o in offs.items():
            pert[:, r] = o

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

        # WHAT THE GAUSSIAN SHOULD MOVE BY: nothing from the clobbered angles, and
        # the full 0.5*((x-c)/w)^2 from each free axis, whose c is the canonical 0.0.
        # Predicted per row rather than asserted as zero, so the two kinds are
        # distinguished instead of averaged into one claim that fits neither.
        expect_gauss = sum(0.5 * (offs[r] / WIDTH) ** 2 for r in free_rows)
        g_base, g_pert = gauss_only(base), gauss_only(pert)
        dg = (g_pert - g_base).mean().item()
        label = ('gaussian term invariant' if not free_rows
                 else f'gaussian moves by its {len(free_rows)} free axis/axes')
        check(f"sg{sg} {label}", abs(dg - expect_gauss) < max(1e-4, 1e-4 * expect_gauss),
              f"measured {dg:.4f}  predicted {expect_gauss:.4f}")

        # ...and the total moves by the bounding term the perturbation creates PLUS
        # whatever the gaussian just moved by. Bounding reads raw_latents, so it sees
        # the emitted offsets for every row, free or not.
        expect = (K * sum(max(abs(o) - 1.0, 0.0) ** 2 for o in offs.values())
                  + expect_gauss)
        got = (e_pert - e_base).mean().item()
        check(f"sg{sg} total moves by bounding + gaussian",
              abs(got - expect) < max(2e-3, 1e-4 * abs(expect)),
              f"measured {got:.6f}  predicted {expect:.6f}")


def test_analytic_log_z():
    """
    Importance sampling with an ANALYTIC proposal against the real energy object.
    On live rows the proposal IS the target, so Var(log w) ~ 0 there and a few
    thousand draws pin log Z to ~0.01 nats. Deliberately does NOT use the policy:
    an untrained P_F on a sigma-0.1 target has enormous weight variance, which would
    measure convergence rather than correctness (feedback: never certify log Z from
    a trained comparison).

    THE PROPOSAL FOR A LIVE-BUT-DEAD ROW DEPENDS ON WHICH KIND IT IS, for the same
    reason the closed form does. On a clobbered angle the target is the soft-wall
    box, so a wide N(0, 1.2) covers it. On a FREE AXIS the target is the narrow
    N(0, w) the gaussian actually applies, and proposing 1.2 against sigma 0.1 gave
    Var(log w) ~ 1.2e3 -- a standard error of sqrt(1230/20000) = 0.25 nats against a
    0.05 bar, i.e. an estimate that could not meet the bar however right the value
    was. Proposing the target collapses it, which is what makes a tight bar honest
    here rather than lucky.
    """
    print("\n4. analytic log Z, real energy, both arms")
    n_draw, batch = 20000, 500
    for sg in SGS:
        d = dead(sg)
        free_rows = free(sg)
        ef = energy_fn(sg)
        for hold in (True, False):
            live_dead = () if hold else d
            g = torch.Generator().manual_seed(11)
            mean = torch.full((DIM,), MODE)
            std = torch.full((DIM,), WIDTH * math.sqrt(T))
            for r in live_dead:
                mean[r] = 0.0
                std[r] = WIDTH * math.sqrt(T) if r in free_rows else 1.2
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
