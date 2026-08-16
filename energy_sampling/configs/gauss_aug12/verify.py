"""
gauss_aug12 -- load-time verification. Run this BEFORE launching anything.

GENERATING A CONFIG IS NOT LOADING IT. Retired schema keys hard-raise on their
PRESENCE at load, not at first use, so a battery can generate ten clean-looking
YAMLs and then die at startup on every one of them. This module puts each arm
through the REAL loader (preflight_config -> resolve_derived_config), then through
the real Stage parser, then re-derives the dead rows and the predicted log Z from
the loaded object rather than from make.py's in-memory copy.

It also builds the actual energy function and GFN for one arm per space group and
checks the resolved dead rows, live_dim and expanded_dim -- the numbers a wrong
index set would move, and which no amount of YAML inspection can reach.

    python configs/gauss_aug12/verify.py
"""
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion',
          r'C:\Users\mikem\Projects\mxt_gfn\mxtaltools'):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(HERE))

import spec  # noqa: E402


def arm_files():
    return sorted(HERE.glob('*_sg*_o*.yaml'))


def check_loads():
    """Through the real loader, exactly as train.py does it."""
    from energy_sampling.utils import load_yaml, dict2namespace, preflight_config, \
        resolve_derived_config
    out = {}
    for f in arm_files():
        args = resolve_derived_config(preflight_config(dict2namespace(load_yaml(str(f)))))
        out[f.stem] = args
        print(f"  {f.stem:<14} loads OK   T={args.integrator.T} batch={args.batch_size} "
              f"hold={args.model.hold_dead_latent_rows} "
              f"periodic_centroids={args.model.periodic_centroids}")
    return out


def check_protocol(loaded):
    """
    Every stage must re-parse through the REAL Stage validator. StageProtocol needs a
    live modeller, so parse the stages directly -- which is the check that matters:
    Stage.__init__ is where unknown keys, bad train_modes and the balance block are
    validated, and a raise there kills an entire battery at startup.
    """
    from energy_sampling.protocol import Stage
    from energy_sampling.utils import load_yaml
    for f in arm_files():
        name = f.stem
        raw = load_yaml(str(f))     # raw dicts: Stage validates mappings, not namespaces
        stages = [Stage(s, i) for i, s in enumerate(raw['protocol']['stages'])]
        print(f"  {name:<14} protocol OK  stages={[s.name for s in stages]}  "
              f"modes={[s.train_mode for s in stages]}")


def check_target(loaded):
    """
    Re-derive `c` from the LOADED object. make.py asserts this too, but on its own
    in-memory dict -- this is the version the run will actually see, after yaml
    round-tripping and namespace conversion.
    """
    bad = 0
    for name, args in loaded.items():
        sg = int(args.space_groups[0])
        dead = spec.dead_rows(sg)
        ak = args.energy_config.analyze_kwargs
        c = ak.c if hasattr(ak, 'c') else ak['c']
        for r in range(spec.DIM):
            want = 0.0 if r in dead else spec.MODE
            if abs(float(c[r]) - want) > 1e-12:
                print(f"  FAIL {name}: c[{r}]={c[r]} expected {want} (dead={dead})")
                bad += 1
        if int(args.space_groups[0]) != sg:
            bad += 1
    if not bad:
        print(f"  all {len(loaded)} arms: c is MODE on live rows, 0.0 on dead rows")
    return bad


def check_model(loaded):
    """
    Build the real energy function and GFN for each arm and read back the numbers a
    wrong index set would move. This is the only check here that exercises code
    rather than configuration.
    """
    import torch
    from energy_sampling.energies.molecular_crystal import MolecularCrystal
    from energy_sampling.models.gfn import GFN
    from energy_sampling.models.dead_latent_rows import resolve_dead_rows

    bad = 0
    print(f"  {'arm':<14} {'dead resolved':<14} {'live':>5} {'expanded':>9} {'pred log Z':>11}")
    for name, args in sorted(loaded.items()):
        sg = int(args.space_groups[0])
        hold = bool(args.model.hold_dead_latent_rows)
        ak = args.energy_config.analyze_kwargs
        c = [float(v) for v in (ak.c if hasattr(ak, 'c') else ak['c'])]
        w = float(ak.width if hasattr(ak, 'width') else ak['width'])

        ef = MolecularCrystal(device=torch.device('cpu'),
                              energy_function=args.energy_function,
                              space_groups=[sg], z_primes=[1],
                              temperature=float(args.energy_config.temperature),
                              bounding_coeff=float(args.energy_config.bounding_coeff),
                              reduction_coeff=float(args.energy_config.reduction_coeff),
                              reward_range=args.energy_config.reward_range,
                              analyze_kwargs={'c': c, 'width': w},
                              internal_oom_recovery=False,
                              host_gas_phase_reference=False)
        if not ef.is_crystal:
            print(f"  FAIL {name}: is_crystal is False -- dead rows would be gated off")
            bad += 1
        if not ef.latent_energy:
            print(f"  FAIL {name}: latent_energy is False -- reduction/jacobian would be live")
            bad += 1
        if ef.computes_require_cluster:
            print(f"  FAIL {name}: requires a cluster; the toy must build nothing")
            bad += 1

        expect_dead = tuple(resolve_dead_rows(sg, is_crystal=True, max_z_prime=1)) if hold else ()
        # mirror train.py's _build_gfn_config exactly -- **vars(args.model) is what
        # carries hold_dead_latent_rows, dplr_*, pb_exact_reversal and the rest, so a
        # hand-written kwarg list here would be a second schema that can drift
        gfn = GFN(dim=ef.data_ndim,
                  conditions_dim=0,
                  conditions_type='vector',
                  periodic_centroid_axes=None,
                  dead_latent_rows=(expect_dead or None),
                  conditional=False,
                  device=torch.device('cpu'),
                  max_z_prime=1,
                  do_periodic_angles=ef.is_crystal,
                  **vars(args.model))
        # read dead_idx, the tensor the SDE actually indexes with -- not the constructor
        # argument. Those can differ (the kwarg is a request; dead_idx is what was applied)
        # and only the second one is evidence.
        got = tuple(sorted(int(v) for v in gfn.dead_idx.tolist()))
        live = int(gfn.live_dim)
        exp = int(gfn.expanded_dim)
        pred = spec.analytic_log_z(sg, hold)

        # the three-way partition invariant, re-checked per arm: 26 space groups have a
        # centroid axis that is BOTH free (dead) and auv == 1 (angular), so these sets
        # can only be trusted if they are disjoint and exhaustive
        parts = sorted(int(v) for v in
                       (gfn.ang_idx.tolist() + gfn.lin_idx.tolist() + gfn.dead_idx.tolist()))
        if parts != list(range(ef.data_ndim)):
            print(f"  FAIL {name}: ang|lin|dead is not a partition of range({ef.data_ndim}): {parts}")
            bad += 1
        if exp != int(gfn.lin_idx.numel()) + 2 * int(gfn.ang_idx.numel()):
            print(f"  FAIL {name}: expanded_dim {exp} != lin + 2*ang "
                  f"({int(gfn.lin_idx.numel())} + 2*{int(gfn.ang_idx.numel())})")
            bad += 1

        if got != tuple(sorted(expect_dead)):
            print(f"  FAIL {name}: gfn dead rows {got} != expected {tuple(sorted(expect_dead))}")
            bad += 1
        if live != ef.data_ndim - len(expect_dead):
            print(f"  FAIL {name}: live_dim {live} != {ef.data_ndim - len(expect_dead)}")
            bad += 1
        print(f"  {name:<14} {str(got) or '()':<14} {live:>5} {exp:>9} {pred:>11.4f}")
    return bad


def main():
    print(f"gauss_aug12 verification -- {len(arm_files())} arms\n")
    print("1. real loader (preflight_config -> resolve_derived_config)")
    loaded = check_loads()
    print("\n2. protocol re-parse")
    check_protocol(loaded)
    print("\n3. target consistency, from the LOADED object")
    bad = check_target(loaded)
    print("\n4. real energy function + GFN")
    bad += check_model(loaded)

    print("\n" + "=" * 72)
    print(f"predicted, for reading results against "
          f"(fictitious volume per live-but-dead row = "
          f"{math.log(2 + math.sqrt(math.pi / spec.BOUNDING_COEFF)):+.4f})")
    for sg, d, n_live, on, off, delta in spec.predictions():
        print(f"  sg{sg:<3} dead={str(d):<12} HELD {on:+9.4f}   LIVE {off:+9.4f}   "
              f"delta {delta:+.4f}")
    print("\n" + ("PASS" if bad == 0 else f"FAIL ({bad} problems)"))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
