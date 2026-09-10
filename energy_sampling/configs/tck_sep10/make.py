"""tck_sep10: does turning trajectory checkpointing OFF on bwd and replay buy a
faster step, and does it OOM when it does?

  tck_all   traj_checkpoint on every branch  -- today's shape, the control
  tck_fwd   traj_checkpoint_modes: [fwd]     -- bwd and replay stop recomputing

THE CLAIM UNDER TEST. Trajectory checkpointing trades ~33x activation memory for
recompute (33.6x measured at T=100) and is applied identically to fwd, bwd and
replay. But only fwd holds the energy function's footprint at the same time, and
under rarer rollouts fwd runs 1 step in N -- so at N=20, NINETEEN STEPS IN TWENTY
pay the recompute while sitting on headroom nothing is using. Turning it off for
those alone should be worth up to ~2x on the step, and nothing on the rollout.
The mechanism shipped in 05b1824 and has never been measured.

THE RISK IT DOES NOT REMOVE, and the reason this is an arm rather than a config
change: peak memory is set by the WORST step. A bwd step that allocates a large
block can leave the caching allocator without contiguous space for the NEXT
rollout's MLIP even though each step fits alone. Fragmentation of that kind
appears as an OOM at a rollout step, several thousand steps in, and no local test
reproduces it. Watch batch/oom_events and the rollout steps specifically.

THE BATCH IS PINNED, and that is the whole design. Freed memory is exactly what
the sizer converts into batch, so leaving it free would confound "the step got
faster" with "the batch got bigger" and let the two arms diverge into different
problems. Pinned, any step-time difference IS the recompute saving, and the freed
headroom reads directly off peak memory -- from which the batch the sizer could
have taken is predictable. 1600 is a MEASURED operating point for this family
(rr08_mipu_b1600), not a guess.

WHY mipu. uma is the slow route, its phase-1 exit exists, and rr08_mipu_b1600
gives a utilisation and step-time baseline at this exact batch. acr would confirm
across energy functions and is the follow-up if the effect is real.
"""
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
#: The KNOWN-GOOD phase-2 config, not mk_dev. Phase 2 needs the re-entry stub,
#: the full-resume wiring and the frozen buffer sidecar; rebuilding that from the
#: base config is how an arm ends up subtly not being phase 2 at all.
BASE = HERE.parent / 'prod_t100_p2' / 'pt100mipu2_lr1p0.yaml'

BATCH = 1600
WALL = '4:00:00'
#: enough to get a stable median step time and cross several rollouts; the wall
#: is what actually ends these, epochs only has to not bind first.
EXTRA_STEPS = 40000

ARMS = {
    'tck_all': None,        # traj_checkpoint_modes unset = every branch
    'tck_fwd': ['fwd'],
}


def build(name, modes):
    cfg = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    cfg['run_name'] = name
    cfg['tag'] = 'tck09'

    # -- the one knob under test -----------------------------------------------
    assert cfg['traj_checkpoint'] is True, 'the control must have it ON'
    if modes:
        cfg['traj_checkpoint_modes'] = list(modes)

    # -- pin the batch ---------------------------------------------------------
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    cfg['grow_batch_size'] = False
    # batch_util_target is the sizer's problem and the sizer is off; leaving 0.95
    # here would read as a live occupancy target that nothing can act on.
    cfg['batch_util_target'] = 0
    cfg['epochs'] = int(cfg['epochs']) + EXTRA_STEPS

    check(cfg, name, modes)
    return cfg


def check(cfg, name, modes):
    assert cfg['traj_checkpoint'] is True, name
    assert cfg.get('traj_checkpoint_modes') == (list(modes) if modes else None), name
    assert cfg['batch_size'] == cfg['max_batch_size'] == BATCH, name
    assert cfg['grow_batch_size'] is False, (
        name + ': a live sizer converts the freed memory into batch and the two '
               'arms stop being the same experiment')
    assert cfg['batch_util_target'] == 0, name
    # the phase-2 wiring this fork exists to inherit rather than rebuild
    assert cfg['load_weights_only'] is False, name + ': phase 2 needs the full resume'
    assert cfg['integrator']['T'] == cfg['eval_T'] == 100, name
    assert cfg['energy_function'] == 'uma', name
    stages = cfg['protocols'][cfg['protocol']]['stages']
    assert stages[0]['name'] == 'train_prior', name + ': the re-entry stub'
    assert any(s.get('train_mode') == 'fused' for s in stages), name


def main():
    for name, modes in ARMS.items():
        cfg = build(name, modes)
        (HERE / f'{name}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding='utf-8')
        print(f"{name:9s} traj_checkpoint_modes={str(modes):>8s}  batch={cfg['batch_size']} "
              f"(pinned)  epochs={cfg['epochs']}  T={cfg['integrator']['T']}")
    print(f'\nread: batch/med_step_s (the claim), peak memory (the headroom the '
          f'sizer\n      could take), batch/oom_events at ROLLOUT steps (the '
          f'fragmentation risk).')


if __name__ == '__main__':
    main()
