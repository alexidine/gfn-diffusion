"""Does compiling the MLIP pay, and what do the other two execution knobs cost?

WHY THIS IS WORTH AN ARM. On every-step arms the MLIP is not a detail, it is most of
the run: measured over whole runs from `energy/seconds`, p02_nehu_lr0p125 spent 73.5
of its 142.8 hours inside the MLIP (51.5%), p02_acr_lr0p05 21.0 of 48.0 (43.7%), and
p02_mipu_lr0p0625 16.4 of 41.8 (39.3%). A 30% MLIP speedup is therefore ~12-15% of an
entire paper arm, and none of these knobs has ever been measured -- `compile` has a
full written analysis in init_uma_crystal_predictor and has never been switched on,
because nothing passed it through.

NB THIS DOES NOT TRANSFER TO RARE-ROLLOUT ARMS, where the same measurement gives 1.4%
(rr08_mipu_b3200, 9 h). Amortising the rollout removed the MLIP; that is why those
arms lost occupancy, and it is also why compiling the MLIP buys them nothing. The
prize is here, on the every-step shape, which is what the paper models run.

THE THREE KNOBS, and what can go wrong with each quietly:
  compile                       -- fuses the model. Every distinct edge count is a
                                   recompile, so without a chunk size it can blow
                                   dynamo's cache and fall back to EAGER SILENTLY.
  edge_chunk_size               -- pads edges into buckets so the traced shape is
                                   fixed. NOT SET IN THIS WAVE: it has to clear the
                                   observed maximum and nobody has ever logged that.
                                   The ctrl arm now reports `energy/uma_edges_max`
                                   (added for this), which is what a second wave sets
                                   it from. Guessing it low is the silent failure
                                   above; guessing high just pads.
  activation_checkpointing      -- off trades memory for a saved forward. Inert on
                                   the training reward here (reward_grads is 0, so
                                   the call runs under no_grad) but LOAD-BEARING for
                                   the init-time prior re-analysis, which runs with
                                   the graph live. That pass is also where this route
                                   has OOM'd before, so `ckoff` may die at startup --
                                   which is a fast, cheap answer, not a surprise.

READ `energy/ms_per_sample`, not the step time: it is the MLIP's own cost per scored
crystal and is not diluted by whatever else the step is doing. `energy/seconds` over
the run is the number that matters for the arm; ms_per_sample is the one that moves
first and settles fastest.

⚠ AND WATCH UTILIZATION. A faster MLIP REMOVES GPU-busy time, and mipu already runs
at 62-66% against a ~54% two-hour kill line. A successful compile could push an arm
toward cancellation while making it faster -- the same trade rare rollouts made. The
counter-effect is that the UMA profile showed 28k `Command Buffer Full` stalls (the
driver blocking the host), and fusing should cut those too, which pushes the other
way. Direction genuinely unknown; that is part of what this measures.

One base, one axis, four cells. mipu rather than nehu because its step is half as
long, so a 1 h wall buys twice the reports; nehu has the bigger prize and is the
follow-up if this pays.
"""
import copy, pathlib, yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'prod_sep02' / 'p02_mipu_lr0p0625.yaml'
TAG = 'mlipc09'

#: name -> the energy_config overrides. Everything else is the committed p02 arm.
ARMS = (
    ('ctrl',        {}),
    ('ckoff',       {'mlip_activation_checkpointing': False}),
    ('comp',        {'mlip_compile': True}),
    ('comp_ckoff',  {'mlip_compile': True, 'mlip_activation_checkpointing': False}),
)


def deltas(cfg, name, ov):
    cfg['run_name'] = 'mlipc_' + name
    cfg['tag'] = TAG
    # the wall ends the job; `epochs` is ABSOLUTE and this resumes at 5010
    cfg['epochs'] = 500000
    # eager, like every other battery since the compile default flipped -- the POLICY
    # compile is a measured loss, and leaving it on would put a second compiled domain
    # in the same dynamo cache the MLIP compile needs.
    cfg['compile_policy'] = False
    ec = cfg['energy_config']
    for k in ('mlip_compile', 'mlip_edge_chunk_size', 'mlip_activation_checkpointing'):
        ec.setdefault(k, {'mlip_activation_checkpointing': True}.get(k, False) if k !=
                      'mlip_edge_chunk_size' else None)
    ec.update(ov)
    return cfg


def check(cfg, name, ov):
    where = name + ': '
    ec = cfg['energy_config']
    assert cfg['energy_function'] == 'uma', where + 'these knobs are uma-only'
    assert cfg['compile_policy'] is False, where + 'policy must be eager'
    # the knob under test must actually differ from the control, or the arm is a
    # duplicate written by omission
    assert (name == 'ctrl') == (not ov), where + 'ctrl carries no overrides; others must'
    for k, v in ov.items():
        assert ec[k] == v, where + 'override %s did not land' % k
    # reward_grads 0 is what makes activation_checkpointing inert on the TRAINING
    # reward; if a future base turns it on, ckoff stops being a free trade
    if ec.get('mlip_activation_checkpointing') is False:
        rg = float(cfg['fwd_loss_coeffs'].get('reward_grads', 0) or 0)
        assert rg == 0, where + ('activation_checkpointing=False needs reward_grads 0; '
                                 'this base has %s' % rg)


def main():
    rows = ['arm\tcompile\tedge_chunk\tact_ckpt']
    for name, ov in ARMS:
        cfg = deltas(copy.deepcopy(yaml.safe_load(BASE.read_text(encoding='utf-8'))), name, ov)
        check(cfg, name, ov)
        (HERE / ('mlipc_%s.yaml' % name)).write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        ec = cfg['energy_config']
        rows.append('%s\t%s\t%s\t%s' % (name, ec['mlip_compile'],
                                        ec['mlip_edge_chunk_size'],
                                        ec['mlip_activation_checkpointing']))
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print('mlipc_sep09: %d arms from %s' % (len(ARMS), BASE.name))
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
