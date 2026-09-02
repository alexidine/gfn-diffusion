"""SMOKE, not science. One arm per system, 20-minute wall.

Its only question is whether the production battery's SHAPE starts on the
cluster: the frozen-anchor keys against their just-pushed consumer, the armed
clip, fixed fractions with no ratio controller, mode fixed with burn_in_scale
pinned to fixed_scale, and the checkpoint glob. Every failure this is built to
catch lands in seconds -- before the MLIP ever initialises -- which is why a
20-minute wall is enough even for arms whose prior scan would take longer.

Each arm is an existing, cluster-VALIDATED battery config with the battery's
deltas applied, rather than a config regenerated from mk_dev. That is deliberate:
it keeps every path, problem hash and MLIP setting at a value already proven on
this cluster, so anything that breaks is attributable to the deltas.
"""
import copy, pathlib, yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# base, warm-src token, fixed_scale.  Scales are the battery's intended centre:
# ELJ 1.0 is the measured winner of a 4-rung fan at T=100; mipu 0.0625 and nehu
# 0.5 are the centres of their scans; acr 0.025 is one rung above acr_c's centre.
ARMS = {
    'smk_mip':  ('prod_t100_p2/pt100mip2_lr1p0.yaml',      'pt100_mip_lr4p0',  1.0),
    'smk_mipu': ('mipu_bwd_aug31/mb31_bwd25_lr1p56.yaml',  'pt100_mipu_lr4p0', 0.0625),
    'smk_nehu': ('uma_stab_aug30/us30_b_lr0p5.yaml',       'pt100_nehu_lr4p0', 0.5),
    'smk_acr':  ('acr_c_aug31/acrb3_b50_lr0p025.yaml',     'pt100_acr_lr4p0',  0.025),
}

ANCHOR = {'frozen': True, 'online_loss_flow': False,
          'thin_every_n_evals': 0, 'refresh_every_n_evals': 0, 'replay_beta': 1.0}
FRACS = {'fwd': 0.05, 'bwd': 0.475, 'replay': 0.475}


def deltas(cfg, name, scale):
    cfg['run_name'] = name
    cfg['tag'] = 'smoke02'
    cfg['checkpoint_name'] = 'WARM_CHECKPOINT_PLACEHOLDER'
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    # unreachable by construction: `epochs` is an ABSOLUTE step index and its
    # only consumer is trange(init_step, epochs+1), so the wall ends the job.
    cfg['epochs'] = 500000

    cfg.setdefault('buffers', {}).setdefault('anchor_buffer', {}).update(ANCHOR)

    lc = cfg.setdefault('lr_control', {})
    lc['mode'] = 'fixed'
    lc['fixed_scale'] = float(scale)
    # EQUAL, deliberately: lr_ctrl.scale is restored from a rewind target, so a
    # fire landing on a burn-in-era checkpoint would otherwise pin the arm at the
    # burn-in rate for the rest of the run. Equal rates also make _arm_cruise_bar
    # a no-op, so the bars stay fitted at the operating rate.
    lc['burn_in_scale'] = float(scale)
    lc['fire_cut_factor'] = 1.0
    lc['repeat_every'] = 0

    n = 0
    for proto in (cfg.get('protocols') or {}).values():
        for stage in (proto.get('stages') or []):
            for sensor in ('hot_lr_sensor',):
                if isinstance(stage.get(sensor), dict):
                    stage[sensor]['action'] = 'report'
            if stage.get('train_mode') == 'fused' and 'fracs' in stage:
                stage['fracs'] = dict(FRACS)
                # the ratio controller is disabled by ABSENCE: protocol.tick only
                # calls _balance_tick when stage.balance is not None.
                stage.pop('balance', None)
                stage.pop('min_fracs', None)
                n += 1
    assert n >= 1, f'{name}: found no fused stage to pin fractions on'
    return cfg


def main():
    (HERE / 'joblogs' / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n',
        encoding='utf-8')
    rows = []
    for name, (base, src, scale) in ARMS.items():
        cfg = yaml.safe_load((ROOT / base).read_text(encoding='utf-8'))
        cfg = deltas(cfg, name, scale)
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append((name, src, scale, base))

    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tscale\tbase\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f'{len(rows)} smoke arms written')
    for r in rows:
        print('   ', r[0], '<-', r[3], f'scale={r[2]}')


if __name__ == '__main__':
    main()
