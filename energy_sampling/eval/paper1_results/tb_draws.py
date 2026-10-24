"""Fresh draws for the TB parity figure only: log P_F, log P_B (frozen snapshot installed by load_gfn), the exact
training reward log R and the learned log Z, with no distance matrix or landscape analysis.

    ONLY_RUN=<run_name> NUM=10000 python eval/paper1_results/tb_draws.py

Reads the run's entry in new_analysis.yaml (it must name `train_config`) and writes <results_dir>/tb/<run_name>_tb.pt,
which generate_figs.py reads for the parity figure when the landscape results predate the frozen-P_B fix.
"""
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from energy_sampling.eval.paper1_results.new_analysis import (   # noqa: E402
    load_gfn, load_predictor, sample_latents, decode_and_score, exact_log_rewards, resolve_molecule)
from energy_sampling.utils import load_yaml, dict2namespace   # noqa: E402

if __name__ == '__main__':
    raw_cfg = load_yaml(os.path.join(HERE, 'new_analysis.yaml'))
    run_name = os.environ['ONLY_RUN']
    run = [r for r in raw_cfg['runs'] if r['run_name'] == run_name]
    assert len(run) == 1, f"run {run_name} not found once in new_analysis.yaml"
    cfg = dict2namespace({**raw_cfg['base'], **run[0]})
    num = int(os.environ.get('NUM', 10_000))
    device = cfg.device
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_MEM_FRACTION', 0.9)), device=0)
    torch.manual_seed(int(os.environ.get('SEED', 0)))

    model, problem_def, train_T, prov = load_gfn(os.path.join(cfg.checkpoints_dir, cfg.checkpoint), device)
    assert prov.get('pb_frozen_installed') is not None, "load_gfn did not report whether the frozen P_B was installed"
    ef = problem_def['energy_function']
    sg_ind, zp = int(problem_def['space_groups'][0]), int(problem_def['z_primes'][0])
    prior_file = torch.load(cfg.dataset_path, map_location='cpu', weights_only=False)
    lj_coeff = float(prior_file.get('thermal_scaling_factor', 1.0))
    molecule = resolve_molecule(cfg, prior_file)
    predictor = load_predictor(ef, device)
    score_batch = int(os.environ.get('SCORE_BATCH', cfg.analysis_batch_size))
    latents, log_pf, log_pb, latents_raw = sample_latents(
        model, num, int(os.environ.get('SAMPLE_BATCH', cfg.sample_batch_size)), train_T, device)
    samples, kept = decode_and_score(latents, molecule, model.max_z_prime, sg_ind, zp, ef, lj_coeff, predictor,
                                     score_batch, device, bool(getattr(cfg, 'std_orientation', True)), score_batch,
                                     cfg.rdf_mode)
    latents_raw, log_pf, log_pb = latents_raw[kept], log_pf[kept], log_pb[kept]
    train_config = cfg.train_config
    if not os.path.isabs(train_config):   # relative to energy_sampling/
        train_config = os.path.join(os.path.dirname(os.path.dirname(HERE)), train_config)
    log_r, info = exact_log_rewards(samples, latents_raw, train_config, ef, lj_coeff, prior_file, device,
                                    stage=prov['stage'], step=prov['step_ind'],
                                    sched_entry=prov.get('coeff_sched_entry'))
    out = os.path.join(cfg.results_dir, 'tb')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{run_name}_tb.pt')
    torch.save({'log_pfs': log_pf, 'log_pbs': log_pb, 'log_r': log_r, 'reward_info': info,
                'learned_log_z': model.flow_model().item(), 'provenance': prov, 'checkpoint': cfg.checkpoint,
                'n': len(samples)}, path)
    resid = (log_pf + model.flow_model().item() - log_pb - log_r.to(log_pf.dtype))
    print(f"saved {path}: {len(samples)} draws, pb_frozen_installed {prov['pb_frozen_installed']}, "
          f"TB residual mean {float(resid.mean()):+.3f} sd {float(resid.std()):.3f} nats")
