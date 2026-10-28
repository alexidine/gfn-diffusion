"""
The likelihood of stored crystals under an archive, for molecules the run trained on and for
molecules it never saw: the held-out reading of a conditional MLE run.

For each archive the sampler is rebuilt with eval/cond_panel/sampler.py::load_run, and for
`--rows` crystals of each set `--paths` backward trajectories from the stored crystal are
drawn with P_B and scored as the trainer's backward step scores a dataset row,

    loss = -(sum log P_F - sum log P_B)      per trajectory (gflownet_losses.terminal_mle)

a bound on the negative log likelihood of the stored crystal, in nats. The two sets are

    training   rows of the run's own prior file (the config's prior_path, `equalized_prior`)
    unseen     rows of --unseen whose molecule is in no row of the training file (asserted)

each drawn once with a fixed seed and cached under --out by file name and row count, so every
archive scored into one --out directory is scored on the same crystals. The gap, unseen minus
training, is the overfit measure; the unseen column is the one to compare between runs.

The bound is P_B's as much as P_F's: a run whose P_B reads more (state features) can tighten
it without drawing better crystals. Read it beside the forward draws' excess energy, not alone.

    cd energy_sampling
    python -m eval.cond_panel.heldout_nll --config configs/qm9full_sep30/qf30_lad_c_s1.yaml \
        --unseen <priors>/qm9full_test_prior.pt --out <dir> <checkpoints>/<stem>_step20000.pt [...]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

from eval.cond_panel.sampler import load_run
from utils import uniform_discretizer

SEED = 20261007


def cached_rows(out, tag, path, n, exclude=None):
    """`n` rows of `path`'s equalized_prior, seeded, cached; `exclude`: identifiers no row may carry."""
    cache = os.path.join(out, f'nll_rows_{tag}_{n}.pt')
    if os.path.exists(cache):
        return torch.load(cache, map_location='cpu', weights_only=False)
    batch = torch.load(path, map_location='cpu', weights_only=False)['equalized_prior']
    order = torch.randperm(batch.num_graphs, generator=torch.Generator().manual_seed(SEED))
    if exclude is not None:
        order = torch.tensor([int(i) for i in order if batch.identifier[int(i)] not in exclude][:n])
    if order.numel() < n:
        raise ValueError(f'{path} holds {order.numel()} eligible rows; {n} were asked for')
    rows = batch.subsample_new_batch(order[:n])
    torch.save(rows, cache)
    return rows


@torch.no_grad()
def nll(run, rows, paths, batch):
    """[rows] the loss of each row, the mean over `paths` backward trajectories from its stored crystal."""
    gfn, ef = run.gfn, run.energy_function
    out = []
    for start in range(0, rows.num_graphs, batch):
        part = rows.subsample_new_batch(torch.arange(start, min(start + batch, rows.num_graphs)))
        acc = torch.zeros(part.num_graphs)
        for _ in range(paths):
            mol_batch = part.clone().to(run.device)
            latents = mol_batch.latent_params(gauge_fix_free_axes=True).to(run.device)
            mol_batch, _, condition, _ = ef.condition_samples(mol_batch)
            _, log_pfs, log_pbs, _ = gfn.get_traj_bwd(latents, lambda b: uniform_discretizer(b, run.eval_T),
                                                      condition.to(gfn.device), mol_batch)
            acc += -(log_pfs.sum(-1) - log_pbs.sum(-1)).float().cpu()
        out.append(acc / paths)
    return torch.cat(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoints', nargs='+', help='archives of one run')
    ap.add_argument('--config', required=True, help="the run's yaml")
    ap.add_argument('--unseen', required=True, help='a prior file holding crystals of molecules the run never saw')
    ap.add_argument('--out', required=True, help='directory for the cached row sets and the result')
    ap.add_argument('--prior', default=None, help="the run's prior file, where it is not at the config's prior_path")
    ap.add_argument('--trunk', default=None, help="the run's trunk, where it is not at the config's path")
    ap.add_argument('--rows', type=int, default=2000)
    ap.add_argument('--paths', type=int, default=4)
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    import yaml
    with open(a.config, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    prior = a.prior or cfg['prior_path']
    stem = lambda path: os.path.splitext(os.path.basename(path))[0]
    seen = set(torch.load(prior, map_location='cpu', weights_only=False)['equalized_prior'].identifier)
    train_rows = cached_rows(a.out, f'train_{stem(prior)}', prior, a.rows)
    unseen_rows = cached_rows(a.out, f'unseen_{stem(a.unseen)}_not_{stem(prior)}', a.unseen, a.rows, exclude=seen)
    if set(unseen_rows.identifier) & seen:
        raise ValueError('an unseen row is of a training molecule')
    widths = [r.embedding.reshape(a.rows, -1).shape[1] for r in (train_rows, unseen_rows)]
    if widths[0] != widths[1]:
        raise ValueError(f'the two files carry conditions of different widths {widths}: the unseen file must be '
                         f"embedded as the run's prior is")

    results = []
    for path in a.checkpoints:
        run = load_run(path, a.config, device=a.device, prior_path=a.prior, trunk_path=a.trunk)
        run.gfn.eval()
        torch.manual_seed(0)
        tr = nll(run, train_rows, a.paths, a.batch)
        un = nll(run, unseen_rows, a.paths, a.batch)
        ok_tr, ok_un = torch.isfinite(tr), torch.isfinite(un)
        row = dict(checkpoint=os.path.basename(path), run=cfg.get('run_name'), step=int(run.step),
                   train=float(tr[ok_tr].mean()), train_se=float(tr[ok_tr].std() / math.sqrt(int(ok_tr.sum()))),
                   unseen=float(un[ok_un].mean()), unseen_se=float(un[ok_un].std() / math.sqrt(int(ok_un.sum()))),
                   nonfinite_train=int((~ok_tr).sum()), nonfinite_unseen=int((~ok_un).sum()),
                   rows=a.rows, paths=a.paths)
        results.append(row)
        print('RESULT ' + json.dumps(row), flush=True)
        del run
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()

    with open(os.path.join(a.out, f"heldout_nll_{cfg.get('run_name')}.json"), 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1)
    print()
    print(f"Bound on the negative log likelihood of stored crystals (nats per crystal, lower is better), run "
          f"{cfg.get('run_name')}: {a.rows} crystals per set, the mean of {a.paths} backward trajectories each; "
          f"+- is the standard error over crystals. Unseen crystals are of molecules in no row of the run's prior. "
          f"The gap, unseen minus training, is the overfit measure; compare runs on the unseen column.")
    print('training step | loss, training molecules (nats) | loss, unseen molecules (nats) | gap (nats) | '
          'non-finite rows (training, unseen)')
    for r in results:
        print(f"{r['step']} | {r['train']:.2f} +- {r['train_se']:.2f} | {r['unseen']:.2f} +- {r['unseen_se']:.2f} | "
              f"{r['unseen'] - r['train']:.2f} +- {math.hypot(r['train_se'], r['unseen_se']):.2f} | "
              f"{r['nonfinite_train']}, {r['nonfinite_unseen']}")


if __name__ == '__main__':
    main()
