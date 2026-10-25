"""Draws from an inn_modeller (exact-density autoregressive flow) checkpoint of an unconditional sg 14 run, for
new_analysis.py's `stored_draws`. The flow is rebuilt from the flags the checkpoint stores and its weights loaded
strictly; the draws are in the trainer's latent chart, with the flow's log-density of each.
usage: python inn_draws.py <checkpoint> <out.pt> <n>, with the run's own commit of energy_sampling first on PYTHONPATH
(inn_modeller is imported from there)."""
import os, sys
# run as a script, this folder leads sys.path and its utils.py would shadow the trainer's (inn_modeller -> buffer)
sys.path = [q for q in sys.path if os.path.abspath(q or '.') != os.path.dirname(os.path.abspath(__file__))]
import torch
from inn_modeller import LatentFlow
ck_path, out_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ck = torch.load(ck_path, map_location='cpu', weights_only=False)
inn = ck['inn']
if 'space_groups=[14]' not in (inn.get('overrides') or []):
    raise ValueError(f"{ck_path}: the sg 14 geometry below is hard-coded; this run's overrides are {inn.get('overrides')}")
# sg 14 geometry as the job printed it: dim 12, wrapped rows [6, 8, 10, 11], dead rows [3, 5]
dim, wrap, dead = 12, [6, 8, 10, 11], (3, 5)
wrap_mask = [i in wrap for i in range(dim)]
assert inn['bounded_rows'] == 'linear' and inn['n_blocks'] == 1 and not inn['mix_k'] and not inn['fold_half_rows']
bounded = [i for i in range(dim) if i not in dead and not wrap_mask[i]]
shape = (inn['n_blocks'], inn['n_bins'], (inn['hidden'],) * inn['hidden_layers'], inn['bound'])
flow = LatentFlow(dim, wrap_mask, dead, *shape, seed=0, transformer=(inn['tf_width'], inn['tf_layers'], inn['tf_heads']),
                  block_orders=inn['block_orders'].split(','), half_rows=[], bounded_rows=bounded, mix_k=0)
flow.load_state_dict(ck['state_dict'], strict=True)
flow = flow.cuda()  # as the trainer: moved before the floor is set from blk.bound
h0 = float(inn['min_bin_height'])
assert inn['min_bin_height_rows'] == 'all'
for blk in flow.flow.blocks:
    blk.min_height = torch.full_like(blk.bound, h0)
    if inn['tied_derivatives']:
        blk.tied = float(inn['tied_factor'])
if inn['tf_fp32']:
    flow.set_bf16(False)
flow = flow.eval()
xs, lq = [], []
torch.manual_seed(1)
with torch.no_grad():
    for i in range(0, n, 2000):
        x = flow.sample(min(2000, n - i))
        xs.append(x.cpu()); lq.append(flow.log_prob(x).cpu())
x, lq = torch.cat(xs), torch.cat(lq)
print(f"{len(x)} draws from {ck_path} (update {ck['step']}); log q median {lq.median():.2f}; "
      f"dead rows max |x| {x[:, list(dead)].abs().max():.2e}; rows outside [-1, 1]: {100 * float((x.abs() > 1).any(1).float().mean()):.2f}%")
torch.save({'latents': x, 'log_q': lq, 'step': int(ck['step']), 'log_z': float(ck['log_z_level']),
            'checkpoint_file': ck_path, 'space_group': 14, 'z_prime': 1, 'max_z_prime': 1, 'energy_function': 'mace',
            'model': 'inn_modeller AR flow (f8b7c177)'}, out_path)
print('saved', out_path)
