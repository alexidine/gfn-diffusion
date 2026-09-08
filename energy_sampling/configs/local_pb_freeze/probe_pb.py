"""How far does the learned P_B depart from the reference bridge, and how much
does phase 2 move it? Scored on STORED replay trajectories (a buffer sidecar's
replay_buffer.traj), so every checkpoint is scored on the same fixed paths.

    python configs/local_pb_freeze/probe_pb.py <sidecar_buffers.pt> name=ckpt.pt [name=ckpt.pt ...]

For each checkpoint: per-step median |kappa - 1| (multiplicative drift
correction, bounded by pb_drift_range 0.4) and median |dlogvar| (additive
log-variance correction, bounded by pb_var_range 6), the trajectory log P_B
under the learned correction and under the pure bridge (learn_pb switched off
on the same weights), and the log P_B gap between checkpoints -- the number of
nats phase 2 moved P_B by on these paths.
"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.gfn import GFN
from utils import uniform_discretizer

sidecar = sys.argv[1]
ckpts = dict(a.split('=', 1) for a in sys.argv[2:])
dev = 'cuda'
sd = torch.load(sidecar, map_location='cpu', weights_only=False)
traj = sd['replay_buffer']['traj'].to(dev)
n, T1, dim = traj.shape
print(f'{n} stored replay paths, T={T1 - 1}, dim={dim}')
disc = lambda b: uniform_discretizer(b, T1 - 1)
B = 1000
traj = traj[:B]

def load(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    g = GFN(**ck['gfn_config']).to(dev)
    g.load_state_dict(ck['model_train']); g.eval()
    return g, ck

@torch.no_grad()
def score(g):
    ts = disc(B).to(dev)
    kap, dlv = [], []
    for i in range(T1 - 1):
        nxt = g._wrap_ang(traj[:, i + 1])
        k, v = g.fwd_get_back_correction(None, i, g.expand_state_for_policy(nxt), ts)
        k, v = g._live_only(k, v)
        kap.append((k - 1).abs().median().item()); dlv.append(v.abs().median().item())
    _, lpf, lpb, _ = g.get_traj_replay(traj, disc, condition=None, mol_batch=None)
    g.learn_pb = False
    _, _, lpb0, _ = g.get_traj_replay(traj, disc, condition=None, mol_batch=None)
    g.learn_pb = True
    return np.array(kap), np.array(dlv), lpf.sum(-1), lpb.sum(-1), lpb0.sum(-1)

res = {}
for name, path in ckpts.items():
    g, ck = load(path)
    kap, dlv, lpf, lpb, lpb0 = score(g)
    res[name] = dict(lpf=lpf, lpb=lpb, lpb0=lpb0)
    print(f'\n== {name}  (step {ck.get("modeller_state", {}).get("step_ind", "?")})')
    print('  median |kappa-1| per step:', np.round(kap, 3))
    print('  median |dlogvar| per step:', np.round(dlv, 3))
    print(f'  log P_B learned  mean {lpb.mean():9.2f}   pure bridge {lpb0.mean():9.2f}   learned-bridge {(lpb - lpb0).mean():8.2f} (sd {(lpb - lpb0).std():.2f})')
    print(f'  log P_F          mean {lpf.mean():9.2f}   log(pf/pb) sd {(lpf - lpb).std():.2f}   log(pf/bridge) sd {(lpf - lpb0).std():.2f}')
names = list(res)
for i in range(1, len(names)):
    a, b = res[names[0]], res[names[i]]
    d = b['lpb'] - a['lpb']
    print(f'\nlog P_B[{names[i]}] - log P_B[{names[0]}] on the same paths: mean {d.mean():.2f}  sd {d.std():.2f}  |.| median {d.abs().median():.2f}')
    d0 = b['lpb0'] - a['lpb0']
    print(f'  (pure-bridge difference, should be 0: {d0.abs().max():.2e})')
