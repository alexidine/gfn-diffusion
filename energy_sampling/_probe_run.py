"""Run the smoke config with the diagnostic's gate instrumented, to find out
whether it never fires or fires and never reaches wandb."""
import sys
sys.argv = ['train.py', '--config', 'configs/gradgeom_smoke.yaml']

import train

_armed = train.Modeller._fused_grad_diag_armed
_log = train.Modeller._log_fused_gradient_geometry
_ten = train.Modeller.ten_step_reporting
_fused = train.Modeller.fused_train_step


def armed(self):
    r = _armed(self)
    print(f"[probe] armed? {r}  fused_step_count={getattr(self, 'fused_step_count', None)} "
          f"step_ind={self.step_ind}", flush=True)
    return r


def fused(self, *a, **k):
    loss, sub = _fused(self, *a, **k)
    print(f"[probe] fused_train_step branches={list(sub)} "
          f"fracs fwd={self.fwd_frac:.3f} bwd={self.bwd_frac:.3f} replay={self.replay_frac:.3f}",
          flush=True)
    return loss, sub


def log(self, sub_losses, weights, total_weight):
    print(f"[probe] _log called: weights={weights} total={total_weight}", flush=True)
    _log(self, sub_losses, weights, total_weight)
    rep = getattr(self, '_fused_grad_geom_report', None)
    print(f"[probe] -> report {'SET (' + str(len(rep)) + ' keys)' if rep else 'NOT SET'}", flush=True)


def ten(self):
    m = _ten(self)
    n = len([k for k in m if k.startswith('fused_grad/')])
    print(f"[probe] ten_step_reporting at step {self.step_ind}: {n} fused_grad keys", flush=True)
    return m


train.Modeller._fused_grad_diag_armed = armed
train.Modeller._log_fused_gradient_geometry = log
train.Modeller.ten_step_reporting = ten
train.Modeller.fused_train_step = fused

train.Modeller().train()
