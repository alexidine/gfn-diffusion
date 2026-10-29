"""
The LR controller's hard-failure bars: per-branch loss and gradient bars fitted
from a window of the run's own observations, plus absolute backstops.

    loss bar  = hi + loss_excursion_k x (hi - lo)     over the window's tail
    grad bar  = grad_excursion_x x max(pre-clip grad norm)

A SPAN AND NOT A RATIO: `hi + k x (hi - lo)` is defined whatever the sign of the
channel, and the MLE channel passes through zero and goes negative.

`LRController` owns the windows and decides when to fit (end of burn-in, a
resume into cruise, the refit after the promotion ramp); this class only fits a
bar from a window it is handed and judges a reading against it.
"""

from __future__ import annotations

import math


class HardFailureBars:
    """Direct, hard safety conditions fitted from an observation window."""

    #: Channels that reach `judge`. Mirrors train_step's `step_type`.
    CHANNELS = ('fwd', 'bwd', 'replay', 'fused')

    def __init__(self, loss_excursion_k=10.0, grad_excursion_x=100.0,
                 loss_abs=1.0e6, grad_abs=1.0e6, root_window=200,
                 min_observations=20):
        self.loss_excursion_k = float(loss_excursion_k)
        self.grad_excursion_x = float(grad_excursion_x)
        self.loss_abs = None if loss_abs is None else float(loss_abs)
        self.grad_abs = None if grad_abs is None else float(grad_abs)
        self.root_window = int(root_window)
        self.min_observations = int(min_observations)
        self.loss_bar = {}          # channel -> absolute value that counts as failure
        self.grad_bar = None
        self.scale_note = ''

    def derive(self, loss_history, grad_history):
        """Fit the bars. `loss_history` is {channel: [values]} and `grad_history`
        a list of pre-clip gradient norms.

        Uses the last `root_window` finite values per channel; a channel with
        fewer than `min_observations` gets no loss bar. Returns None when at
        least one loss bar was fitted, else a string saying why none was.
        """
        self.loss_bar, notes = {}, []
        for channel, values in (loss_history or {}).items():
            tail = [v for v in list(values)[-self.root_window:] if math.isfinite(v)]
            if len(tail) < self.min_observations:
                continue
            hi, lo = max(tail), min(tail)
            span = hi - lo
            if not (span > 0):
                # A flat channel has no span; the magnitude is the scale used.
                span = abs(hi) if abs(hi) > 0 else 1.0
            self.loss_bar[channel] = hi + self.loss_excursion_k * span
            notes.append(f'{channel}: [{lo:.4g}, {hi:.4g}] -> bar '
                         f'{self.loss_bar[channel]:.4g}')
        gtail = [g for g in list(grad_history or ())[-self.root_window:]
                 if math.isfinite(g) and g > 0]
        if len(gtail) >= self.min_observations:
            self.grad_bar = max(gtail) * self.grad_excursion_x
            notes.append(f'grad: max {max(gtail):.4g} -> bar {self.grad_bar:.4g}')
        self.scale_note = '; '.join(notes)
        if not self.loss_bar:
            return (f'no channel accumulated {self.min_observations} finite loss '
                    f'observations, so no loss bar was fitted; only the non-finite '
                    f'and absolute checks can fire.')
        return None

    def judge(self, channel, loss, grad_norm):
        """The reason this reading fails, or None. Non-finite values, the
        absolute backstops, and the fitted bars, in that order."""
        if loss is not None:
            if not math.isfinite(loss):
                return f'nonfinite_loss_{channel}'
            if self.loss_abs is not None and abs(loss) >= self.loss_abs:
                return f'loss_abs_{channel}_{loss:.4g}'
            bar = self.loss_bar.get(channel)
            if bar is not None and loss >= bar:
                return f'loss_excursion_{channel}_{loss:.4g}_over_{bar:.4g}'
        if grad_norm is not None:
            if not math.isfinite(grad_norm):
                return 'nonfinite_grad'
            if self.grad_abs is not None and grad_norm >= self.grad_abs:
                return f'grad_abs_{grad_norm:.4g}'
            if self.grad_bar is not None and grad_norm >= self.grad_bar:
                return f'grad_excursion_{grad_norm:.4g}_over_{self.grad_bar:.4g}'
        return None

    def report(self) -> dict:
        out = {}
        for c, bar in self.loss_bar.items():
            out[f'lr_bracket/bar_loss_{c}'] = float(bar)
        if self.grad_bar is not None:
            out['lr_bracket/bar_grad'] = float(self.grad_bar)
        return out
