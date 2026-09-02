"""
Acceptance gate for any density model proposed as a null-test target.

THE QUESTION THIS ANSWERS. A lambda=0 null test runs VarGrad against the prior's
own implied energy and asks whether the policy stays put. That is only a valid
test if the energy really is -log p_prior UP TO AN ADDITIVE CONSTANT. VarGrad
reads the spread of log-weights within a condition group, so a constant offset
cancels -- but ANY OTHER systematic error does not, and a multiplicative error
on log p is indistinguishable, at a glance, from the sampler misbehaving.

So the criterion is not "does the model fit well" in any general sense. It is
specifically:

    regressing  E_model(x)  on  -log p_true(x)  must give SLOPE ~ 1

A slope of s means exp(-E) is proportional to p^s. The policy then sees a real
gradient at the prior, every logged number stays finite, and the drift is
attributed to the sampler -- which is the one conclusion the null test exists to
rule out. Correlation is NOT sufficient and is the trap here: a slope-0.72 model
still correlates 0.99 with the truth.

This gate was written after `latent_knn` passed every structural test in the
suite -- geometry, gauge, chunking, digests, loud failure on mismatch -- and
still failed this one at d=12. Structural correctness does not imply the energy
means what it is supposed to mean. Run any candidate (kNN, GMM, MLP, flow)
through here before trusting a null result from it.

    python -m energies.density_calibration
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch

#: Slope band within which a model is usable as a null-test target. Narrow on
#: purpose: at the shipped log-density spread (sd ~0.7-2.4 nats) a slope error of
#: 0.05 is already a systematic force of order the signal the test looks for.
DEFAULT_SLOPE_TOLERANCE = 0.05


class Calibration:
    __slots__ = ('slope', 'intercept', 'corr', 'residual_sd', 'signal_sd', 'n', 'tolerance')

    def __init__(self, slope, intercept, corr, residual_sd, signal_sd, n, tolerance):
        self.slope = slope
        self.intercept = intercept
        self.corr = corr
        self.residual_sd = residual_sd
        self.signal_sd = signal_sd
        self.n = n
        self.tolerance = tolerance

    @property
    def passes(self) -> bool:
        return abs(self.slope - 1.0) <= self.tolerance

    def __str__(self) -> str:
        verdict = 'PASS' if self.passes else 'FAIL'
        return (f'{verdict}  slope={self.slope:.3f} (tol +-{self.tolerance:.2f})  '
                f'corr={self.corr:.4f}  resid_sd={self.residual_sd:.3f} nats  '
                f'signal_sd={self.signal_sd:.3f} nats  n={self.n}')

    def explain(self) -> str:
        if self.passes:
            return 'usable as a null-test target at this geometry'
        direction = 'inward (over-concentrating)' if self.slope > 1 else 'outward (over-spreading)'
        return (f'exp(-E) is approximately p^{self.slope:.2f}, not p. A policy sitting exactly '
                f'at this prior would drift {direction}. Correlation of {self.corr:.4f} does not '
                f'rescue this -- it is a slope error, not a fit error.')


def calibrate(energy_fn: Callable[[torch.Tensor], torch.Tensor],
              samples: torch.Tensor,
              true_log_p: torch.Tensor,
              tolerance: float = DEFAULT_SLOPE_TOLERANCE) -> Calibration:
    """Regress E_model on -log p_true and report the slope.

    `energy_fn` is anything mapping [B, D] -> [B]; `true_log_p` is the exact
    log-density of `samples` under the distribution the reference was drawn from.
    """
    with torch.no_grad():
        # the model may live on the accelerator while the analytic log p is built
        # on the host; the regression is tiny, so settle both on cpu rather than
        # making every caller match devices by hand
        e = energy_fn(samples).detach().to('cpu', torch.float64).flatten()
    target = (-true_log_p).detach().to('cpu', torch.float64).flatten()
    if e.numel() != target.numel():
        raise ValueError(f'energy returned {e.numel()} values for {target.numel()} samples')

    tc = target - target.mean()
    ec = e - e.mean()
    denom = float((tc * tc).sum())
    if denom <= 0:
        raise ValueError('true_log_p is constant across the sample; nothing to regress against')

    slope = float((tc * ec).sum() / denom)
    intercept = float(e.mean() - slope * target.mean())
    resid = ec - slope * tc
    corr = float((tc * ec).sum() / (tc.norm() * ec.norm()).clamp_min(1e-30))
    return Calibration(slope=slope, intercept=intercept, corr=corr,
                       residual_sd=float(resid.std()), signal_sd=float(target.std()),
                       n=int(target.numel()), tolerance=float(tolerance))


def wrapped_gaussian_draw(n: int, dim: int, wrap_mask: Sequence[bool], width: float,
                          period: float = 2.0, images: int = 3,
                          generator: Optional[torch.Generator] = None):
    """Samples and their EXACT log-density, in the metric PriorKNN uses.

    Linear dims are plain normals; wrapped dims are wrapped normals of the same
    period as the minimum-image fold, summed over `images` periodic copies either
    side. Matching the metric matters -- calibrating against an unwrapped gaussian
    would charge the estimator for a mismatch that is the test's own fault.
    """
    wrap = torch.as_tensor(list(wrap_mask), dtype=torch.bool)
    if wrap.numel() != dim:
        raise ValueError(f'wrap_mask has {wrap.numel()} entries for dim={dim}')

    x = width * torch.randn(n, dim, generator=generator)
    x[:, wrap] = x[:, wrap] - period * torch.round(x[:, wrap] / period)

    # log N(x; 0, width^2) per dim, wrapped dims summed over periodic images
    log_p = torch.zeros(n, dtype=torch.float64)
    norm = -0.5 * math.log(2 * math.pi * width * width)
    for d in range(dim):
        col = x[:, d].to(torch.float64)
        if bool(wrap[d]):
            offs = torch.arange(-images, images + 1, dtype=torch.float64) * period
            terms = norm - 0.5 * ((col.unsqueeze(1) + offs.unsqueeze(0)) / width) ** 2
            log_p += torch.logsumexp(terms, dim=1)
        else:
            log_p += norm - 0.5 * (col / width) ** 2
    return x, log_p


def wrapped_mixture_draw(n: int, dim: int, wrap_mask: Sequence[bool], n_modes: int = 8,
                         width: float = 0.18, spread: float = 0.55, period: float = 2.0,
                         images: int = 3, generator: Optional[torch.Generator] = None,
                         mode_seed: int = 1234):
    """A MULTIMODAL target with exact log p, in PriorKNN's metric.

    A unimodal gaussian flatters every candidate -- a single full-covariance
    gaussian fits it perfectly and a flow has nothing to learn. A real MLE prior
    over crystal latents is not unimodal, so a density model that only passes on
    the easy target has not been tested on the thing it will be asked to do.
    """
    wrap = torch.as_tensor(list(wrap_mask), dtype=torch.bool)
    if wrap.numel() != dim:
        raise ValueError(f'wrap_mask has {wrap.numel()} entries for dim={dim}')

    # modes are a property of the TARGET, so they must not move with the draw seed
    mg = torch.Generator().manual_seed(mode_seed)
    means = spread * torch.randn(n_modes, dim, generator=mg)
    means[:, wrap] = means[:, wrap] - period * torch.round(means[:, wrap] / period)
    logits = 0.5 * torch.randn(n_modes, generator=mg)
    log_w = torch.log_softmax(logits.to(torch.float64), dim=0)

    which = torch.multinomial(log_w.exp().float(), n, replacement=True, generator=generator)
    x = means[which] + width * torch.randn(n, dim, generator=generator)
    x[:, wrap] = x[:, wrap] - period * torch.round(x[:, wrap] / period)

    log_p = _wrapped_mixture_log_p(x, means, log_w, wrap, width, period, images)
    return x, log_p


def warped_mixture_draw(n: int, dim: int, wrap_mask: Sequence[bool], amp: float = 0.10,
                        n_modes: int = 8, width: float = 0.18, spread: float = 0.55,
                        period: float = 2.0, images: int = 3,
                        generator: Optional[torch.Generator] = None, mode_seed: int = 1234):
    """A NON-GAUSSIAN target with exact log p.

    The other two targets are wrapped gaussians, which places the truth INSIDE a
    Gaussian mixture's own family -- so a GMM scores well there for a reason that
    will not transfer to a real MLE prior. This one pushes a wrapped mixture
    through the componentwise bijection

        y = z + amp * sin(pi * z)

    which is a genuine bijection of the circle for amp*pi < 1 and fixes z = +-1,
    so the period survives. Each mode comes out skewed and non-elliptical, and the
    exact density follows by change of variables:

        log p_y(y) = log p_z(z) - sum_i log(1 + amp*pi*cos(pi*z_i))

    No inverse is needed because the sample is generated in z and pushed forward,
    so every returned point arrives with its own pre-image.

    `amp` sets the difficulty and needs calibrating, not maximising. The
    log-jacobian is a high-frequency function of z, so a large amp makes it
    DOMINATE: measured at d=12, amp 0.25 puts 74% of the log-density variance in
    the jacobian term, which stops being "a non-gaussian prior" and becomes "can
    you regress a fast oscillation". The default 0.10 leaves that share at ~36%,
    so the mixture structure still leads and the departure from the gaussian
    family is real rather than overwhelming.
    """
    if amp * math.pi >= 1.0:
        raise ValueError(f'amp={amp} is not invertible; need amp < 1/pi = {1/math.pi:.4f}')

    z, log_p_z = wrapped_mixture_draw(n, dim, wrap_mask, n_modes=n_modes, width=width,
                                      spread=spread, period=period, images=images,
                                      generator=generator, mode_seed=mode_seed)
    scale = math.pi / (period / 2.0)        # sin period matches the wrap period
    y = z + amp * torch.sin(scale * z)
    log_det = torch.log1p(amp * scale * torch.cos(scale * z).to(torch.float64)).sum(dim=1)
    return y, log_p_z - log_det


def _wrapped_mixture_log_p(x, means, log_w, wrap, width, period, images):
    x64 = x.to(torch.float64)
    mu = means.to(torch.float64)
    norm = -0.5 * math.log(2 * math.pi * width * width)
    offs = torch.arange(-images, images + 1, dtype=torch.float64) * period

    per_mode = torch.zeros(x64.shape[0], mu.shape[0], dtype=torch.float64)
    for d in range(x64.shape[1]):
        delta = x64[:, d].unsqueeze(1) - mu[:, d].unsqueeze(0)          # [B, M]
        if bool(wrap[d]):
            terms = norm - 0.5 * ((delta.unsqueeze(2) + offs) / width) ** 2
            per_mode += torch.logsumexp(terms, dim=2)
        else:
            per_mode += norm - 0.5 * (delta / width) ** 2
    return torch.logsumexp(per_mode + log_w.unsqueeze(0), dim=1)


def _report(dim, wrap_mask, width, n_ref, k, seed=0):
    from energies.prior_knn import PriorKNN

    g = torch.Generator().manual_seed(seed)
    ref, _ = wrapped_gaussian_draw(n_ref, dim, wrap_mask, width, generator=g)
    query, log_p = wrapped_gaussian_draw(3000, dim, wrap_mask, width, generator=g)
    knn = PriorKNN(ref, wrap_mask=wrap_mask, k=k)
    cal = calibrate(knn.energy, query, log_p)
    print(f'  d={dim:<3} width={width:<5} N={n_ref:<7} k={k:<4} {cal}')
    return cal


if __name__ == '__main__':
    print('kNN calibration against a closed-form wrapped gaussian\n')
    print('low dimension -- the estimator is local, slope ~ 1:')
    for d in (2, 3):
        _report(d, [True] * d, 0.45, 20000, 32)

    print('\nshipped geometry (SG2/Z\'=1, d=12, wrapped {7,8,10,11}):')
    mask12 = [i in (7, 8, 10, 11) for i in range(12)]
    for width in (0.30, 0.45, 0.80):
        _report(12, mask12, width, 20000, 32)
    print('\nmore data does not fix a slope error (N^(-1/6)):')
    for n in (20000, 100000, 400000):
        _report(12, mask12, 0.45, n, 32)
