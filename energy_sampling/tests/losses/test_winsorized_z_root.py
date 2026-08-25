"""`winsorized_z_root` must solve for the SHIPPING TB loss's own Z fixed point.

The point of the function is to replace a rollout-driven servo with a closed-form
answer when log_Z is far from its fixed point: the Z-only sidecar freezes the
policy, so the target does not move across a tick and fresh energy calls buy
nothing but a fresh draw of the sampling error. Two claims carry that, and both
are pinned here against the real loss rather than against a restatement of it:

  * the root really is where `get_tb_loss` has zero gradient in log_Z, and
  * the reported `se` is the scale that decides whether a gap is signal.

The root is a WINSORIZED location estimator, so it moves with beta. That is the
claim most likely to be got backwards, and `test_raising_beta_pulls_the_root_down`
is the one that would catch it.
"""
import pytest
import torch

from gflownet_losses import get_tb_loss, winsorized_z_root


def _tb_grad_at(z_value, logw, beta):
    """dL/d log_Z of the REAL get_tb_loss, by autograd, at a given log_Z.

    Rebuilds the loss's own inputs so nothing here re-implements the residual:
    with log_pf = log_pb = 0 and log_r = logw, `log_pf + log_Z - log_pb - log_r`
    is exactly `z - logw`.
    """
    z = torch.tensor(z_value, dtype=torch.float64, requires_grad=True)
    zeros = torch.zeros_like(logw)
    loss = get_tb_loss(z, zeros, zeros, logw, beta=beta).mean()
    loss.backward()
    return float(z.grad)


def _left_skewed(n=2048, n_clash=40, seed=0):
    """Bulk of good rows plus a few catastrophic ones -- the shape a
    mode-covering forward objective actually produces, where a clash gives a
    catastrophic log_r and therefore a large POSITIVE residual."""
    g = torch.Generator().manual_seed(seed)
    bulk = torch.randn(n - n_clash, generator=g, dtype=torch.float64) * 3.0 - 40.0
    clash = torch.full((n_clash,), -800.0, dtype=torch.float64)
    return torch.cat([bulk, clash])


# --------------------------------------------------------------- the root itself

def test_root_zeroes_the_real_tb_loss_gradient():
    """THE load-bearing test: the analytic root must kill the shipped loss's
    autograd gradient, not merely satisfy our own restatement of it."""
    logw = _left_skewed()
    root, _, _ = winsorized_z_root(logw, beta=10.0)
    assert abs(_tb_grad_at(root, logw, 10.0)) < 1e-8


def test_the_gradient_test_can_actually_fail():
    """Re-introduce the error the test above is meant to catch. A root one nat
    off must leave a gradient the assertion above would reject -- otherwise that
    assertion passes on any number and pins nothing."""
    logw = _left_skewed()
    root, _, _ = winsorized_z_root(logw, beta=10.0)
    assert abs(_tb_grad_at(root + 1.0, logw, 10.0)) > 1e-3
    assert abs(_tb_grad_at(root - 1.0, logw, 10.0)) > 1e-3


def test_unclipped_root_is_the_plain_mean():
    """With beta above every residual nothing winsorizes, so the Huber loss is
    a plain quadratic and its root is the arithmetic mean -- a closed form the
    bisection can be checked against."""
    logw = _left_skewed()
    root, _, frac = winsorized_z_root(logw, beta=1.0e6)
    assert root == pytest.approx(float(logw.mean()), abs=1e-6)
    assert frac == pytest.approx(1.0)


def test_raising_beta_pulls_the_root_down():
    """Winsorization is what keeps the clash tail from setting the level. Rows
    far below the bulk produce large POSITIVE residuals, so they push the root
    DOWN, and a larger beta lets each of them push harder. Getting this
    backwards is what would make 'just raise beta for Z training' look free."""
    logw = _left_skewed()
    tight, _, _ = winsorized_z_root(logw, beta=10.0)
    loose, _, _ = winsorized_z_root(logw, beta=200.0)
    plain, _, _ = winsorized_z_root(logw, beta=1.0e6)
    assert tight > loose > plain
    # and the effect is large enough to matter, not a rounding artefact
    assert tight - plain > 1.0


def test_root_is_bracketed_by_the_data():
    """The bracket argument the solver relies on: no root can sit outside
    [min(logw) - beta, max(logw) + beta]."""
    for seed in range(5):
        logw = _left_skewed(n=512, seed=seed)
        root, _, _ = winsorized_z_root(logw, beta=10.0)
        assert float(logw.min()) - 10.0 <= root <= float(logw.max()) + 10.0


def test_is_reproducible():
    """Fixed iteration count, no data-dependent stopping: the same batch must
    give the same bits, or a fill fired from it is not auditable."""
    logw = _left_skewed()
    assert winsorized_z_root(logw, beta=10.0) == winsorized_z_root(logw, beta=10.0)


# ------------------------------------------------------------------ the SE gate

def test_se_shrinks_as_one_over_sqrt_batch():
    """The gate's whole meaning: se is the batch's sampling error on the root,
    so 16x the rows must roughly halve it twice."""
    small, _ = winsorized_z_root(_left_skewed(n=512, n_clash=10, seed=1), beta=10.0)[1:]
    large, _ = winsorized_z_root(_left_skewed(n=8192, n_clash=160, seed=1), beta=10.0)[1:]
    assert large < small
    assert small / large == pytest.approx(4.0, rel=0.35)


def test_se_matches_the_bootstrap_spread_of_the_root():
    """se must MEAN something: the batch-to-batch standard deviation of the root.
    Resampling these rows with replacement and re-solving measures that spread
    directly, and the analytic sandwich has to reproduce it.

    This is the only assertion that pins BOTH halves of the formula at once. An
    se that is merely PROPORTIONAL to the right answer -- which is what dropping
    the curvature divisor produces -- satisfies the 1/sqrt(B) scaling and the
    tail-insensitivity check and fails only here. The setup deliberately spreads
    the bulk wide enough to saturate a large fraction of rows, because at
    frac ~ 1 the divisor is invisible by construction."""
    g = torch.Generator().manual_seed(5)
    bulk = torch.randn(760, generator=g, dtype=torch.float64) * 12.0 - 40.0
    logw = torch.cat([bulk, torch.full((40,), -800.0, dtype=torch.float64)])
    _, se, frac = winsorized_z_root(logw, beta=10.0)
    assert frac < 0.8, f'setup failed to saturate enough rows (frac={frac})'

    n = logw.numel()
    roots = []
    for i in range(300):
        idx = torch.randint(n, (n,), generator=torch.Generator().manual_seed(1000 + i))
        roots.append(winsorized_z_root(logw[idx], beta=10.0)[0])
    empirical = float(torch.tensor(roots, dtype=torch.float64).std())
    assert se == pytest.approx(empirical, rel=0.15), f'analytic {se}, bootstrap {empirical}'


def test_se_is_infinite_when_every_row_is_saturated():
    """A batch split far either side of the knee balances at +-beta everywhere,
    so the loss is FLAT in z over a wide interval and the root is not
    identified. se must say so rather than return a number -- a caller gating
    on it has to decline, and a finite se here would license a fill onto an
    arbitrary point of that flat interval."""
    logw = torch.tensor([-1000.0] * 64 + [1000.0] * 64, dtype=torch.float64)
    _, se, frac = winsorized_z_root(logw, beta=10.0)
    assert frac == 0.0
    assert se == float('inf')


def test_se_uses_the_clipped_residual_not_the_raw_spread():
    """se must not inherit the unbounded tail. Moving the clash rows further out
    changes the raw spread enormously and the winsorized one not at all, so the
    reported se must be essentially unchanged."""
    g = torch.Generator().manual_seed(3)
    bulk = torch.randn(2000, generator=g, dtype=torch.float64) * 3.0 - 40.0
    near = torch.cat([bulk, torch.full((40,), -300.0, dtype=torch.float64)])
    far = torch.cat([bulk, torch.full((40,), -30000.0, dtype=torch.float64)])
    _, se_near, _ = winsorized_z_root(near, beta=10.0)
    _, se_far, _ = winsorized_z_root(far, beta=10.0)
    assert se_far == pytest.approx(se_near, rel=1e-6)


# --------------------------------------------------------------------- the guards

def test_non_finite_input_raises_rather_than_returning_nan():
    """A nan root filled into log_Z is unrecoverable, and bisection produces one
    silently from a single inf."""
    logw = _left_skewed(n=128)
    logw[7] = float('inf')
    with pytest.raises(ValueError):
        winsorized_z_root(logw, beta=10.0)
    logw[7] = float('nan')
    with pytest.raises(ValueError):
        winsorized_z_root(logw, beta=10.0)


def test_empty_and_non_positive_beta_raise():
    with pytest.raises(ValueError):
        winsorized_z_root(torch.empty(0, dtype=torch.float64), beta=10.0)
    with pytest.raises(ValueError):
        winsorized_z_root(_left_skewed(n=64), beta=0.0)


def test_curvature_never_exceeds_one():
    """frac_unclipped IS d2L/dz2. It bounding at 1 is what makes plain gradient
    descent on this scalar stable at any step below 2 -- the reason the fill
    path needs no optimizer and no learning rate."""
    for beta in (1.0, 10.0, 200.0, 1.0e6):
        _, _, frac = winsorized_z_root(_left_skewed(n=512), beta=beta)
        assert 0.0 <= frac <= 1.0
