"""fwd_loss_coeffs.pooled_center: what pooled_condition_vargrad scores each group's rows against.

'mean' is the shipped VarGrad term: the centre is the group's own row mean, kept in the
graph, so the per-row coefficients sum to zero and one bad row spills onto its partners.
'head' and 'head_stored' score the same rows against a DETACHED level, so the term is
per-row TB and a row's coefficient is its own psi(u_j - L). These tests pin the three
levels, the spill that 'mean' has and the detached centres do not, and the readouts.
"""
import pytest
import torch

from energy_sampling.gflownet_losses import pooled_condition_vargrad

BETA = 40.0


def _group(replay_u, stored_u, head, cid=0):
    """One condition's live stashes on the NF: log_pb 0, log_pf the parameter, so u = log_r - log_pf."""
    def side(us):
        n = len(us)
        log_pf = torch.zeros(n, requires_grad=True)
        return {'log_r': torch.tensor(us, dtype=torch.float32), 'log_pb': torch.zeros(n), 'log_pf': log_pf,
                'condition_id': torch.full((n,), cid), 'log_z': torch.full((n,), float(head))}
    return side(replay_u), side(stored_u)


def _coeffs(center, replay_u, stored_u, head=0.0, beta_fwd=None):
    """d loss / d log_pf per row, replay rows first: the sign is the direction descent moves log q there."""
    rep, sto = _group(replay_u, stored_u, head)
    rows, stats = pooled_condition_vargrad(rep, sto, beta=BETA, ratio=0.5, center=center, beta_fwd=beta_fwd)
    rows.sum().backward()
    return torch.cat([rep['log_pf'].grad, sto['log_pf'].grad]), stats


def test_mean_coefficients_sum_to_zero_and_a_bad_row_spills():
    g, _ = _coeffs('mean', [-2.0, -2.0], [2.0, 2.0])
    assert abs(float(g.sum())) < 1e-4
    bad, _ = _coeffs('mean', [-202.0, -2.0], [2.0, 2.0])
    # past the knee the bad row's partners all take +beta/2 more (in -dL/dlogpf terms)
    assert torch.allclose(-bad[1:], torch.full((3,), BETA / 2), atol=1e-3)


@pytest.mark.parametrize('center', ['head', 'head_stored'])   # not 'mean_detached': its level is the group mean
def test_detached_centres_keep_a_bad_row_to_itself(center):
    healthy, _ = _coeffs(center, [-2.0, -2.0], [2.0, 2.0], head=0.0)
    bad, _ = _coeffs(center, [-202.0, -2.0], [2.0, 2.0], head=0.0)
    assert torch.allclose(bad[1:], healthy[1:], atol=1e-5), 'the partners do not see the bad row'
    assert abs(float(bad[0])) == pytest.approx(BETA, abs=1e-4), 'its own influence is capped at beta'


def test_head_is_tb_against_the_head():
    g, stats = _coeffs('head', [-2.0, 1.0], [3.0, 5.0], head=1.0)
    # 40 * smooth_l1(L, u, 40) = (u - L)^2 / 2 inside the knee; u = log_r - log_pf, so d/dlogpf = -(u - L)
    assert torch.allclose(g, -torch.tensor([-3.0, 0.0, 2.0, 4.0]), atol=1e-5)
    assert float(stats['pooled_center_minus_head']) == pytest.approx(0.0)


def test_head_stored_is_half_head_half_stored_mean():
    g, stats = _coeffs('head_stored', [-2.0, 1.0], [3.0, 5.0], head=1.0)
    level = 0.5 * 1.0 + 0.5 * 4.0
    assert torch.allclose(g, -(torch.tensor([-2.0, 1.0, 3.0, 5.0]) - level), atol=1e-5)
    assert float(stats['pooled_center_minus_head']) == pytest.approx(level - 1.0)


def test_readouts():
    _, stats = _coeffs('mean', [-202.0, -2.0], [2.0, 2.0], head=0.0)
    assert float(stats['pooled_gap_signed']) == pytest.approx(2.0 - (-102.0))
    # centre (-202 - 2 + 2 + 2) / 4 = -50 drags every row past the knee, the healthy three included
    assert float(stats['pooled_knee_frac']) == pytest.approx(1.0)
    assert float(stats['pooled_center_minus_head']) == pytest.approx(-50.0)
    _, stats = _coeffs('head', [-202.0, -2.0], [2.0, 2.0], head=0.0)
    assert float(stats['pooled_knee_frac']) == pytest.approx(0.25), 'against the head only the bad row is'


def test_unknown_centre_and_a_stash_without_the_head_raise():
    rep, sto = _group([0.0, 0.0], [0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match='pooled_center|center='):
        pooled_condition_vargrad(rep, sto, center='pooled')
    for d in (rep, sto):
        d.pop('log_z')
    rows, stats = pooled_condition_vargrad(rep, sto, center='mean')
    assert rows is not None and 'pooled_center_minus_head' not in stats
    with pytest.raises(ValueError, match="'log_z'"):
        pooled_condition_vargrad(rep, sto, center='head')


def test_a_first_side_knee_bounds_the_push_on_own_samples_only_with_a_detached_mean():
    """The worked case: 2 replay rows 40 nats below the group mean and 2 stored rows 40 above, replay knee 5, stored
    knee 40. Detached: each replay row's log q is pushed down at 5, each stored row's raised at 40. 'mean' in the
    graph would balance the sides (22.5 each), so it refuses the knee."""
    g, stats = _coeffs('mean_detached', [-40.0, -40.0], [40.0, 40.0], beta_fwd=5.0)
    # u = log_r - log_pf; a replay row 40 below the level: d loss/d log_pf = +knee (descent lowers its log q)
    assert torch.allclose(g, torch.tensor([5.0, 5.0, -40.0, -40.0]), atol=1e-4)
    assert float(stats['pooled_knee_frac']) == pytest.approx(0.5), 'replay rows past their 5-nat knee, stored at 40'
    with pytest.raises(ValueError, match='detached centre'):
        _coeffs('mean', [-40.0, -40.0], [40.0, 40.0], beta_fwd=5.0)


def test_mean_detached_is_mean_without_the_coupling():
    inside, _ = _coeffs('mean_detached', [-2.0, -1.0], [1.0, 2.0])
    ref, _ = _coeffs('mean', [-2.0, -1.0], [1.0, 2.0])
    # inside the knee the squared loss has no shared term (deviations sum to 0), so the two agree
    assert torch.allclose(inside, ref, atol=1e-4)
    past, _ = _coeffs('mean_detached', [-202.0, -2.0], [2.0, 2.0])
    coupled, _ = _coeffs('mean', [-202.0, -2.0], [2.0, 2.0])
    assert not torch.allclose(past, coupled, atol=1e-2), 'past the knee only the in-graph mean couples the rows'


def test_fwd_mean_ranks_own_samples_and_pulls_stored_rows_to_their_level():
    """'fwd_mean': the level is the replay rows' own mean. Replay at -3 and +1 (mean -1) rank about it with
    coefficients summing to zero; stored rows 80 and 90 nats up are saturated at the knee and pulled up at 40 each."""
    g, stats = _coeffs('fwd_mean', [-3.0, 1.0], [80.0, 90.0], head=50.0)
    assert torch.allclose(g[:2], torch.tensor([2.0, -2.0]), atol=1e-5), 'u - L = -2, +2: pure ranking'
    assert abs(float(g[:2].sum())) < 1e-6, 'no level push on the policy samples'
    assert torch.allclose(g[2:], torch.full((2,), -BETA), atol=1e-4), 'stored rows pulled up at the knee'
    assert float(stats['pooled_center_minus_head']) == pytest.approx(-51.0)   # the head plays no part


def test_fwd_mean_drops_a_group_with_no_policy_sample():
    """A condition present only among the stored rows has no level: its rows carry no loss, while a paired group in
    the same batch still trains."""
    rep, sto = _group([-3.0, 1.0], [80.0, 90.0], head=0.0, cid=0)
    _, lone = _group([0.0], [70.0, 75.0], head=0.0, cid=1)
    sto = {k: torch.cat([sto[k], lone[k]]) if k != 'log_pf' else None for k in sto}
    sto['log_pf'] = torch.zeros(4, requires_grad=True)
    rows, _ = pooled_condition_vargrad(rep, sto, beta=BETA, center='fwd_mean')
    rows.sum().backward()
    assert torch.allclose(sto['log_pf'].grad[2:], torch.zeros(2)), 'the unpaired group sits out'
    assert torch.allclose(sto['log_pf'].grad[:2], torch.full((2,), -BETA), atol=1e-4)
