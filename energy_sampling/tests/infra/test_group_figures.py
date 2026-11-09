"""eval/evaluations.py figures on a run over several space groups:

  * group_dropdown_figure: one figure, one dropdown button per group; each button shows exactly its own group's
    traces and swaps in that group's axes and subplot titles, hiding an axis its figure lacks; one figure passes
    through unchanged;
  * flow_parity_plot: 'Space group' joins the colour switcher only when the rows hold more than one group, coloured
    by the group's index on a banded (one colour per group) scale.

CPU, synthetic figures.

    python -m pytest -q tests/infra/test_group_figures.py
"""
import numpy as np
import plotly.graph_objects as go
import pytest
import torch
from plotly.subplots import make_subplots

from eval.evaluations import flow_parity_plot, group_dropdown_figure

pytestmark = pytest.mark.fast


def _fig(n_panels, title, offset):
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=[f'{title} {i}' for i in range(n_panels)])
    for i in range(n_panels):
        fig.add_trace(go.Scatter(x=[0, 1], y=[offset, offset + i]), row=1, col=i + 1)
        fig.update_xaxes(title_text=f'{title} x{i}', row=1, col=i + 1)
    return fig


def test_each_button_shows_its_own_group_and_its_axes():
    out = group_dropdown_figure({'sg2': _fig(3, 'P-1', 0.0), 'sg14': _fig(2, 'P21c', 10.0)})
    assert len(out.data) == 5
    assert [bool(t.visible) for t in out.data] == [True, True, True, False, False]   # the first group on open
    (menu,) = out.layout.updatemenus
    b2, b14 = menu.buttons
    assert b2.label == 'sg2' and b14.label == 'sg14'
    assert b14.args[0]['visible'] == [False, False, False, True, True]
    assert b14.args[1]['xaxis']['title']['text'] == 'P21c x0'
    assert b14.args[1]['xaxis3.visible'] is False                                   # P2_1/c has no third panel
    assert [a['text'] for a in b14.args[1]['annotations']] == ['P21c 0', 'P21c 1']
    assert 'xaxis3.visible' not in b2.args[1]


def test_one_figure_passes_through():
    f = _fig(2, 'P-1', 0.0)
    assert group_dropdown_figure({'sg2': f}) is f


def _parity(sg):
    n = len(sg)
    g = torch.Generator().manual_seed(0)
    log_r = torch.randn(n, generator=g)
    return flow_parity_plot(log_r, torch.zeros(n), torch.zeros(n, 3), torch.randn(n, 3, generator=g),
                            cond_inds=torch.arange(n), sg_inds=torch.tensor(sg))[0]


def test_parity_colours_by_space_group_only_on_several_groups():
    labels = lambda f: [b.label for b in (f.layout.updatemenus[0].buttons if f.layout.updatemenus else [])]
    mixed = _parity([2] * 30 + [14] * 30)
    assert 'Space group' in labels(mixed)
    btn = next(b for b in mixed.layout.updatemenus[0].buttons if b.label == 'Space group')
    colours = np.asarray(btn.args[0]['marker.color'][0])
    assert set(np.unique(colours).tolist()) == {0.0, 1.0}
    assert (btn.args[0]['marker.cmin'], btn.args[0]['marker.cmax']) == (0.0, 1.0)      # full span, not percentiles
    assert btn.args[0]['marker.colorbar.ticktext'] == [['sg2', 'sg14']]
    rew = next(b for b in mixed.layout.updatemenus[0].buttons if b.label == 'Reward')
    assert rew.args[0]['marker.colorbar.ticktext'] is None                             # plotly's own ticks restored
    assert 'Space group' not in labels(_parity([2] * 60))
