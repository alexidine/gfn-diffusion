from typing import Optional

import torch
import numpy as np
from einops import rearrange
from torch import nn
import math
from mxtaltools.models.modules.components import scalarMLP


class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(
            self,
            in_shape=2,
            out_shape=2,
            num_layers=2,
            channels=128,
            zero_init=True,
    ):
        super().__init__()

        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int,
                 dim: int,
                 hidden_dim: int = 64,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 bias: Optional[bool] = True):
        super(TimeEncoding, self).__init__()

        pe = torch.arange(1, harmonics_dim + 1).float().unsqueeze(0) * 2 * math.pi

        self.t_model = scalarMLP(
            layers=1,
            input_dim=2 * harmonics_dim,
            filters=hidden_dim,
            output_dim=dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )
        self.register_buffer('pe', pe)

    def forward(self, t: torch.Tensor = None):
        """
        Arguments:
            t: torch.Tensor
            adjusted to work with tensor t
        """
        t_sin = (t[:, None] * self.pe[0]).sin()
        t_cos = (t[:, None] * self.pe[0]).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncoding(nn.Module):
    def __init__(self, s_dim: int,
                 layers: int,
                 hidden_dim: int = 64,
                 conditioning_dim: int = 0,
                 s_emb_dim: int = 64,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 bias: Optional[bool] = True,
                 extra_dim: int = 0,
                 ):
        super(StateEncoding, self).__init__()

        # A second input block: features of the state computed outside the latent (GFN's
        # state_features_dim). Projected and normalised per row before it joins the state and
        # the condition, so its scale is its own business and no row depends on another.
        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.extra_in = nn.Sequential(nn.Linear(extra_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        self.x_model = scalarMLP(
            layers=layers,
            input_dim=s_dim + conditioning_dim + (hidden_dim if extra_dim > 0 else 0),
            filters=hidden_dim,
            output_dim=s_emb_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

    def forward(self, s, conditioning=None, extra=None):
        if conditioning is not None:
            model_inputs = torch.cat([s, conditioning], dim=-1)
        else:
            model_inputs = s
        if self.extra_dim > 0:
            if extra is None:
                raise ValueError('this state encoder was built with extra_dim > 0 and was given no extra input')
            model_inputs = torch.cat([model_inputs, self.extra_in(extra)], dim=-1)
        return self.x_model(model_inputs)


class PolicyModel(nn.Module):
    def __init__(self, s_dim: int,
                 s_emb_dim: int,
                 t_dim: int,
                 hidden_dim: int = 64,
                 layers: int = 4,
                 out_dim: int = None,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 bias: Optional[bool] = True,
                 zero_init: bool = False):
        super(PolicyModel, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        self.model = scalarMLP(
            layers=layers,
            input_dim=s_emb_dim + t_dim,
            filters=hidden_dim,
            output_dim=out_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

        if zero_init:
            self.model.output_layer.weight.data.fill_(0.0)
            #self.model.output_layer.bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class JointPolicy(nn.Module):
    def __init__(self, s_dim: int,
                 s_emb_dim: int,
                 t_dim: int,
                 hidden_dim: int = 64,
                 layers: int = 4,
                 out_dim: int = None,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 bias: Optional[bool] = True,
                 zero_init: bool = False):
        super(JointPolicy, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        self.model = scalarMLP(
            layers=layers,
            input_dim=s_emb_dim + t_dim,
            filters=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

        self.forward_model = scalarMLP(
            layers=2,
            input_dim=hidden_dim,
            filters=hidden_dim,
            output_dim=out_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )
        self.backward_model = scalarMLP(
            layers=2,
            input_dim=hidden_dim,
            filters=hidden_dim,
            output_dim=out_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

        if zero_init:
            self.model.output_layer.weight.data.fill_(0.0)
            self.model.output_layer.bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))

    def forward_policy(self, s, t):
        state_embedding = self.model(torch.cat([s, t], dim=-1))
        return self.forward_model(state_embedding)

    def backward_policy(self, s, t):
        state_embedding = self.model(torch.cat([s, t], dim=-1))
        return self.backward_model(state_embedding)


class LearnableScalar(nn.Module):
    def __init__(self, init_value=0.0, device=None):
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(init_value, device=device))

    def forward(self, *args, **kwargs):
        return self.scalar


class NoneModule(nn.Module):
    def __init__(self, device=None):
        super().__init__()

    def forward(self, *args, **kwargs):
        return None


class FlowModel(nn.Module):
    def __init__(self, conditioning_dim: int,
                 hidden_dim: int = 64,
                 layers: int = 4,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 bias: Optional[bool] = True,
                 out_dim: int = 1):
        super(FlowModel, self).__init__()

        # self.model = nn.Sequential(
        #     nn.Linear(s_emb_dim + t_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, out_dim)
        # )
        self.model = scalarMLP(
            layers=layers,
            input_dim=conditioning_dim,
            filters=hidden_dim,
            output_dim=out_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

    def forward(self, z):
        return self.model(z)


class ForceGate(nn.Module):
    """Per-coordinate multiplier on a force term in a kernel mean, as a function of time.

        gate(t) = init + net(t_emb)            [B, out_dim]

    `net`'s output layer is zero at construction, so the gate starts at `init` for every
    time and coordinate and moves only by training. `learned=False` builds no parameters
    and holds the gate at `init`.
    """

    def __init__(self, t_dim: int, out_dim: int, init: float = 0.0, hidden_dim: int = 64,
                 learned: bool = True):
        super(ForceGate, self).__init__()
        self.out_dim = out_dim
        self.init = float(init)
        self.learned = learned
        if learned:
            self.net = nn.Sequential(nn.Linear(t_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
            self.net[-1].weight.data.fill_(0.0)
            self.net[-1].bias.data.fill_(0.0)

    def forward(self, t_emb):
        if not self.learned:
            return t_emb.new_full((t_emb.shape[0], self.out_dim), self.init)
        return self.init + self.net(t_emb)

    def restart(self):
        """Back to `init` at every time: the output layer is zeroed, as at construction."""
        if self.learned:
            self.net[-1].weight.data.fill_(0.0)
            self.net[-1].bias.data.fill_(0.0)


class LangevinScalingModel(nn.Module):
    def __init__(self, s_emb_dim: int,
                 t_dim: int,
                 hidden_dim: int = 64,
                 layers: int = 3,
                 out_dim: int = 1,
                 dropout: Optional[float] = 0,
                 norm: Optional[str] = None,
                 bias: Optional[bool] = True,
                 zero_init: bool = False):
        super(LangevinScalingModel, self).__init__()

        # self.model = nn.Sequential(
        #     nn.Linear(s_emb_dim + t_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, out_dim)
        # )
        self.model = scalarMLP(
            layers=layers,
            input_dim=s_emb_dim + t_dim,
            filters=hidden_dim,
            output_dim=out_dim,
            dropout=dropout,
            norm=norm,
            bias=bias,
        )

        if zero_init:
            self.model.output_layer.weight.data.fill_(0.0)
            #self.model.output_layer.bias.data.fill_(0.01)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
        super(TimeEncodingPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncodingPIS, self).__init__()

        self.x_model = nn.Linear(s_dim, s_emb_dim)

    def forward(self, s):
        return self.x_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(JointPolicyPIS, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class FlowModelPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(FlowModelPIS, self).__init__()

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class LangevinScalingModelPIS(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 3,
                 zero_init: bool = False):
        super(LangevinScalingModelPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.register_buffer('pe', pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t):
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)
