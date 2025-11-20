import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def smooth_multiwell(x, centers, depths, widths):
    """Sum of negative Gaussian wells."""
    V = np.zeros_like(x)
    for c, d, w in zip(centers, depths, widths):
        V -= d * np.exp(-0.5 * ((x - c) / w)**2)
    return V

def sample_from_potential(x, V, beta=12.0, n_samples=600):
    """Samples ∝ exp(-β V(x)). Warmer (beta≈12) preserves weaker modes."""
    p = np.exp(-beta * V)
    p /= np.trapz(p, x)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    inv = lambda u: np.interp(u, cdf, x)
    return inv(np.random.rand(n_samples))

# ---------------------------------------------------------------
# Domain
# ---------------------------------------------------------------
x = np.linspace(-1, 1, 4000)

# ---------------------------------------------------------------
# PHASE 1 — symmetric double well (clean & strong)
# ---------------------------------------------------------------
centers1 = np.array([-0.45, 0.45])
depths1  = np.array([2.0, 2.0])       # deep & symmetric
widths1  = np.array([0.20, 0.20])
V1 = smooth_multiwell(x, centers1, depths1, widths1)

# ---------------------------------------------------------------
# PHASE 2 — asymmetric tilt (left well deeper)
# ---------------------------------------------------------------
centers2 = centers1.copy()
depths2  = np.array([2.6, 1.0])       # left dominates strongly
widths2  = widths1
V2 = smooth_multiwell(x, centers2, depths2, widths2)

# ---------------------------------------------------------------
# PHASE 3 — asymmetry + substructure (mini double-well in left basin)
# ---------------------------------------------------------------
centers3 = np.array([
    -0.45, 0.45, -0.45
])

depths3 = np.array([
    3.0, 0.95, -0.5,
])

widths3 = np.array([
    0.2, 0.2, 0.025
])

V3 = smooth_multiwell(x, centers3, depths3, widths3)

# ---------------------------------------------------------------
# Normalize all potentials to same global y-range
# ---------------------------------------------------------------
allV = np.concatenate([V1, V2, V3])
vmin, vmax = allV.min(), allV.max()

def normalize_global(V):
    return (V - vmin) / (vmax - vmin)

V1n = normalize_global(V1)
V2n = normalize_global(V2)
V3n = normalize_global(V3)

# ---------------------------------------------------------------
# Sampling (slightly warmer to avoid killing shallower modes)
# ---------------------------------------------------------------
beta_samples = 12.0

samples1 = sample_from_potential(x, V1n, beta=beta_samples, n_samples=600)
samples2 = sample_from_potential(x, V2n, beta=beta_samples, n_samples=600)
samples3 = sample_from_potential(x, V3n, beta=beta_samples, n_samples=600)

# violin vertical placement above the potentials
y_violin = np.full(600, 1.0)

# Colors
c1 = "#636EFA"
c2 = "#EF553B"
c3 = "#00CC96"

# ---------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------
fig = go.Figure()

# -------------------- VIOLINS --------------------
fig.add_trace(go.Violin(
    x=samples1, y=y_violin,
    side="positive", orientation="h",
    width=0.45, points=False, opacity=0.55,
    fillcolor=c1, line_color=c1,
    name="Phase 1"
))
fig.add_trace(go.Violin(
    x=samples2, y=y_violin,
    side="positive", orientation="h",
    width=0.45, points=False, opacity=0.55,
    fillcolor=c2, line_color=c2,
    name="Phase 2"
))
fig.add_trace(go.Violin(
    x=samples3, y=y_violin,
    side="positive", orientation="h",
    width=0.45, points=False, opacity=0.55,
    fillcolor=c3, line_color=c3,
    name="Phase 3"
))

# -------------------- POTENTIALS --------------------
fig.add_trace(go.Scatter(
    x=x, y=V1n, mode="lines",
    line=dict(color=c1, width=4),
    showlegend=False,
    #name="Phase 1: Symmetric Double Well"
))
fig.add_trace(go.Scatter(
    x=x, y=V2n, mode="lines",
    line=dict(color=c2, width=4),
    showlegend=False,
    #name="Phase 2: Asymmetric Tilt"
))
fig.add_trace(go.Scatter(
    x=x, y=V3n, mode="lines",
    line=dict(color=c3, width=4),
    showlegend=False,
    #name="Phase 3: Substructured Landscape"
))

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
fig.update_layout(
    #height=650,
    plot_bgcolor="white",
    margin=dict(l=60, r=40, t=40, b=60),
    font=dict(size=20),
    xaxis=dict(
        range=[-1.05, 1.05],
        showgrid=False, zeroline=False, ticks="",
        title="x-coordinate",
        linecolor="black", linewidth=2,
    ),
    yaxis=dict(
        range=[0, 1.35],            # <-- extra headroom for violins
        showgrid=False, zeroline=False, ticks="",
        title="Energy",
        linecolor="black", linewidth=2,
    )
)

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

fig.show()

aa = 1
