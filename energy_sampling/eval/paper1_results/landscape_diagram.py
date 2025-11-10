import numpy as np
import plotly.graph_objects as go

def smooth_multiwell(x, centers, depths, widths):
    """Sum of Gaussians creating multiwell potential."""
    V = np.zeros_like(x)
    for c, d, w in zip(centers, depths, widths):
        V -= d * np.exp(-0.5 * ((x - c) / w) ** 2)
    return V

def normalize_range(V, ref_max=1.0, ref_min=0.0):
    """Shift and scale potential to given range."""
    V = (V - np.min(V)) / (np.max(V) - np.min(V) + 1e-9)
    return ref_min + (ref_max - ref_min) * V

def sample_from_potential(x, V, beta=5.0, n_samples=400):
    """Draw samples ∝ exp(-beta * V(x))."""
    p = np.exp(-beta * V)
    p /= np.trapz(p, x)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    inv_cdf = lambda u: np.interp(u, cdf, x)
    return inv_cdf(np.random.rand(n_samples))

def renormalize_to_equal_Z(x, V_list, beta=5.0):
    """
    Adjust all potentials by a constant so that exp(-βV) integrates to the same value for each.
    """
    Zs = [np.trapz(np.exp(-beta * V), x) for V in V_list]
    target_Z = np.mean(Zs)
    corrected = [V + (1 / beta) * np.log(Z / target_Z) for V, Z in zip(V_list, Zs)]
    return corrected

# --- domain ---
x = np.linspace(-1, 1, 4000)

# --- 1. base (data-driven) potential: shallow 3 wells ---
centers1 = np.array([-0.6, 0.0, 0.6])
depths1  = np.array([0.5, 0.6, 0.4])
widths1  = np.array([0.18, 0.15, 0.18])
V1 = smooth_multiwell(x, centers1, depths1, widths1)
V1 = normalize_range(V1, ref_max=0.6, ref_min=0.0)  # shallow overall

# --- 2. reshaped potential: slightly shifted + one well splits ---
centers2 = np.array([-0.55, -0.05, 0.35, 0.55])  # rightmost becomes double
depths2  = np.array([0.6, 0.8, 0.8, 0.6])
widths2  = np.array([0.16, 0.12, 0.10, 0.12])
V2 = smooth_multiwell(x, centers2, depths2, widths2)
V2 += 0.03 * np.sin(8 * x)
V2 = normalize_range(V2, ref_max=np.max(V1), ref_min=np.min(V1) - 0.4)

# --- 3. redistributed potential: shifted and smoothed ---
centers3 = np.array([-0.5, -0.1, 0.25, 0.55])
depths3  = np.array([0.5, 0.9, 0.7, 0.5])
widths3  = np.array([0.16, 0.10, 0.11, 0.13])
V3 = smooth_multiwell(x, centers3, depths3, widths3)
V3 += 0.02 * np.sin(10 * x + 1.2)
V3 = normalize_range(V3, ref_max=np.max(V1), ref_min=np.min(V1) - 0.4)

# --- equalize partition functions (so total probability same) ---
V1, V2, V3 = renormalize_to_equal_Z(x, [V1, V2, V3], beta=5.0)

# --- sample points from the first potential only ---
x_samples = sample_from_potential(x, V1, beta=5.0, n_samples=2000)
y_samples = np.full_like(x_samples, np.max(V1) + 0.15)  # at top of plot

# --- build figure ---
fig = go.Figure()

# Points (top axis)
fig.add_trace(go.Scatter(
    x=x_samples, y=y_samples + np.random.randn(len(x_samples)) / 50,
    mode="markers",
    marker=dict(size=6, color="#636EFA", opacity=0.25),
    showlegend=True,
    name="Prior Dataset"
))

# Potentials (aligned maxima, same normalization)
fig.add_trace(go.Scatter(
    x=x, y=V1,
    mode="lines",
    line=dict(color="#636EFA", width=3),
    name="Data-Implied"
))
fig.add_trace(go.Scatter(
    x=x, y=V2,
    mode="lines",
    line=dict(color="#EF553B", width=3),#, dash="dash"),
    opacity=0.6,
    name="Backwards-Thermalized",
))
fig.add_trace(go.Scatter(
    x=x, y=V3,
    mode="lines",
    line=dict(color="#00CC96", width=3),#, dash="dot"),
    opacity=0.6,
    name="Forward-Converged"
))

# --- layout ---
fig.update_layout(
    #height=400, width=800,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(range=[-1.05, 1.05], showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    plot_bgcolor="white",
    font_size=22,
)
fig.update_layout(
    xaxis=dict(
        range=[-1.05, 1.05],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="x-coordinate",     # X-axis label
        title_font=dict(size=16),
        ticks="",                        # no tick marks
        mirror=False,                    # only bottom line
        linecolor="black",
        linewidth=1.5,
        side="bottom",
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="Energy",        # Y-axis label
        title_font=dict(size=16),
        ticks="",                        # no tick marks
        mirror=False,                    # only left line
        linecolor="black",
        linewidth=1.5,
        side="left",
    ),
    plot_bgcolor="white",
    margin=dict(l=50, r=20, t=20, b=50)
)
fig.show()
