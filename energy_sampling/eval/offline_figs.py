import numpy as np
import torch
from plotly import graph_objects as go, express as px


def create_energy_distribution_plot(sg_sampling_dict):
    """Figure 1: Distribution of energies as stacked violin plots"""
    fig = go.Figure()

    space_groups = sorted(sg_sampling_dict.keys())

    for i, sg in enumerate(space_groups):
        energies = sg_sampling_dict[sg]['energies'].flatten()

        fig.add_trace(go.Violin(
            #y=[f'SG={sg}'],  # y-axis position (space group number)
            x=energies,  # x-axis values (energy distribution)
            name=f'SG {sg}',
            orientation='h',  # horizontal orientation
            side='positive',
            width=0.8,
            points=False,
            meanline_visible=True,
            line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
        ))

    fig.update_layout(
        title='Energy Distributions by Space Group',
        xaxis_title='Energy',
        yaxis_title='Space Group Number',
        yaxis=dict(
            tickmode='array',
            tickvals=space_groups,
            ticktext=[f'SG {sg}' for sg in space_groups]
        ),
        showlegend=False,
    )

    return fig


def create_density_distribution_plot(sg_sampling_dict):
    """Figure 2: Distribution of densities as stacked violin plots"""
    fig = go.Figure()

    space_groups = sorted(sg_sampling_dict.keys())

    for i, sg in enumerate(space_groups):
        densities = sg_sampling_dict[sg]['densities'].flatten()

        fig.add_trace(go.Violin(
            #y=[sg] * len(densities),  # y-axis position (space group number)
            x=densities,  # x-axis values (density distribution)
            name=f'SG {sg}',
            orientation='h',  # horizontal orientation
            side='positive',
            width=0.8,
            points=False,
            meanline_visible=True,
            line_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
        ))

    fig.update_layout(
        title='Density Distributions by Space Group',
        xaxis_title='Density',
        yaxis_title='Space Group Number',
        yaxis=dict(
            tickmode='array',
            tickvals=space_groups,
            ticktext=[f'SG {sg}' for sg in space_groups]
        ),
        showlegend=False,
    )

    return fig


def create_cell_params_variance_plot(sg_sampling_dict):
    """Figure 3: Cell parameter variances as stacked bar plots"""
    fig = go.Figure()
    space_groups = sorted(sg_sampling_dict.keys())
    colors = px.colors.qualitative.Set1

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    vertical_step = 0
    for i, sg in enumerate(space_groups):
        params = sg_sampling_dict[sg]['cell_params']
        params = params.reshape(params.shape[0] * params.shape[1], params.shape[2])
        vars = params.var(0)
        # Histogram binning
        fig.add_trace(go.Bar(
            y=vars,
            x=lattice_features,
            base=vertical_step,
            orientation='v',
            name=f'SG {sg}',
            marker_color=colors[i % len(colors)],
            showlegend=False,
            text=[f"{v:.2e}" for v in vars],
            textposition='auto',
            opacity=0.7,
        ))
        fig.add_annotation(
            xref='paper', yref='y',
            x=-0.022,  # slightly left of the plot
            y=vertical_step + np.amax(vars) * 0.25,
            text=f"SG {sg}",
            showarrow=False,
            font=dict(size=28),
            align='right',
        )
        vertical_step += np.amax(vars)

    fig.update_layout(
        title='Per-Dim Variance by Space Group',
        xaxis_title='Dimension',
        # yaxis_title='Space Group',
        barmode='overlay',  # bars can overlap without stacking additively
        bargap=0.05,
        barcornerradius=15,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    return fig


def crystal_sample_funnel_plot(packing_coeff,
                               energies,
                               dists,
                               ref_packing_coeff,
                               ref_energies):
    fig = go.Figure()
    fontsize = 26
    fig.add_scatter(x=packing_coeff, y=energies,
                    mode='markers', marker_size=12,
                    showlegend=True,
                    marker_color=dists.clip(max=torch.quantile(dists, 0.99)).log10().cpu().detach(),
                    # 'blue',
                    marker_colorbar=dict(title=dict(text="log RDF EMD")),
                    marker_colorscale='bluered',
                    opacity=0.6, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples',
                    )

    fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_energies.cpu(), mode='markers',
                    marker_color='yellow',
                    marker_size=25, marker_line_color='black', marker_line_width=2,
                    name='Experimental Sample')

    fig.update_layout(xaxis1_title='Packing Coefficient', yaxis1_title='LJ Energy',
                      )
    fig.update_annotations(font=dict(size=fontsize))
    fig.update_layout(font_size=fontsize)
    fig.update_layout(
        coloraxis2=dict(
            colorscale='bluered',
            colorbar=dict(
                title='log RDF EMD',
                x=1  # shift it to the right so it doesn't overlap
            )
        )
    )
    fig.update_layout(legend_orientation='h')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,  # Move the legend above the plot
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100),  # Increase top margin to make space
        yaxis_range=[min(energies.min(), ref_energies.min()) - 1, 0]  # show only bound states
    )

    return fig
