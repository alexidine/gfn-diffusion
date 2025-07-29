import json
import sys

import plotly


def get_plotly_fig_size_mb(fig) -> float:
    # Convert Plotly figure to JSON string
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return sys.getsizeof(fig_json) / (1024 * 1024)
