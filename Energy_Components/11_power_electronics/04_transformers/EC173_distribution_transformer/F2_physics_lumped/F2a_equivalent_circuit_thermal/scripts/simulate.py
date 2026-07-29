"""
EC173 -- Distribution Transformer -- F2a
Optional Plotly report: efficiency-vs-load curve + 24h daily thermal transient.
Plotly import is wrapped so its absence does not crash anything.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel
from model import DistributionTransformerF2a


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # Efficiency curve
    K = np.linspace(0.01, 1.2, 200)
    eta = m.efficiency(K, 1.0, 75.0, 1.0)
    K_opt = m.optimal_load_fraction()

    # 24h daily thermal transient
    r = cm.predict({"daily": True, "ambient_temperature": 25.0,
                    "dt": 300.0, "duration_s": 86400.0})
    t_h = r["t"] / 3600.0

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] plotly unavailable ({e}); printing summary instead.")
        print(f"Peak efficiency {eta.max()*100:.3f}% at PLR={K_opt:.3f}")
        print(f"Daily peak hot-spot {r['T_hot_spot'].max():.1f} degC")
        return None

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Efficiency vs per-unit load", "24 h hot-spot / top-oil transient"))
    fig.add_trace(go.Scatter(x=K, y=eta * 100, name="efficiency"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[K_opt], y=[float(m.efficiency(K_opt, 1.0, 75.0, 1.0)) * 100],
                             mode="markers", name="PLR_opt"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=r["T_hot_spot"], name="hot-spot"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_h, y=r["T_top_oil"], name="top-oil"), row=1, col=2)
    fig.update_xaxes(title_text="K (pu)", row=1, col=1)
    fig.update_yaxes(title_text="eta (%)", row=1, col=1)
    fig.update_xaxes(title_text="hour", row=1, col=2)
    fig.update_yaxes(title_text="degC", row=1, col=2)
    fig.update_layout(title="EC173 Distribution Transformer F2a")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
