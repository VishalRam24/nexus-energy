"""EC070 — Water-Source Heat Pump — F1b Part-Load — Simulation & HTML Report"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "COP vs PLR at Different Source Temps",
            "COP vs PLR at Different Sink Temps",
            "COP vs Water Flow Rate (PLR=1)",
            "COP Degradation Factor vs PLR",
        ],
        vertical_spacing=0.14)

    plr = np.linspace(0.05, 1.0, 60)

    # COP vs PLR, different source temps
    for t_src in [5.0, 10.0, 15.0, 20.0, 25.0]:
        r = model.predict({"T_source": t_src, "T_sink": 45.0, "part_load_ratio": plr})
        fig.add_trace(go.Scatter(x=plr, y=r["cop"],
                                  name=f"Ts={t_src:.0f}°C"), row=1, col=1)

    # COP vs PLR, different sink temps
    for t_sink in [30.0, 40.0, 50.0, 60.0]:
        r = model.predict({"T_source": 15.0, "T_sink": t_sink, "part_load_ratio": plr})
        fig.add_trace(go.Scatter(x=plr, y=r["cop"],
                                  name=f"Tk={t_sink:.0f}°C",
                                  showlegend=False), row=1, col=2)

    # COP vs flow rate
    flow_range = np.linspace(0.5, 6.0, 60)
    r = model.predict({"T_source": 15.0, "T_sink": 45.0,
                       "part_load_ratio": 1.0, "water_flow_rate_ls": flow_range})
    fig.add_trace(go.Scatter(x=flow_range, y=r["cop"], name="COP vs flow",
                              showlegend=False), row=2, col=1)
    # Mark rated point
    fig.add_trace(go.Scatter(x=[3.0], y=[float(model.predict(
        {"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": 1.0,
         "water_flow_rate_ls": 3.0})["cop"])],
        mode="markers", marker=dict(size=10, color="red"),
        name="Rated flow", showlegend=False), row=2, col=1)

    # COP degradation factor
    r = model.predict({"T_source": 15.0, "T_sink": 45.0, "part_load_ratio": plr})
    fig.add_trace(go.Scatter(x=plr, y=r["cop_degradation_factor"],
                              name="COP degradation", showlegend=False), row=2, col=2)
    # Mark PLR_min
    fig.add_vline(x=0.30, line_dash="dash", line_color="red",
                  annotation_text="PLR_min=0.30", row=2, col=2)

    fig.update_xaxes(title_text="Part Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Part Load Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Water Flow Rate (L/s)", row=2, col=1)
    fig.update_xaxes(title_text="Part Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="COP", row=2, col=1)
    fig.update_yaxes(title_text="COP Degradation Factor", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Part-Load + Flow Rate",
        height=800, template="plotly_white"
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
