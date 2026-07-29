"""
EC099 — Stirling Engine — F1a — Simulation & HTML Report
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required: pip install plotly")
    sys.exit(1)


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    plr = np.linspace(0.0, 1.0, 200)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cycle Efficiency vs Part-Load Ratio",
            "Heat Input vs Part-Load Ratio",
            "Efficiency vs Hot-Side Temperature (PLR=1)",
            "Efficiency vs Cold-Side Temperature (PLR=1)",
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    r = model.predict({"PLR": plr})
    fig.add_trace(
        go.Scatter(x=plr, y=r["efficiency"], name="Cycle η",
                   line=dict(width=2, color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=plr, y=np.full_like(plr, float(r["carnot_efficiency"])),
                   name="Carnot Limit",
                   line=dict(dash="dash", color="red")),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=plr, y=r["heat_input"] / 1e3, name="Q_in",
                   line=dict(width=2, color="#ff7f0e"), showlegend=False),
        row=1, col=2,
    )

    T_h_range = np.linspace(400, 800, 80)
    eff_T_h, carnot_T_h = [], []
    for T in T_h_range:
        rr = model.predict({"PLR": 1.0, "T_h": T})
        eff_T_h.append(float(rr["efficiency"]))
        carnot_T_h.append(float(rr["carnot_efficiency"]))
    fig.add_trace(
        go.Scatter(x=T_h_range, y=eff_T_h, name="η at PLR=1",
                   line=dict(width=2, color="#2ca02c"), showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=T_h_range, y=carnot_T_h, name="Carnot",
                   line=dict(dash="dash", color="red"), showlegend=False),
        row=2, col=1,
    )

    T_c_range = np.linspace(5, 80, 80)
    eff_T_c, carnot_T_c = [], []
    for T in T_c_range:
        rr = model.predict({"PLR": 1.0, "T_c": T})
        eff_T_c.append(float(rr["efficiency"]))
        carnot_T_c.append(float(rr["carnot_efficiency"]))
    fig.add_trace(
        go.Scatter(x=T_c_range, y=eff_T_c, name="η at PLR=1",
                   line=dict(width=2, color="#9467bd"), showlegend=False),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(x=T_c_range, y=carnot_T_c, name="Carnot",
                   line=dict(dash="dash", color="red"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=2)
    fig.update_xaxes(title_text="T_hot (°C)",      row=2, col=1)
    fig.update_xaxes(title_text="T_cold (°C)",     row=2, col=2)
    fig.update_yaxes(title_text="Efficiency",      row=1, col=1)
    fig.update_yaxes(title_text="Heat Input (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency",      row=2, col=1)
    fig.update_yaxes(title_text="Efficiency",      row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}: {info['description']}",
        height=820,
        template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
