"""
EC097 — Rankine Cycle (Steam Turbine) — F1a — Simulation & HTML Report

Generates interactive Plotly charts:
1. Efficiency vs PLR
2. Heat input vs PLR
3. Efficiency vs steam temperature
4. Steam flow vs PLR
"""

import sys
import json
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
            "Efficiency vs Steam Temperature (PLR=1.0)",
            "Steam Mass Flow vs Part-Load Ratio",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: Efficiency vs PLR
    r = model.predict({"PLR": plr})
    fig.add_trace(
        go.Scatter(x=plr, y=r["efficiency"], name="Cycle Efficiency", line=dict(width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=plr, y=np.full_like(plr, r["carnot_efficiency"]),
                   name="Carnot Limit", line=dict(dash="dash", color="red")),
        row=1, col=1,
    )

    # Plot 2: Heat input vs PLR
    fig.add_trace(
        go.Scatter(x=plr, y=r["heat_input"] / 1e6, name="Heat Input",
                   line=dict(width=2, color="orange"), showlegend=True),
        row=1, col=2,
    )

    # Plot 3: Efficiency vs steam temperature
    T_steam_range = np.linspace(400, 600, 100)
    eff_vs_T = []
    carnot_vs_T = []
    for T in T_steam_range:
        rr = model.predict({"PLR": 1.0, "T_steam": T})
        eff_vs_T.append(float(rr["efficiency"]))
        carnot_vs_T.append(float(rr["carnot_efficiency"]))
    fig.add_trace(
        go.Scatter(x=T_steam_range, y=eff_vs_T, name="Efficiency at PLR=1",
                   line=dict(width=2, color="green"), showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=T_steam_range, y=carnot_vs_T, name="Carnot",
                   line=dict(dash="dash", color="red"), showlegend=False),
        row=2, col=1,
    )

    # Plot 4: Steam flow vs PLR
    fig.add_trace(
        go.Scatter(x=plr, y=r["steam_flow"], name="Steam Flow",
                   line=dict(width=2, color="purple"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Steam Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency", row=1, col=1)
    fig.update_yaxes(title_text="Heat Input (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency", row=2, col=1)
    fig.update_yaxes(title_text="Steam Flow (kg/s)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}: {info['description']}",
        height=800,
        template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
