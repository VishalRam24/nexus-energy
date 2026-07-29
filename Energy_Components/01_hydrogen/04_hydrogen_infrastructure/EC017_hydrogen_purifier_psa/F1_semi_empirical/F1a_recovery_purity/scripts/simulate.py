"""EC017 — Hydrogen Purifier (PSA) — F1a Recovery-Purity — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "H2 Recovery vs Feed Pressure",
            "Specific Energy vs Feed Pressure",
            "Recovery vs Feed H2 Fraction (20 bar)",
            "Product Flow vs Feed Flow",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P_vals = np.linspace(5, 80, 100)
    y_vals = np.linspace(0.3, 0.99, 100)

    # Plot 1: Recovery vs Pressure at different feed compositions
    y_feed_cases = [0.5, 0.75, 0.9]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for y, c in zip(y_feed_cases, colors):
        r = model.predict({
            "feed_flow_kg_s": 0.1,
            "feed_h2_fraction": y,
            "feed_pressure_bar": P_vals,
            "target_purity": 0.9999,
        })
        fig.add_trace(go.Scatter(
            x=P_vals, y=r["recovery"] * 100,
            name=f"y_H2={y:.0%}", line=dict(color=c)
        ), row=1, col=1)

    # Plot 2: Specific energy vs pressure
    r_ref = model.predict({
        "feed_flow_kg_s": 0.1, "feed_h2_fraction": 0.75,
        "feed_pressure_bar": P_vals, "target_purity": 0.9999,
    })
    fig.add_trace(go.Scatter(
        x=P_vals, y=r_ref["specific_energy_kWh_per_kg"],
        name="Specific Energy (y=75%)", line=dict(color="#9467bd")
    ), row=1, col=2)

    # Plot 3: Recovery vs feed H2 fraction
    r_y = model.predict({
        "feed_flow_kg_s": 0.1, "feed_h2_fraction": y_vals,
        "feed_pressure_bar": 20.0, "target_purity": 0.9999,
    })
    fig.add_trace(go.Scatter(
        x=y_vals * 100, y=r_y["recovery"] * 100,
        name="Recovery (20 bar, 99.99%)", line=dict(color="#8c564b")
    ), row=2, col=1)

    # Plot 4: Product flow vs feed flow
    F_feeds = np.linspace(0.001, 1.0, 100)
    r_flow = model.predict({
        "feed_flow_kg_s": F_feeds, "feed_h2_fraction": 0.75,
        "feed_pressure_bar": 20.0, "target_purity": 0.9999,
    })
    fig.add_trace(go.Scatter(
        x=F_feeds * 3600, y=r_flow["product_flow_kg_s"] * 3600,
        name="Product H2 (75% feed, 20 bar)", line=dict(color="#e377c2")
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=F_feeds * 3600, y=r_flow["tail_gas_flow_kg_s"] * 3600,
        name="Tail gas", line=dict(color="#7f7f7f", dash="dash")
    ), row=2, col=2)

    fig.update_xaxes(title_text="Feed Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Feed Pressure (bar)", row=1, col=2)
    fig.update_xaxes(title_text="Feed H2 Fraction (%)", row=2, col=1)
    fig.update_xaxes(title_text="Feed Flow (kg/h)", row=2, col=2)
    fig.update_yaxes(title_text="H2 Recovery (%)", row=1, col=1)
    fig.update_yaxes(title_text="Specific Energy (kWh/kg H2)", row=1, col=2)
    fig.update_yaxes(title_text="H2 Recovery (%)", row=2, col=1)
    fig.update_yaxes(title_text="Flow (kg/h)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>{info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
