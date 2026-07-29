"""EC122 — Pumped Hydro Storage — F1a — Simulation & HTML Report"""
import sys, json, numpy as np
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
            "Generation Power vs Flow Rate (multiple heads)",
            "Pumping Power vs Flow Rate (multiple heads)",
            "Power vs Head (Q=50 m³/s)",
            "Energy Capacity vs Head (V_res=5×10⁶ m³)",
        ],
        vertical_spacing=0.14,
    )

    Q_range = np.linspace(5, 150, 100)
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # Plot 1: Generation power vs flow rate
    for i, H in enumerate([100, 200, 300, 400, 500]):
        r = model.predict({"mode": "generate", "flow_rate": Q_range, "head": H})
        fig.add_trace(go.Scatter(
            x=Q_range, y=r["power_kw"] / 1000,
            name=f"H={H} m", line=dict(color=colors[i]),
            legendgroup=f"H{H}", showlegend=True,
        ), row=1, col=1)

    # Plot 2: Pumping power vs flow rate
    for i, H in enumerate([100, 200, 300, 400, 500]):
        r = model.predict({"mode": "pump", "flow_rate": Q_range, "head": H})
        fig.add_trace(go.Scatter(
            x=Q_range, y=r["power_kw"] / 1000,
            name=f"H={H} m", line=dict(color=colors[i], dash="dash"),
            legendgroup=f"H{H}", showlegend=False,
        ), row=1, col=2)

    # Plot 3: Power vs head at fixed Q
    H_range = np.linspace(50, 600, 100)
    r_gen = model.predict({"mode": "generate", "flow_rate": 50.0, "head": H_range})
    r_pump = model.predict({"mode": "pump", "flow_rate": 50.0, "head": H_range})
    fig.add_trace(go.Scatter(
        x=H_range, y=r_gen["power_kw"] / 1000,
        name="Generation", line=dict(color="#636EFA"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=H_range, y=r_pump["power_kw"] / 1000,
        name="Pumping", line=dict(color="#EF553B", dash="dash"),
    ), row=2, col=1)

    # Add round-trip efficiency annotation
    rte = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})["round_trip_eta"]
    fig.add_annotation(
        text=f"Round-trip η = {rte:.3f} ({rte*100:.1f}%)",
        xref="paper", yref="paper", x=0.72, y=0.45,
        showarrow=False, font=dict(size=12, color="#2ca02c"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="#2ca02c", borderwidth=1,
    )

    # Plot 4: Energy capacity vs head
    fig.add_trace(go.Scatter(
        x=H_range, y=r_gen["energy_capacity_gwh"],
        name="Energy Capacity", line=dict(color="#00CC96"),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Flow Rate Q (m³/s)", row=1, col=1)
    fig.update_xaxes(title_text="Flow Rate Q (m³/s)", row=1, col=2)
    fig.update_xaxes(title_text="Head H (m)", row=2, col=1)
    fig.update_xaxes(title_text="Head H (m)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Energy Capacity (GWh)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Round-Trip Model",
        height=800,
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=0.98),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Print summary table
    print("\n--- Summary at design point (Q=50 m³/s, H=300 m) ---")
    r_g = model.predict({"mode": "generate", "flow_rate": 50.0, "head": 300.0})
    r_p = model.predict({"mode": "pump", "flow_rate": 50.0, "head": 300.0})
    print(f"  Generation power : {float(r_g['power_kw'])/1000:.2f} MW")
    print(f"  Pumping power    : {float(r_p['power_kw'])/1000:.2f} MW")
    print(f"  Round-trip eta   : {r_g['round_trip_eta']:.4f}")
    print(f"  Energy capacity  : {float(r_g['energy_capacity_gwh']):.3f} GWh")


if __name__ == "__main__":
    generate_report()
