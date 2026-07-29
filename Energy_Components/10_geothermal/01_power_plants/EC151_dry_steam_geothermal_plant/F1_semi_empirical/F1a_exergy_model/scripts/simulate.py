"""EC151 — Dry Steam Geothermal Plant — F1a Exergy Model — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_geos = np.linspace(180, 280, 100)
    T_rejs = np.linspace(10, 50, 100)
    flows  = np.linspace(10, 200, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power & Efficiency vs T_geothermal (steam)",
            "Power vs Steam Flow Rate (T_geo=200°C)",
            "Efficiency vs T_rejection",
            "Power Map: T_geo × Flow Rate (kW)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Power vs T_geo
    for T_rej in [15, 30, 45]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 50.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["power_kw"] / 1000.0, name=f"Power T_rej={T_rej}°C",
                       line=dict(width=2)),
            row=1, col=1
        )
    for T_rej in [15, 30, 45]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 50.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["efficiency"] * 100.0, name=f"Eff T_rej={T_rej}°C",
                       line=dict(width=2, dash="dash")),
            row=1, col=1
        )

    # Plot 2: Power vs flow rate
    for T_geo in [180, 200, 220, 240, 260]:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 30.0, "flow_rate_kgs": flows})
        fig.add_trace(
            go.Scatter(x=flows, y=r["power_kw"] / 1000.0, name=f"T_geo={T_geo}°C",
                       line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Efficiency vs T_rejection
    for T_geo in [180, 200, 220, 260]:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": T_rejs, "flow_rate_kgs": 50.0})
        fig.add_trace(
            go.Scatter(x=T_rejs, y=r["efficiency"] * 100.0, name=f"Eff T_geo={T_geo}°C",
                       line=dict(width=2), showlegend=False),
            row=2, col=1
        )

    # Plot 4: Power heatmap (T_geo vs flow)
    T_geo_g = np.linspace(180, 280, 40)
    flow_g  = np.linspace(10, 200, 40)
    P_map   = np.zeros((len(T_geo_g), len(flow_g)))
    for i, Tg in enumerate(T_geo_g):
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": 30.0, "flow_rate_kgs": flow_g})
        P_map[i, :] = r["power_kw"] / 1000.0

    fig.add_trace(
        go.Heatmap(x=flow_g, y=T_geo_g, z=P_map,
                   colorscale="Hot", colorbar=dict(title="Power (MW)"),
                   name="Power Map"),
        row=2, col=2
    )

    # Design point marker
    r_dp = model.predict({"T_geothermal": 200.0, "T_rejection": 35.0, "flow_rate_kgs": 50.0})
    fig.add_trace(
        go.Scatter(x=[200.0], y=[float(r_dp["power_kw"]) / 1000.0],
                   mode="markers", marker=dict(size=12, color="black", symbol="star"),
                   name="Design Point"),
        row=1, col=1
    )

    fig.update_xaxes(title_text="T_geothermal (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Flow Rate (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="T_rejection (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Flow Rate (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW) / Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="T_geothermal (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Exergy Model<br>"
              f"<sup>Source: DiPippo (2015), Geothermal Power Plants, 4th ed., Chapter 7</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (T_geo=200°C, T_rej=35°C, 50 kg/s) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.3f}")


if __name__ == "__main__":
    generate_report()
