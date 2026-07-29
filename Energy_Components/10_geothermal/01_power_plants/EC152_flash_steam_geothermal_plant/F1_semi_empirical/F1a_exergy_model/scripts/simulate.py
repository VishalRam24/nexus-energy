"""EC152 — Flash Steam Geothermal Plant — F1a Exergy Model — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_geos = np.linspace(200, 320, 100)
    T_rejs = np.linspace(10, 60, 100)
    flows  = np.linspace(10, 500, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power & Efficiency vs T_geothermal",
            "Power vs Brine Flow Rate (T_geo=240°C)",
            "Flash Temperature vs T_geothermal (optimal)",
            "Power Map: T_geo × Flow Rate (kW)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Power + Efficiency vs T_geo
    for T_rej in [20, 40, 55]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 100.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["power_kw"] / 1000.0, name=f"Power T_rej={T_rej}°C",
                       line=dict(width=2)),
            row=1, col=1
        )
    for T_rej in [20, 40, 55]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 100.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["efficiency"] * 100.0, name=f"Eff T_rej={T_rej}°C",
                       line=dict(width=2, dash="dash")),
            row=1, col=1
        )

    # Plot 2: Power vs flow rate
    for T_geo in [200, 220, 240, 270, 300]:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": 40.0, "flow_rate_kgs": flows})
        fig.add_trace(
            go.Scatter(x=flows, y=r["power_kw"] / 1000.0, name=f"T_geo={T_geo}°C",
                       line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Optimal flash T vs T_geo
    for T_rej in [20, 40]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 100.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["T_flash_c"], name=f"T_flash T_rej={T_rej}°C",
                       line=dict(width=2)),
            row=2, col=1
        )
    # Add 45-degree line for reference (T_flash=T_geo would mean no flash)
    fig.add_trace(
        go.Scatter(x=T_geos, y=T_geos, name="T_flash=T_geo (ref)",
                   line=dict(width=1, dash="dot", color="gray"), showlegend=False),
        row=2, col=1
    )

    # Plot 4: Power heatmap (T_geo vs flow)
    T_geo_g = np.linspace(200, 320, 40)
    flow_g  = np.linspace(10, 500, 40)
    P_map   = np.zeros((len(T_geo_g), len(flow_g)))
    for i, Tg in enumerate(T_geo_g):
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": 40.0, "flow_rate_kgs": flow_g})
        P_map[i, :] = r["power_kw"] / 1000.0

    fig.add_trace(
        go.Heatmap(x=flow_g, y=T_geo_g, z=P_map,
                   colorscale="Blues", colorbar=dict(title="Power (MW)"),
                   name="Power Map"),
        row=2, col=2
    )

    # Design point
    r_dp = model.predict({"T_geothermal": 240.0, "T_rejection": 40.0, "flow_rate_kgs": 100.0})
    fig.add_trace(
        go.Scatter(x=[240.0], y=[float(r_dp["power_kw"]) / 1000.0],
                   mode="markers", marker=dict(size=12, color="black", symbol="star"),
                   name="Design Point"),
        row=1, col=1
    )

    fig.update_xaxes(title_text="T_geothermal (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Brine Flow (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="T_geothermal (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Brine Flow (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW) / Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Flash Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="T_geothermal (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Exergy Model<br>"
              f"<sup>Source: DiPippo (2015), Chapters 5-6</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (T_geo=240°C, T_rej=40°C, 100 kg/s) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.3f}")


if __name__ == "__main__":
    generate_report()
