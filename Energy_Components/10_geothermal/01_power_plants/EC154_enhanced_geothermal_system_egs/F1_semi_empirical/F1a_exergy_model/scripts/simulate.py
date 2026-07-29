"""EC154 — EGS — F1a Exergy Model — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_geos = np.linspace(150, 350, 100)
    T_rejs = np.linspace(10, 40, 100)
    flows  = np.linspace(5, 200, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Net Power & Efficiency vs T_rock",
            "Gross vs Net Power (parasitic deduction)",
            "Net Efficiency vs T_rejection",
            "Net Power Map: T_rock × Flow Rate (MW)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Net Power + Efficiency vs T_geo
    for T_rej in [15, 25, 35]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 50.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["power_net_kw"] / 1000.0, name=f"P_net T_rej={T_rej}°C",
                       line=dict(width=2)),
            row=1, col=1
        )
    for T_rej in [25]:
        r = model.predict({"T_geothermal": T_geos, "T_rejection": float(T_rej), "flow_rate_kgs": 50.0})
        fig.add_trace(
            go.Scatter(x=T_geos, y=r["efficiency_net"] * 100.0, name=f"η_net T_rej={T_rej}°C",
                       line=dict(width=2, dash="dash")),
            row=1, col=1
        )

    # Plot 2: Gross vs Net vs Parasitic at one T_rej
    r = model.predict({"T_geothermal": T_geos, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    fig.add_trace(go.Scatter(x=T_geos, y=r["power_gross_kw"] / 1000.0, name="P_gross",
                             line=dict(width=2, color="green")), row=1, col=2)
    fig.add_trace(go.Scatter(x=T_geos, y=r["power_net_kw"] / 1000.0, name="P_net",
                             line=dict(width=2, color="blue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=T_geos, y=r["parasitic_kw"] / 1000.0, name="P_parasitic",
                             line=dict(width=2, color="red", dash="dash")), row=1, col=2)

    # Plot 3: Efficiency vs T_rejection
    for T_geo in [150, 200, 250, 300]:
        r = model.predict({"T_geothermal": float(T_geo), "T_rejection": T_rejs, "flow_rate_kgs": 50.0})
        fig.add_trace(
            go.Scatter(x=T_rejs, y=r["efficiency_net"] * 100.0, name=f"T_geo={T_geo}°C",
                       line=dict(width=2), showlegend=False),
            row=2, col=1
        )

    # Plot 4: Net Power heatmap
    T_geo_g = np.linspace(150, 350, 40)
    flow_g  = np.linspace(5, 200, 40)
    P_map   = np.zeros((len(T_geo_g), len(flow_g)))
    for i, Tg in enumerate(T_geo_g):
        r = model.predict({"T_geothermal": float(Tg), "T_rejection": 25.0, "flow_rate_kgs": flow_g})
        P_map[i, :] = r["power_net_kw"] / 1000.0

    fig.add_trace(
        go.Heatmap(x=flow_g, y=T_geo_g, z=P_map,
                   colorscale="Oranges", colorbar=dict(title="P_net (MW)"),
                   name="Power Map"),
        row=2, col=2
    )

    # Design point
    r_dp = model.predict({"T_geothermal": 200.0, "T_rejection": 25.0, "flow_rate_kgs": 50.0})
    fig.add_trace(
        go.Scatter(x=[200.0], y=[float(r_dp["power_net_kw"]) / 1000.0],
                   mode="markers", marker=dict(size=12, color="black", symbol="star"),
                   name="Design Point"),
        row=1, col=1
    )

    fig.update_xaxes(title_text="T_rock (°C)", row=1, col=1)
    fig.update_xaxes(title_text="T_rock (°C)", row=1, col=2)
    fig.update_xaxes(title_text="T_rejection (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Flow Rate (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW) / Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Net Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="T_rock (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Exergy Model<br>"
              f"<sup>Source: Tester et al. (2006) MIT/DOE; DiPippo (2015) Ch.16</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (T_geo=200°C, T_rej=25°C, 50 kg/s) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.3f}")


if __name__ == "__main__":
    generate_report()
