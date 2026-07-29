"""EC155 — Geothermal District Heating — F1a Heat Extraction — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_srcs = np.linspace(50, 150, 100)
    T_rets = np.linspace(20, 60, 100)
    flows  = np.linspace(5, 500, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Heat Extracted & Delivered vs T_source",
            "Heat Delivered vs Flow Rate",
            "Heat Coefficient vs T_source",
            "Delivered Heat Map: T_source × Flow Rate (kW)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Q_extracted & Q_delivered vs T_source
    for T_ret in [25, 40, 55]:
        r = model.predict({"T_source": T_srcs, "T_return": float(T_ret), "flow_rate_kgs": 50.0})
        fig.add_trace(go.Scatter(x=T_srcs, y=r["heat_extracted_kw"] / 1000.0,
                                 name=f"Q_ext T_ret={T_ret}°C", line=dict(width=2, dash="dot")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=T_srcs, y=r["heat_delivered_kw"] / 1000.0,
                                 name=f"Q_del T_ret={T_ret}°C", line=dict(width=2)),
                      row=1, col=1)

    # Plot 2: Q_delivered vs flow rate
    for T_src in [60, 80, 100, 120]:
        r = model.predict({"T_source": float(T_src), "T_return": 40.0, "flow_rate_kgs": flows})
        fig.add_trace(go.Scatter(x=flows, y=r["heat_delivered_kw"] / 1000.0,
                                 name=f"T_src={T_src}°C", line=dict(width=2)),
                      row=1, col=2)

    # Plot 3: Heat coefficient vs T_source (should be constant — good sanity check)
    for T_ret in [30, 40, 50]:
        r = model.predict({"T_source": T_srcs, "T_return": float(T_ret), "flow_rate_kgs": 50.0})
        fig.add_trace(go.Scatter(x=T_srcs, y=r["heat_coefficient"],
                                 name=f"T_ret={T_ret}°C", line=dict(width=2), showlegend=False),
                      row=2, col=1)

    # Plot 4: Heatmap
    T_src_g = np.linspace(50, 150, 40)
    flow_g  = np.linspace(5, 500, 40)
    Q_map   = np.zeros((len(T_src_g), len(flow_g)))
    for i, Ts in enumerate(T_src_g):
        r = model.predict({"T_source": float(Ts), "T_return": 40.0, "flow_rate_kgs": flow_g})
        Q_map[i, :] = r["heat_delivered_kw"] / 1000.0

    fig.add_trace(
        go.Heatmap(x=flow_g, y=T_src_g, z=Q_map,
                   colorscale="YlOrRd", colorbar=dict(title="Q_del (MW)"),
                   name="Heat Map"),
        row=2, col=2
    )

    # Design point
    r_dp = model.predict({"T_source": 80.0, "T_return": 40.0, "flow_rate_kgs": 50.0})
    fig.add_trace(
        go.Scatter(x=[80.0], y=[float(r_dp["heat_delivered_kw"]) / 1000.0],
                   mode="markers", marker=dict(size=12, color="black", symbol="star"),
                   name="Design Point"),
        row=1, col=1
    )

    fig.update_xaxes(title_text="T_source (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Flow Rate (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="T_source (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Flow Rate (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Heat (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Q_delivered (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Coefficient (-)", row=2, col=1)
    fig.update_yaxes(title_text="T_source (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Heat Extraction Model<br>"
              f"<sup>Source: Lund & Toth (2021), Geothermics; Rybach (2003)</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (T_src=80°C, T_ret=40°C, 50 kg/s) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.3f}")


if __name__ == "__main__":
    generate_report()
