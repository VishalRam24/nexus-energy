"""EC193 — Methanation Reactor — F1a Sabatier Equilibrium — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    Ts = np.linspace(200, 500, 100)
    Ps = np.linspace(1, 30, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "CO2 Conversion vs Temperature",
            "Conversion vs Pressure (at T=300°C)",
            "Energy Efficiency vs Temperature",
            "Conversion Map: Temperature × Pressure",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Conversion vs T at various pressures
    for P in [1, 5, 10, 20, 30]:
        r = model.predict({"temperature": Ts, "pressure": float(P), "h2_co2_ratio": 4.0})
        fig.add_trace(
            go.Scatter(x=Ts, y=r["conversion"], name=f"P={P} bar", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: Conversion vs P at T=300°C, different H2/CO2 ratios
    for ratio in [3.5, 4.0, 4.5, 5.0]:
        r = model.predict({"temperature": 300.0, "pressure": Ps, "h2_co2_ratio": ratio})
        fig.add_trace(
            go.Scatter(x=Ps, y=r["conversion"], name=f"H2/CO2={ratio}", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Efficiency vs T
    for P in [1, 10, 30]:
        r = model.predict({"temperature": Ts, "pressure": float(P)})
        fig.add_trace(
            go.Scatter(x=Ts, y=np.array(r["efficiency"]) * 100.0,
                       name=f"Eff P={P} bar", line=dict(width=2), showlegend=False),
            row=2, col=1
        )

    # Plot 4: Conversion heatmap (T vs P)
    T_g = np.linspace(200, 500, 40)
    P_g = np.linspace(1, 30, 40)
    X_map = np.zeros((len(T_g), len(P_g)))
    for i, T in enumerate(T_g):
        r = model.predict({"temperature": float(T), "pressure": P_g})
        X_map[i, :] = r["conversion"]

    fig.add_trace(
        go.Heatmap(x=P_g, y=T_g, z=X_map,
                   colorscale="Viridis", colorbar=dict(title="Conversion (-)"),
                   name="X Map"),
        row=2, col=2
    )

    # Design point
    r_dp = model.predict({"temperature": 300.0, "pressure": 10.0, "h2_co2_ratio": 4.0})
    fig.add_trace(
        go.Scatter(x=[300.0], y=[float(r_dp["conversion"])],
                   mode="markers", marker=dict(size=12, color="black", symbol="star"),
                   name="Design Point (300°C, 10 bar)", showlegend=True),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Pressure (bar)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Pressure (bar)", row=2, col=2)
    fig.update_yaxes(title_text="Conversion (-)", row=1, col=1)
    fig.update_yaxes(title_text="Conversion (-)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>CO2 + 4H2 → CH4 + 2H2O | Source: Gao et al. (2012), RSC Advances</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (T=300°C, P=10 bar, H2/CO2=4) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.4f}")


if __name__ == "__main__":
    generate_report()
