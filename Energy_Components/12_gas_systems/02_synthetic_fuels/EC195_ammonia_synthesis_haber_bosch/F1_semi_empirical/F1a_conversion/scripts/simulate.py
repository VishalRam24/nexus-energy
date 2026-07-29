"""EC195 — Ammonia Synthesis (Haber-Bosch) — F1a Conversion — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    Ts = np.linspace(350, 550, 100)
    Ps = np.linspace(100, 300, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Per-Pass Conversion vs Temperature",
            "Per-Pass Conversion vs Pressure",
            "Specific Energy vs Temperature",
            "Conversion Map: T × P",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Conversion vs T at various pressures
    for P in [100, 150, 200, 250, 300]:
        r = model.predict({"temperature": Ts, "pressure": float(P)})
        fig.add_trace(
            go.Scatter(x=Ts, y=r["conversion_per_pass"], name=f"P={P} bar", line=dict(width=2)),
            row=1, col=1
        )
        # Also add equilibrium limit (dashed)
        X_eq = np.array([float(model._model.equilibrium_conversion(float(T), float(P))) for T in Ts])
        fig.add_trace(
            go.Scatter(x=Ts, y=X_eq, name=f"X_eq P={P}bar",
                       line=dict(width=1.5, dash="dot"), showlegend=False),
            row=1, col=1
        )

    # Plot 2: Conversion vs P at different temperatures
    for T in [400, 430, 450, 480, 500]:
        r = model.predict({"temperature": float(T), "pressure": Ps})
        fig.add_trace(
            go.Scatter(x=Ps, y=r["conversion_per_pass"], name=f"T={T}°C", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Specific energy vs T
    for P in [150, 200, 250]:
        r = model.predict({"temperature": Ts, "pressure": float(P)})
        fig.add_trace(
            go.Scatter(x=Ts, y=r["energy_gj_per_ton"],
                       name=f"E P={P} bar", line=dict(width=2), showlegend=False),
            row=2, col=1
        )

    # Plot 4: Conversion heatmap
    T_g = np.linspace(350, 550, 40)
    P_g = np.linspace(100, 300, 40)
    X_map = np.zeros((len(T_g), len(P_g)))
    for i, T in enumerate(T_g):
        r = model.predict({"temperature": float(T), "pressure": P_g})
        X_map[i, :] = r["conversion_per_pass"]

    fig.add_trace(
        go.Heatmap(x=P_g, y=T_g, z=X_map,
                   colorscale="YlOrRd", colorbar=dict(title="X per pass"),
                   name="X Map"),
        row=2, col=2
    )

    # Design point
    r_dp = model.predict({"temperature": 450.0, "pressure": 200.0})
    fig.add_trace(
        go.Scatter(x=[450.0], y=[float(r_dp["conversion_per_pass"])],
                   mode="markers", marker=dict(size=14, color="black", symbol="star"),
                   name="Design Point (450°C, 200 bar)", showlegend=True),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Pressure (bar)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Pressure (bar)", row=2, col=2)
    fig.update_yaxes(title_text="Conversion per pass (-)", row=1, col=1)
    fig.update_yaxes(title_text="Conversion per pass (-)", row=1, col=2)
    fig.update_yaxes(title_text="Specific Energy (GJ/tNH3)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>N2 + 3H2 → 2NH3 | Source: Appl (2011), Ullmann's Encyclopedia</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (T=450°C, P=200 bar) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.4f}")


if __name__ == "__main__":
    generate_report()
