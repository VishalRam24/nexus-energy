"""EC048 — Perovskite Solar Cell — F1a — Simulation & HTML Report"""
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
        rows=2, cols=3,
        subplot_titles=[
            "Power vs Irradiance (25°C)",
            "Voc & Vmp vs Temperature",
            "Efficiency vs Irradiance",
            "Efficiency Map (G vs T)",
            "Isc vs Irradiance",
            "P-V Curve Families (1000 W/m²)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 1) Power vs Irradiance at fixed T
    G_vals = np.linspace(50, 1200, 200)
    for T in [0, 25, 45, 65]:
        r = model.predict({"irradiance": G_vals, "cell_temperature": T})
        fig.add_trace(
            go.Scatter(x=G_vals, y=r["p_mp"], name=f"T={T}°C", legendgroup="g1"),
            row=1, col=1,
        )

    # 2) Voc & Vmp vs Temperature at 1000 W/m2
    T_vals = np.linspace(-10, 80, 100)
    r_T = model.predict({"irradiance": 1000.0, "cell_temperature": T_vals})
    fig.add_trace(
        go.Scatter(x=T_vals, y=r_T["v_oc"], name="Voc", line=dict(dash="solid"), legendgroup="g2"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=T_vals, y=r_T["v_mp"], name="Vmp", line=dict(dash="dash"), legendgroup="g2"),
        row=1, col=2,
    )

    # 3) Efficiency vs Irradiance
    for T in [0, 25, 45, 65]:
        r = model.predict({"irradiance": G_vals, "cell_temperature": T})
        fig.add_trace(
            go.Scatter(x=G_vals, y=r["efficiency"] * 100, name=f"T={T}°C", legendgroup="g3", showlegend=False),
            row=1, col=3,
        )

    # 4) Efficiency heatmap
    G_grid = np.linspace(100, 1200, 40)
    T_grid = np.linspace(-10, 80, 40)
    eta_map = np.zeros((40, 40))
    for i, T in enumerate(T_grid):
        r = model.predict({"irradiance": G_grid, "cell_temperature": T})
        eta_map[i, :] = r["efficiency"] * 100
    fig.add_trace(
        go.Heatmap(x=G_grid, y=T_grid, z=eta_map, colorscale="RdYlGn",
                   colorbar=dict(title="%", x=0.63), name="Efficiency %"),
        row=2, col=1,
    )

    # 5) Isc vs Irradiance
    r_isc = model.predict({"irradiance": G_vals, "cell_temperature": 25.0})
    fig.add_trace(
        go.Scatter(x=G_vals, y=r_isc["i_sc"], name="Isc @25°C", line=dict(color="darkorange")),
        row=2, col=2,
    )

    # 6) Simplified I-V curve families (different irradiance levels)
    # Use Voc and Isc to sketch simplified I-V curves
    for G_pt in [200, 400, 600, 800, 1000]:
        r_pt = model.predict({"irradiance": float(G_pt), "cell_temperature": 25.0})
        voc = float(r_pt["v_oc"])
        isc = float(r_pt["i_sc"])
        # Approximate I-V as exponential characteristic
        v_range = np.linspace(0, voc, 100)
        # I = Isc * (1 - exp((V - Voc)/(Voc/15)))  — simplified shape
        i_curve = isc * (1 - np.exp((v_range - voc) / max(voc / 12, 0.01)))
        i_curve = np.maximum(i_curve, 0.0)
        p_curve = v_range * i_curve
        fig.add_trace(
            go.Scatter(x=v_range, y=p_curve, name=f"{G_pt} W/m²", legendgroup="g6", showlegend=True),
            row=2, col=3,
        )

    # Axis labels
    fig.update_xaxes(title_text="Irradiance (W/m²)", row=1, col=1)
    fig.update_xaxes(title_text="Cell Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Irradiance (W/m²)", row=1, col=3)
    fig.update_xaxes(title_text="Irradiance (W/m²)", row=2, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m²)", row=2, col=2)
    fig.update_xaxes(title_text="Voltage (V)", row=2, col=3)

    fig.update_yaxes(title_text="P_mp (W)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=3)
    fig.update_yaxes(title_text="Cell Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="I_sc (A)", row=2, col=2)
    fig.update_yaxes(title_text="Power (W)", row=2, col=3)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Single-Diode Model<br>"
              f"<sup>MAPbI3 perovskite, 25 cm², Eg=1.55 eV, n=1.5 | De Soto framework via pvlib</sup>",
        height=850,
        template="plotly_white",
        legend=dict(groupclick="toggleitem"),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")

    # Print STC summary
    r_stc = model.predict({"irradiance": 1000.0, "cell_temperature": 25.0})
    print("\n=== STC Performance Summary ===")
    for k, v in r_stc.items():
        unit = {"v_mp": "V", "i_mp": "A", "p_mp": "W", "v_oc": "V", "i_sc": "A", "efficiency": "-"}.get(k, "")
        print(f"  {k:12s}: {float(v):.4f} {unit}")


if __name__ == "__main__":
    generate_report()
