"""EC143 — Biomass Gasifier — F1a Equilibrium — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    ERs = np.linspace(0.20, 0.50, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Syngas Composition vs Equivalence Ratio",
            "LHV vs Equivalence Ratio",
            "Cold Gas Efficiency vs ER",
            "Composition at T=800°C vs ER (Heatmap)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Composition vs ER
    species_colors = {"CO": "blue", "H2": "red", "CO2": "green", "CH4": "purple", "N2": "gray"}
    r_ref = model.predict({"equivalence_ratio": ERs, "temperature": 800.0})
    comp = r_ref["syngas_composition"]
    for sp, color in species_colors.items():
        fig.add_trace(
            go.Scatter(x=ERs, y=np.array(comp[sp]), name=sp,
                       line=dict(color=color, width=2)),
            row=1, col=1
        )

    # Plot 2: LHV vs ER at different temperatures
    for T in [700, 800, 900, 1000]:
        r = model.predict({"equivalence_ratio": ERs, "temperature": float(T)})
        fig.add_trace(
            go.Scatter(x=ERs, y=r["lhv_syngas_mjnm3"], name=f"T={T}°C",
                       line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Cold gas efficiency vs ER
    for T in [700, 800, 900, 1000]:
        r = model.predict({"equivalence_ratio": ERs, "temperature": float(T)})
        fig.add_trace(
            go.Scatter(x=ERs, y=r["cold_gas_efficiency"], name=f"CGE T={T}°C",
                       line=dict(width=2), showlegend=False),
            row=2, col=1
        )

    # Plot 4: Heatmap — CO fraction across ER x Temperature
    Ts = np.linspace(700, 1000, 40)
    ER_grid = np.linspace(0.20, 0.50, 40)
    co_map = np.zeros((len(Ts), len(ER_grid)))
    for i, T in enumerate(Ts):
        r = model.predict({"equivalence_ratio": ER_grid, "temperature": float(T)})
        co_map[i, :] = r["syngas_composition"]["CO"]

    fig.add_trace(
        go.Heatmap(x=ER_grid, y=Ts, z=co_map,
                   colorscale="Blues", colorbar=dict(title="CO fraction"),
                   name="CO fraction"),
        row=2, col=2
    )

    # Mark design point
    r_dp = model.predict({"equivalence_ratio": 0.25, "temperature": 800.0})
    fig.add_trace(
        go.Scatter(x=[0.25], y=[float(r_dp["lhv_syngas_mjnm3"])],
                   mode="markers", marker=dict(size=12, color="black", symbol="star"),
                   name="Design Point (ER=0.25)", showlegend=True),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Equivalence Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Equivalence Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Equivalence Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Equivalence Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="Mole Fraction (-)", row=1, col=1)
    fig.update_yaxes(title_text="LHV (MJ/Nm³)", row=1, col=2)
    fig.update_yaxes(title_text="Cold Gas Efficiency (-)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Equilibrium Model<br>"
              f"<sup>Source: Zainal et al. (2001), Energy Conversion and Management</sup>",
        height=850,
        template="plotly_white",
        legend=dict(groupclick="toggleitem"),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Print summary
    r_dp = model.predict({"equivalence_ratio": 0.25, "temperature": 800.0})
    comp_dp = r_dp["syngas_composition"]
    print("\n--- Design Point Summary (ER=0.25, T=800°C) ---")
    print(f"  CO  = {comp_dp['CO']:.3f}")
    print(f"  H2  = {comp_dp['H2']:.3f}")
    print(f"  CO2 = {comp_dp['CO2']:.3f}")
    print(f"  CH4 = {comp_dp['CH4']:.3f}")
    print(f"  N2  = {comp_dp['N2']:.3f}")
    print(f"  Sum = {sum(comp_dp.values()):.4f}")
    print(f"  LHV = {float(r_dp['lhv_syngas_mjnm3']):.3f} MJ/Nm3")
    print(f"  CGE = {float(r_dp['cold_gas_efficiency']):.3f}")


if __name__ == "__main__":
    generate_report()
