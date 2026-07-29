"""EC068 — ASHP — F2a Vapor Compression SS — Simulation & HTML Report"""
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
            "COP vs Evaporator Temp (vapor cycle)",
            "Compressor Power vs Evaporator Temp",
            "Mass Flow Rate vs Conditions",
            "COP Map (Evap vs Cond Temp)",
        ],
        vertical_spacing=0.14,
    )

    T_evaps = np.linspace(-20, 15, 30)

    # Row 1, Col 1: COP vs T_evap for different T_cond
    for T_cond in [30, 35, 45, 55]:
        cops = []
        for Te in T_evaps:
            r = model.predict({"T_evap_degC": float(Te), "T_cond_degC": T_cond})
            cops.append(r["cop"])
        fig.add_trace(
            go.Scatter(x=T_evaps, y=cops, name=f"T_cond={T_cond}C", mode="lines"),
            row=1, col=1,
        )

    # Row 1, Col 2: Compressor power vs T_evap
    for T_cond in [35, 45, 55]:
        powers = []
        for Te in T_evaps:
            r = model.predict({"T_evap_degC": float(Te), "T_cond_degC": T_cond})
            powers.append(r["compressor_power_kw"])
        fig.add_trace(
            go.Scatter(x=T_evaps, y=powers, name=f"W T_c={T_cond}C", mode="lines", showlegend=False),
            row=1, col=2,
        )

    # Row 2, Col 1: Mass flow vs T_evap
    for T_cond in [35, 45, 55]:
        flows = []
        for Te in T_evaps:
            r = model.predict({"T_evap_degC": float(Te), "T_cond_degC": T_cond})
            flows.append(r["mass_flow_kg_s"])
        fig.add_trace(
            go.Scatter(x=T_evaps, y=flows, name=f"m T_c={T_cond}C", mode="lines", showlegend=False),
            row=2, col=1,
        )

    # Row 2, Col 2: COP heatmap
    T_evap_grid = np.linspace(-20, 15, 25)
    T_cond_grid = np.linspace(25, 60, 25)
    cop_map = np.zeros((len(T_evap_grid), len(T_cond_grid)))
    for i, Te in enumerate(T_evap_grid):
        for j, Tc in enumerate(T_cond_grid):
            if Te < Tc - 5:
                r = model.predict({"T_evap_degC": float(Te), "T_cond_degC": float(Tc)})
                cop_map[i, j] = r["cop"]
            else:
                cop_map[i, j] = np.nan
    fig.add_trace(
        go.Heatmap(x=T_cond_grid, y=T_evap_grid, z=cop_map, colorscale="Viridis", name="COP"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="T_evap (C)", row=1, col=1)
    fig.update_xaxes(title_text="T_evap (C)", row=1, col=2)
    fig.update_xaxes(title_text="T_evap (C)", row=2, col=1)
    fig.update_xaxes(title_text="T_cond (C)", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="kW_e", row=1, col=2)
    fig.update_yaxes(title_text="kg/s", row=2, col=1)
    fig.update_yaxes(title_text="T_evap (C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Vapor Compression Cycle",
        height=900, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
