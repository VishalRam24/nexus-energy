"""EC095 — Thermoelectric Cooler (Peltier) — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info  = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Q_c vs Current (ΔT varied)",
            "COP vs Current (ΔT varied)",
            "Q_c vs ΔT at I=I_opt",
            "COP Map — Current vs ΔT",
        ],
        vertical_spacing=0.13,
    )

    I = np.linspace(0.0, 6.0, 200)
    Tc_design = 5.0

    for dT in [5, 15, 25, 40]:
        Th = Tc_design + dT
        r = model.predict({"current": I, "T_cold": Tc_design, "T_hot": Th})
        fig.add_trace(go.Scatter(x=I, y=r["cooling_power_w"], name=f"ΔT={dT}K"), row=1, col=1)

    for dT in [5, 15, 25, 40]:
        Th = Tc_design + dT
        r = model.predict({"current": I, "T_cold": Tc_design, "T_hot": Th})
        fig.add_trace(go.Scatter(x=I, y=r["cop"], name=f"ΔT={dT}K"), row=1, col=2)

    dT_arr = np.linspace(0, 60, 100)
    Th_arr = Tc_design + dT_arr
    Iopt = model._model.optimum_current(Tc_design, Th_arr)
    r = model.predict({"current": Iopt, "T_cold": Tc_design, "T_hot": Th_arr})
    fig.add_trace(go.Scatter(x=dT_arr, y=r["cooling_power_w"], name="Q_c at I_opt",
                             line=dict(color="firebrick"), showlegend=False), row=2, col=1)

    I_grid  = np.linspace(0.5, 6.0, 50)
    dT_grid = np.linspace(0,    50,  50)
    cop_map = np.zeros((50, 50))
    for i, dT in enumerate(dT_grid):
        Th = Tc_design + dT
        r = model.predict({"current": I_grid, "T_cold": Tc_design, "T_hot": Th})
        cop_map[i, :] = r["cop"]
    fig.add_trace(go.Heatmap(x=I_grid, y=dT_grid, z=cop_map, colorscale="RdYlGn",
                             colorbar=dict(title="COP")), row=2, col=2)

    fig.update_xaxes(title_text="Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="ΔT (K)",      row=2, col=1)
    fig.update_xaxes(title_text="Current (A)", row=2, col=2)
    fig.update_yaxes(title_text="Q_c (W)", row=1, col=1)
    fig.update_yaxes(title_text="COP",     row=1, col=2)
    fig.update_yaxes(title_text="Q_c (W)", row=2, col=1)
    fig.update_yaxes(title_text="ΔT (K)",  row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
