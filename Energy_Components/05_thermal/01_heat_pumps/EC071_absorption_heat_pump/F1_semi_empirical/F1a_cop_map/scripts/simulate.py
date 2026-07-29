"""EC071 — Absorption Heat Pump — F1a — Simulation & HTML Report"""
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
            "COP vs T_gen (T_cond varied)",
            "COP vs T_cond (T_evap varied)",
            "Heat Duties vs T_gen",
            "COP Map — T_gen vs T_cond",
        ],
        vertical_spacing=0.13,
    )

    Tg = np.linspace(70, 110, 100)
    for tc in [28, 32, 36, 40, 45]:
        r = model.predict({"T_gen": Tg, "T_evap": 10.0, "T_cond": tc})
        fig.add_trace(go.Scatter(x=Tg, y=r["cop"], name=f"T_cond={tc}C"), row=1, col=1)

    Tc = np.linspace(25, 50, 100)
    for te in [2, 7, 12, 18, 25]:
        r = model.predict({"T_gen": 90.0, "T_evap": te, "T_cond": Tc})
        fig.add_trace(go.Scatter(x=Tc, y=r["cop"], name=f"T_evap={te}C"), row=1, col=2)

    r = model.predict({"T_gen": Tg, "T_evap": 10.0, "T_cond": 35.0})
    fig.add_trace(go.Scatter(x=Tg, y=r["heating_capacity_kw"], name="Q_heating", line=dict(color="firebrick")), row=2, col=1)
    fig.add_trace(go.Scatter(x=Tg, y=r["driving_heat_kw"],     name="Q_driving", line=dict(color="orange")),    row=2, col=1)
    fig.add_trace(go.Scatter(x=Tg, y=r["evaporator_heat_kw"],  name="Q_evap",    line=dict(color="steelblue")), row=2, col=1)

    Tg_grid = np.linspace(70, 110, 50)
    Tc_grid = np.linspace(25, 50, 50)
    cop_map = np.zeros((50, 50))
    for i, tg in enumerate(Tg_grid):
        r = model.predict({"T_gen": tg, "T_evap": 10.0, "T_cond": Tc_grid})
        cop_map[i, :] = r["cop"]
    fig.add_trace(go.Heatmap(x=Tc_grid, y=Tg_grid, z=cop_map, colorscale="Viridis",
                             colorbar=dict(title="COP")), row=2, col=2)

    fig.update_xaxes(title_text="T_gen (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_cond (degC)", row=1, col=2)
    fig.update_xaxes(title_text="T_gen (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_cond (degC)", row=2, col=2)
    fig.update_yaxes(title_text="COP_h", row=1, col=1)
    fig.update_yaxes(title_text="COP_h", row=1, col=2)
    fig.update_yaxes(title_text="kW_th",  row=2, col=1)
    fig.update_yaxes(title_text="T_gen (degC)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
