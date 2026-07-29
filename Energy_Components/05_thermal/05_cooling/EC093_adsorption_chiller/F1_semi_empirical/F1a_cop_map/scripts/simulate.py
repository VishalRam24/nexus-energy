"""EC093 — Adsorption Chiller — F1a — Simulation & HTML Report"""
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
            "COP vs T_hot (T_cool varied)",
            "COP vs T_cool (T_chilled varied)",
            "Heat duties vs T_hot",
            "COP Map — T_hot vs T_cool",
        ],
        vertical_spacing=0.13,
    )

    Th = np.linspace(55, 95, 100)
    for tc in [25, 30, 35, 40]:
        r = model.predict({"T_hot": Th, "T_cool": tc, "T_chilled": 14.0})
        fig.add_trace(go.Scatter(x=Th, y=r["cop"], name=f"T_cool={tc}C"), row=1, col=1)

    Tc = np.linspace(22, 40, 100)
    for tx in [6, 10, 14, 18]:
        r = model.predict({"T_hot": 85.0, "T_cool": Tc, "T_chilled": tx})
        fig.add_trace(go.Scatter(x=Tc, y=r["cop"], name=f"T_chw={tx}C"), row=1, col=2)

    r = model.predict({"T_hot": Th, "T_cool": 30.0, "T_chilled": 14.0})
    q_cool_arr = np.full_like(Th, float(r["cooling_kw"]))
    fig.add_trace(go.Scatter(x=Th, y=q_cool_arr,             name="Q_cool",   line=dict(color="dodgerblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=Th, y=r["driving_heat_kw"],   name="Q_drive",  line=dict(color="orange")),     row=2, col=1)
    fig.add_trace(go.Scatter(x=Th, y=r["heat_rejection_kw"], name="Q_reject", line=dict(color="firebrick")),  row=2, col=1)

    Th_grid = np.linspace(55, 95, 50)
    Tc_grid = np.linspace(22, 40, 50)
    cop_map = np.zeros((50, 50))
    for i, th in enumerate(Th_grid):
        r = model.predict({"T_hot": th, "T_cool": Tc_grid, "T_chilled": 14.0})
        cop_map[i, :] = r["cop"]
    fig.add_trace(go.Heatmap(x=Tc_grid, y=Th_grid, z=cop_map, colorscale="Viridis",
                             colorbar=dict(title="COP")), row=2, col=2)

    fig.update_xaxes(title_text="T_hot (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_cool (degC)", row=1, col=2)
    fig.update_xaxes(title_text="T_hot (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_cool (degC)", row=2, col=2)
    fig.update_yaxes(title_text="COP_c", row=1, col=1)
    fig.update_yaxes(title_text="COP_c", row=1, col=2)
    fig.update_yaxes(title_text="kW_th", row=2, col=1)
    fig.update_yaxes(title_text="T_hot (degC)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
