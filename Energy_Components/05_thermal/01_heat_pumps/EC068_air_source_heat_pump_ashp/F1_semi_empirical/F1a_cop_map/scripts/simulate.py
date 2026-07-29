"""EC068 — ASHP — F1a — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_report():
    model = ComponentModel()
    info = model.get_info()
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["COP vs T_source", "COP vs T_sink", "Electrical Input vs T_source", "COP Map (Heatmap)"],
        vertical_spacing=0.12)

    Ts = np.linspace(-20, 35, 100)
    for Tk in [30, 35, 45, 55, 65]:
        r = model.predict({"T_source": Ts, "T_sink": Tk})
        fig.add_trace(go.Scatter(x=Ts, y=r["cop"], name=f"T_sink={Tk}C"), row=1, col=1)

    Tk = np.linspace(25, 65, 100)
    for Tsi in [-10, 0, 7, 15, 25]:
        r = model.predict({"T_source": Tsi, "T_sink": Tk})
        fig.add_trace(go.Scatter(x=Tk, y=r["cop"], name=f"T_src={Tsi}C"), row=1, col=2)

    for Tk in [35, 45, 55]:
        r = model.predict({"T_source": Ts, "T_sink": Tk})
        fig.add_trace(go.Scatter(x=Ts, y=r["electrical_input_kw"], name=f"W T_sink={Tk}C", showlegend=False), row=2, col=1)

    Ts_grid = np.linspace(-20, 35, 50)
    Tk_grid = np.linspace(25, 65, 50)
    cop_map = np.zeros((50, 50))
    for i, ts in enumerate(Ts_grid):
        r = model.predict({"T_source": ts, "T_sink": Tk_grid})
        cop_map[i, :] = r["cop"]
    fig.add_trace(go.Heatmap(x=Tk_grid, y=Ts_grid, z=cop_map, colorscale="Viridis", name="COP"), row=2, col=2)

    fig.update_xaxes(title_text="T_source (C)", row=1, col=1)
    fig.update_xaxes(title_text="T_sink (C)", row=1, col=2)
    fig.update_xaxes(title_text="T_source (C)", row=2, col=1)
    fig.update_xaxes(title_text="T_sink (C)", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="kW_e", row=2, col=1)
    fig.update_yaxes(title_text="T_source (C)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}", height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

if __name__ == "__main__":
    generate_report()
