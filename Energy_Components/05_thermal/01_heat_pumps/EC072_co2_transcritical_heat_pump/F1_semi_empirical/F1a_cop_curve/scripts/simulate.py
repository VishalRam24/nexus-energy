"""EC072 — CO2 Transcritical Heat Pump — F1a — Simulation & HTML Report"""
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
            "COP vs T_evap (T_water_out varied)",
            "COP vs T_water_out (T_water_in varied)",
            "Electrical input vs T_evap",
            "COP Map — T_evap vs T_water_out",
        ],
        vertical_spacing=0.13,
    )

    Te = np.linspace(-20, 20, 100)
    for two in [50, 60, 70, 80, 90]:
        r = model.predict({"T_evap": Te, "T_water_in": 15.0, "T_water_out": two})
        fig.add_trace(go.Scatter(x=Te, y=r["cop"], name=f"Tw_out={two}C"), row=1, col=1)

    Two = np.linspace(40, 90, 100)
    for twi in [10, 15, 25, 35]:
        r = model.predict({"T_evap": 0.0, "T_water_in": twi, "T_water_out": Two})
        fig.add_trace(go.Scatter(x=Two, y=r["cop"], name=f"Tw_in={twi}C"), row=1, col=2)

    for two in [55, 65, 75]:
        r = model.predict({"T_evap": Te, "T_water_in": 15.0, "T_water_out": two})
        fig.add_trace(go.Scatter(x=Te, y=r["electrical_input_kw"],
                                 name=f"W Tw_out={two}C", showlegend=False), row=2, col=1)

    Te_grid  = np.linspace(-20, 20, 50)
    Two_grid = np.linspace(40, 90, 50)
    cop_map  = np.zeros((50, 50))
    for i, te in enumerate(Te_grid):
        r = model.predict({"T_evap": te, "T_water_in": 15.0, "T_water_out": Two_grid})
        cop_map[i, :] = r["cop"]
    fig.add_trace(go.Heatmap(x=Two_grid, y=Te_grid, z=cop_map, colorscale="Viridis",
                             colorbar=dict(title="COP")), row=2, col=2)

    fig.update_xaxes(title_text="T_evap (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_water_out (degC)", row=1, col=2)
    fig.update_xaxes(title_text="T_evap (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_water_out (degC)", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="kW_e", row=2, col=1)
    fig.update_yaxes(title_text="T_evap (degC)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
