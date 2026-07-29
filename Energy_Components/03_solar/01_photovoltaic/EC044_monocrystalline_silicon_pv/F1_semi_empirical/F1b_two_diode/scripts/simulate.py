"""EC044 — Mono-Si PV — F1b Two-Diode — Simulation & HTML Report"""

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
        subplot_titles=["Power vs Irradiance", "Power vs Cell Temperature",
                        "Fill Factor vs Irradiance", "I-V Curve at STC"],
        vertical_spacing=0.12)

    # Power vs Irradiance at different temperatures
    G = np.linspace(50, 1200, 24)
    for T in [10, 25, 40, 55]:
        r = model.predict({"irradiance_w_m2": G, "temperature_cell_degC": T})
        fig.add_trace(go.Scatter(x=G, y=r["p_mp"], name=f"T={T}C"), row=1, col=1)

    # Power vs Temperature at different irradiances
    T = np.linspace(-10, 70, 17)
    for Gi in [200, 600, 1000]:
        r = model.predict({"irradiance_w_m2": Gi, "temperature_cell_degC": T})
        fig.add_trace(go.Scatter(x=T, y=r["p_mp"], name=f"G={Gi}W/m2"), row=1, col=2)

    # Fill factor vs Irradiance (key F1b advantage)
    G = np.linspace(50, 1200, 24)
    for T in [25, 45]:
        r = model.predict({"irradiance_w_m2": G, "temperature_cell_degC": T})
        fig.add_trace(go.Scatter(x=G, y=r["fill_factor"], name=f"FF T={T}C",
                                  showlegend=False), row=2, col=1)

    # I-V curve at STC
    V, I = model._model.iv_curve(1000.0, 25.0, n_points=100)
    fig.add_trace(go.Scatter(x=V, y=I, name="I-V STC", showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Cell Temp (C)", row=1, col=2)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="Voltage (V)", row=2, col=2)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Fill Factor", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Two-Diode",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
