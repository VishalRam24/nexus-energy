"""EC047 — CIGS Thin-Film PV — F1a — Simulation & HTML Report"""

import sys, numpy as np
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
                        "Efficiency vs Irradiance", "Voc vs Irradiance"],
        vertical_spacing=0.12)

    G = np.linspace(50, 1200, 80)
    for T in [10, 25, 40, 55]:
        r = model.predict({"irradiance": G, "cell_temperature": T})
        fig.add_trace(go.Scatter(x=G, y=r["p_mp"], name=f"T={T}C"), row=1, col=1)

    T = np.linspace(-10, 70, 80)
    for Gi in [200, 400, 600, 800, 1000]:
        r = model.predict({"irradiance": Gi, "cell_temperature": T})
        fig.add_trace(go.Scatter(x=T, y=r["p_mp"], name=f"G={Gi}W/m2"), row=1, col=2)

    G = np.linspace(100, 1200, 80)
    for T in [25, 45]:
        r = model.predict({"irradiance": G, "cell_temperature": T})
        fig.add_trace(go.Scatter(x=G, y=r["efficiency"]*100, name=f"eta T={T}C",
                                  showlegend=False), row=2, col=1)

    r = model.predict({"irradiance": G, "cell_temperature": 25.0})
    fig.add_trace(go.Scatter(x=G, y=r["v_oc"], name="Voc", showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Cell Temp (C)", row=1, col=2)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=2)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Voc (V)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
