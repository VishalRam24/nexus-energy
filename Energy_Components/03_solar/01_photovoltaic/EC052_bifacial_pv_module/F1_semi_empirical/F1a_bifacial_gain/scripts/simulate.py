"""EC052 — Bifacial PV Module — F1a — Simulation & HTML Report"""

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
        subplot_titles=["Power vs Albedo (front=1000)",
                        "Bifacial Gain vs Rear Irradiance",
                        "Power vs Front Irradiance (several albedos)",
                        "Power vs Cell Temperature"],
        vertical_spacing=0.13)

    # Panel 1: Power vs albedo
    alb = np.linspace(0, 0.9, 80)
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0, "albedo": alb})
    fig.add_trace(go.Scatter(x=alb, y=r["p_mp"], name="P_mp"), row=1, col=1)

    # Panel 2: Bifacial gain vs rear irradiance
    G_r = np.linspace(0, 400, 80)
    r = model.predict({"irradiance_front": 1000.0, "cell_temperature": 25.0,
                       "irradiance_rear": G_r})
    fig.add_trace(go.Scatter(x=G_r, y=np.asarray(r["bifacial_gain"])*100,
                             name="Gain (%)"), row=1, col=2)

    # Panel 3: P vs G_front for several albedos
    G_f = np.linspace(50, 1200, 80)
    for a in [0.0, 0.2, 0.4, 0.6]:
        r = model.predict({"irradiance_front": G_f, "cell_temperature": 25.0,
                           "albedo": a})
        fig.add_trace(go.Scatter(x=G_f, y=r["p_mp"], name=f"albedo={a}"), row=2, col=1)

    # Panel 4: P vs T
    T = np.linspace(-10, 70, 80)
    for a in [0.0, 0.3]:
        r = model.predict({"irradiance_front": 1000.0, "cell_temperature": T, "albedo": a})
        fig.add_trace(go.Scatter(x=T, y=r["p_mp"], name=f"P(T) alb={a}"), row=2, col=2)

    fig.update_xaxes(title_text="Ground Albedo (-)", row=1, col=1)
    fig.update_xaxes(title_text="Rear Irradiance (W/m2)", row=1, col=2)
    fig.update_xaxes(title_text="Front Irradiance (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="Cell Temperature (C)", row=2, col=2)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Bifacial Gain (%)", row=1, col=2)
    fig.update_yaxes(title_text="Power (W)", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=850, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
