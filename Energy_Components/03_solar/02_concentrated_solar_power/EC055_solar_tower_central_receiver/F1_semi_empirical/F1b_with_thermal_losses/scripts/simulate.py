"""EC055 — Solar Tower — F1b With Thermal Losses — Simulation Scenarios + HTML Report"""

import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def run_simulations(model: ComponentModel):
    results = {}

    # 1. Receiver efficiency vs T_recv at different T_amb
    T_recvs = np.linspace(300, 800, 80)
    results["eff_vs_T_recv"] = {}
    for T_amb in [-10.0, 10.0, 30.0, 50.0]:
        r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                           "T_receiver_degC": T_recvs, "T_ambient_degC": T_amb})
        results["eff_vs_T_recv"][T_amb] = {
            "T_recv": T_recvs,
            "eta_recv": r["receiver_efficiency"],
            "Q_loss": r["thermal_loss_kw"],
        }

    # 2. Useful heat vs DNI at fixed T_recv and T_amb
    dnis = np.linspace(0, 1100, 80)
    zeniths = np.linspace(5, 70, 80)
    r = model.predict({"dni_w_m2": dnis, "solar_zenith_deg": 25.0,
                       "T_receiver_degC": 600.0, "T_ambient_degC": 25.0})
    results["heat_vs_dni"] = {"dni": dnis, "Q_useful": r["useful_heat_kw"],
                               "Q_field": r["Q_field_kw"]}

    # 3. Thermal loss breakdown: radiative vs convective
    T_recvs2 = np.linspace(200, 800, 60)
    r = model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                       "T_receiver_degC": T_recvs2, "T_ambient_degC": 25.0,
                       "wind_speed_m_s": 5.0})
    results["loss_breakdown"] = {
        "T_recv": T_recvs2,
        "Q_rad": r["Q_radiative_kw"],
        "Q_conv": r["Q_convective_kw"],
    }

    # 4. Wind effect on useful heat
    winds = np.linspace(0, 15, 50)
    qs = [float(model.predict({"dni_w_m2": 900.0, "solar_zenith_deg": 20.0,
                                "T_receiver_degC": 600.0, "T_ambient_degC": 25.0,
                                "wind_speed_m_s": float(w)})["useful_heat_kw"]) for w in winds]
    results["wind_effect"] = {"wind": winds, "Q_useful": np.array(qs)}

    return results


def generate_report(model: ComponentModel, output_path: Path):
    data = run_simulations(model)
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Receiver Efficiency vs T_recv (by T_amb)",
                                        "Useful Heat vs DNI",
                                        "Thermal Loss Breakdown vs T_recv",
                                        "Wind Speed Effect on Useful Heat"])

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for i, (T_amb, d) in enumerate(data["eff_vs_T_recv"].items()):
        fig.add_trace(go.Scatter(x=d["T_recv"], y=d["eta_recv"] * 100,
                                 name=f"T_amb={T_amb}C", line=dict(color=colors[i])), row=1, col=1)

    d = data["heat_vs_dni"]
    fig.add_trace(go.Scatter(x=d["dni"], y=d["Q_useful"], name="Q_useful", line=dict(color="blue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=d["dni"], y=d["Q_field"], name="Q_field", line=dict(color="gray", dash="dot")), row=1, col=2)

    d = data["loss_breakdown"]
    fig.add_trace(go.Scatter(x=d["T_recv"], y=d["Q_rad"], name="Q_radiative", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["T_recv"], y=d["Q_conv"], name="Q_convective", line=dict(color="cyan")), row=2, col=1)

    d = data["wind_effect"]
    fig.add_trace(go.Scatter(x=d["wind"], y=d["Q_useful"], name="Q_useful (kW)",
                             line=dict(color="purple")), row=2, col=2)

    fig.update_xaxes(title_text="T_receiver (degC)", row=1, col=1)
    fig.update_xaxes(title_text="DNI (W/m2)", row=1, col=2)
    fig.update_xaxes(title_text="T_receiver (degC)", row=2, col=1)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Receiver Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Heat (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Loss (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Useful Heat (kW)", row=2, col=2)

    fig.update_layout(title="EC055 Solar Tower — F1b Thermal Losses Simulation", height=700)
    fig.write_html(str(output_path))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    model = ComponentModel()
    generate_report(model, Path(__file__).parent.parent / "simulation_report.html")
