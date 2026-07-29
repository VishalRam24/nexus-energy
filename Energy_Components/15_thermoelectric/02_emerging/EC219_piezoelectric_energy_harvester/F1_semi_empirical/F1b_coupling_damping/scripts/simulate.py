"""EC219 — Piezoelectric Harvester — F1b Coupling+Damping — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    f_n = model._model.f_n
    R_opt = 1.0 / (2.0 * np.pi * f_n * model._model.C_p)

    freqs = np.linspace(10, 500, 200)
    R_loads = np.logspace(2, 7, 60)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power vs Frequency (sweep) at 1g",
            "Power vs Load Resistance at Resonance",
            "Voltage vs Frequency at 1g",
            "Power vs Acceleration at Resonance",
        ],
        vertical_spacing=0.15,
    )

    # Plot 1: Frequency sweep at different accelerations
    for a in [1.0, 2.0, 5.0]:
        powers = []
        for f in freqs:
            r = model.predict({"acceleration_ms2": a * 9.81, "frequency_hz": float(f), "R_load_ohm": R_opt})
            powers.append(float(np.atleast_1d(r["power_uw"])[0]))
        fig.add_trace(go.Scatter(x=freqs, y=powers, name=f"a={a}g", line=dict(width=2)), row=1, col=1)

    # Plot 2: Power vs R_load at resonance
    P_vs_R = []
    for R in R_loads:
        r = model.predict({"acceleration_ms2": 9.81, "frequency_hz": f_n, "R_load_ohm": float(R)})
        P_vs_R.append(float(np.atleast_1d(r["power_uw"])[0]))
    fig.add_trace(go.Scatter(x=R_loads, y=P_vs_R, name="P vs R_load", line=dict(width=2, color="navy")), row=1, col=2)
    fig.add_vline(x=R_opt, line_dash="dash", line_color="red", row=1, col=2)

    # Plot 3: Voltage vs frequency
    for a in [1.0, 2.0]:
        volts = []
        for f in freqs:
            r = model.predict({"acceleration_ms2": a * 9.81, "frequency_hz": float(f), "R_load_ohm": R_opt})
            volts.append(float(np.atleast_1d(r["voltage_v"])[0]))
        fig.add_trace(go.Scatter(x=freqs, y=volts, name=f"V a={a}g", line=dict(width=2)), row=2, col=1)

    # Plot 4: Power vs acceleration (quadratic)
    accels = np.linspace(0.5, 20, 50)
    P_vs_a = []
    for a in accels:
        r = model.predict({"acceleration_ms2": float(a), "frequency_hz": f_n, "R_load_ohm": R_opt})
        P_vs_a.append(float(np.atleast_1d(r["power_uw"])[0]))
    fig.add_trace(go.Scatter(x=accels, y=P_vs_a, name="P vs a", line=dict(width=2, color="navy")), row=2, col=2)

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="R_load (ohm)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    fig.update_yaxes(title_text="Power (uW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (uW)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Power (uW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Coupled electromechanical model | k31={model._model.k31} | f_n={f_n:.0f} Hz | R_opt={R_opt:.0f} Ω</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
