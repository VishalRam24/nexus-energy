"""EC220 — TENG — F1b Surface Charge Dynamics — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    freqs = np.logspace(-1, 2, 60)
    R_loads = np.logspace(4, 10, 60)
    t_array = np.linspace(0, 5 * model._model.tau_decay, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power vs Frequency (R=10 MΩ, t=0)",
            "Power vs Load Resistance (f=3 Hz, t=0)",
            "Power Decay Over Time (f=3 Hz, R=10 MΩ)",
            "V_oc vs Frequency (t=0 and t=tau)",
        ],
        vertical_spacing=0.15,
    )

    # Plot 1: Power vs frequency
    for t_s in [0, 1000, 5000]:
        powers = []
        for f in freqs:
            r = model.predict({"frequency_hz": float(f), "R_load_ohm": 1e7, "t_s": float(t_s)})
            powers.append(float(np.atleast_1d(r["power_net_w"])[0]) * 1000)  # mW
        fig.add_trace(go.Scatter(x=freqs, y=powers, name=f"t={t_s}s", line=dict(width=2)), row=1, col=1)

    # Plot 2: Power vs R_load
    P_vs_R = []
    for R in R_loads:
        r = model.predict({"frequency_hz": 3.0, "R_load_ohm": float(R), "t_s": 0.0})
        P_vs_R.append(float(np.atleast_1d(r["power_net_w"])[0]) * 1000)
    fig.add_trace(go.Scatter(x=R_loads, y=P_vs_R, name="P vs R (f=3Hz)", line=dict(width=2, color="navy")), row=1, col=2)

    # Plot 3: Power decay over time
    P_time = []
    for t in t_array:
        r = model.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "t_s": float(t)})
        P_time.append(float(np.atleast_1d(r["power_net_w"])[0]) * 1000)
    fig.add_trace(go.Scatter(x=t_array, y=P_time, name="Power decay", line=dict(width=2, color="red")), row=2, col=1)

    # Plot 4: V_oc vs frequency
    for t_s in [0, model._model.tau_decay]:
        vocs = []
        for f in freqs:
            r = model.predict({"frequency_hz": float(f), "R_load_ohm": 1e12, "t_s": float(t_s)})
            vocs.append(float(np.atleast_1d(r["V_oc_peak_V"])[0]))
        fig.add_trace(go.Scatter(x=freqs, y=vocs, name=f"V_oc t={t_s:.0f}s", line=dict(width=2)), row=2, col=2)

    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="R_load (Ω)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Power (mW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (mW)", row=1, col=2)
    fig.update_yaxes(title_text="Power (mW)", row=2, col=1)
    fig.update_yaxes(title_text="V_oc (V)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Dual-dielectric PTFE+Nylon | Charge decay tau={model._model.tau_decay:.0f}s | tan_delta={model._model.tan_delta}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
