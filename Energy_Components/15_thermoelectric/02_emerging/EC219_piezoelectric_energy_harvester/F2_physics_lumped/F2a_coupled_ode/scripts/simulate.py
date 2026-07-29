"""
EC219 -- Piezoelectric Energy Harvester -- F2a Coupled ODE
Optional Plotly report: frequency sweep, load sweep, and a time-domain trace.
Plotly import is guarded so its absence does not crash the model.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    m = cm._model
    R_opt = m.optimal_load()

    freqs = np.linspace(70.0, 130.0, 31)
    P_freq = [m.simulate(9.81, f, R_opt)["P_avg"] * 1e6 for f in freqs]

    R_vals = np.logspace(3, 7, 25)
    P_load = [m.simulate(9.81, 100.0, R)["P_avg"] * 1e6 for R in R_vals]

    trace = m.simulate(9.81, 100.0, R_opt, n_periods=30)

    print(f"Peak power {max(P_freq):.3f} uW at resonance; R_opt={R_opt:.0f} ohm")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly not installed -> skip report
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Power vs Frequency (1 g, R_opt)",
                                        "Power vs Load (f_n)",
                                        "Voltage trace (f_n, R_opt)"))
    fig.add_trace(go.Scatter(x=freqs, y=P_freq, mode="lines+markers", name="P(f)"), 1, 1)
    fig.add_trace(go.Scatter(x=R_vals, y=P_load, mode="lines+markers", name="P(R)"), 1, 2)
    fig.add_trace(go.Scatter(x=trace["t"], y=trace["voltage"], mode="lines", name="V(t)"), 1, 3)
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_xaxes(title_text="R_load [ohm]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Time [s]", row=1, col=3)
    fig.update_yaxes(title_text="P_avg [uW]", row=1, col=1)
    fig.update_yaxes(title_text="P_avg [uW]", row=1, col=2)
    fig.update_yaxes(title_text="Voltage [V]", row=1, col=3)
    fig.update_layout(title="EC219 Piezoelectric Harvester -- F2a Coupled ODE",
                      showlegend=False, height=420, width=1300)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Report written to {out}")


if __name__ == "__main__":
    run()
