"""
EC073 -- Shell-and-Tube Heat Exchanger -- F2a Lumped-Capacitance Transient
Simulation scenarios + optional Plotly HTML report.

Plotly is optional: the import is wrapped so the script still runs (and prints
a text summary) if plotly is not installed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


def run():
    cm = ComponentModel()

    # Scenario A: cold start to steady state
    a = cm.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                    "m_dot_hot": 2.0, "m_dot_cold": 2.0,
                    "duration_s": 600.0, "dt": 2.0, "T_init": 20.0})

    # Scenario B: hot inlet step at t=300 s (60 -> 95 degC)
    def th_step(t):
        return 60.0 if t < 300.0 else 95.0
    b = cm.predict({"T_h_in": th_step, "T_c_in": 20.0,
                    "m_dot_hot": 2.0, "m_dot_cold": 2.0,
                    "duration_s": 700.0, "dt": 2.0, "T_init": 20.0})

    ss = a["steady_state_reference"]
    print("Scenario A (cold start):")
    print(f"  Final T_h_out={a['T_h_out'][-1]:.2f}, T_c_out={a['T_c_out'][-1]:.2f}, "
          f"Q={a['Q_kw'][-1]:.2f} kW")
    print(f"  eps-NTU ref:  T_h_out={ss['T_h_out']:.2f}, T_c_out={ss['T_c_out']:.2f}, "
          f"Q={ss['Q_kw']:.2f} kW")
    print("Scenario B (hot inlet step 60->95 at 300 s):")
    print(f"  Final Q={b['Q_kw'][-1]:.2f} kW")

    if not _HAVE_PLOTLY:
        print("\n[plotly not installed -- skipping HTML report]")
        return

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Cold start to steady state",
                                        "Hot-inlet step response"))
    fig.add_trace(go.Scatter(x=a["t"], y=a["T_h_out"], name="T_h_out (A)"), 1, 1)
    fig.add_trace(go.Scatter(x=a["t"], y=a["T_c_out"], name="T_c_out (A)"), 1, 1)
    fig.add_hline(y=ss["T_h_out"], line_dash="dot", row=1, col=1)
    fig.add_hline(y=ss["T_c_out"], line_dash="dot", row=1, col=1)
    fig.add_trace(go.Scatter(x=b["t"], y=b["Q_kw"], name="Q [kW] (B)"), 2, 1)
    fig.update_xaxes(title_text="time [s]")
    fig.update_layout(title="EC073 F2a -- Lumped Transient Shell-and-Tube HX",
                      height=720)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"\nWrote {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
