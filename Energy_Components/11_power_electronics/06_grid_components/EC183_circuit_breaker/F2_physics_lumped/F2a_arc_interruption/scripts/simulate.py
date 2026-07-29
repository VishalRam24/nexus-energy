"""
EC183 -- Circuit Breaker -- F2a Arc Interruption
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    m = ComponentModel()
    cases = {}
    for label, If in [("5 kA", 5.0), ("20 kA", 20.0), ("25 kA (rated)", 25.0),
                      ("40 kA (over-capacity)", 40.0)]:
        cases[label] = m.predict({"I_fault_kA": If, "duration_ms": 15.0, "dt_us": 0.1})
    return cases


def main():
    cases = run_scenarios()
    print("EC183 F2a Arc Interruption -- scenario summary")
    for label, r in cases.items():
        print(f"  {label:24s} | zeros={r['n_current_zeros']:4d} | "
              f"E_arc={r['arc_energy_total_J']/1e3:7.1f} kJ | "
              f"TRV={r['trv_peak_V']/1e3:5.1f} kV | "
              f"within_cap={r['within_capacity']} | success={r['interruption_success']}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Arc current i(t)", "Arc / gap voltage u(t)",
                        "Arc conductance g(t) [log]"))
    for label, r in cases.items():
        t_ms = r["t"] * 1e3
        fig.add_trace(go.Scatter(x=t_ms, y=r["current"]/1e3, name=label),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t_ms, y=r["arc_voltage"]/1e3, name=label,
                                 showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_ms, y=np.log10(np.maximum(r["conductance"], 1e-12)),
                                 name=label, showlegend=False), row=3, col=1)
    fig.update_xaxes(title_text="time [ms]", row=3, col=1)
    fig.update_yaxes(title_text="i [kA]", row=1, col=1)
    fig.update_yaxes(title_text="u [kV]", row=2, col=1)
    fig.update_yaxes(title_text="log10 g [S]", row=3, col=1)
    fig.update_layout(title="EC183 F2a -- Cassie-Mayr Arc Interruption",
                      height=900)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
