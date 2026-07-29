"""
EC094 -- Evaporative Cooler -- F2a Psychrometric
Optional Plotly simulation report. Plotly import is guarded so absence does not crash.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()

    # Scenario: hot dry inlet step + transient cooldown
    def step_T(t):
        return 32.0 if t < 200 else 42.0

    r = cm.predict({"T_db_C": step_T, "RH": 0.20, "dt": 2.0, "duration_s": 400.0})

    # Psychrometric sweep over RH at fixed dry-bulb
    rh_grid = [0.05 * i for i in range(2, 18)]
    m = cm._model
    T_out_sweep = [m.steady_state(38.0, rh, 1.0)["T_out"] for rh in rh_grid]
    T_wb_sweep = [m.wet_bulb(38.0, rh) for rh in rh_grid]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML, printing summary.")
        ss = r["steady_state"]
        print(f"  T_wb={ss['T_wb']:.2f}C T_out_final={r['T_out'][-1]:.2f}C "
              f"water={ss['m_dot_water_kg_s']*1000:.3f} g/s")
        return None

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Transient response (inlet step at t=200s)",
                                        "Outlet T vs inlet RH (T_db=38C)"))
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_db"], name="T_db (inlet)"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_pad"], name="T_pad"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_out"], name="T_out"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_wb"], name="T_wb", line=dict(dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=rh_grid, y=T_out_sweep, name="T_out"), 1, 2)
    fig.add_trace(go.Scatter(x=rh_grid, y=T_wb_sweep, name="T_wb", line=dict(dash="dot")), 1, 2)
    fig.update_layout(title="EC094 Evaporative Cooler F2a -- Psychrometric Lumped",
                      template="plotly_white")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    run_report()
