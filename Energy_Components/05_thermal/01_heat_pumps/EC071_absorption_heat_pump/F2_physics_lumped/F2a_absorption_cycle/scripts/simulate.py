"""
EC071 -- Absorption Heat Pump (LiBr-H2O) -- F2a Physics-Lumped
Optional Plotly report. Plotly import is guarded; absence does not crash.
Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    # transient warm-up
    r = cm.predict({"T_gen0_C": 40.0, "duration_s": 2400.0, "dt": 10.0})

    # COP vs driving temperature sweep (steady cycle)
    m = cm._model
    T_drive_sweep = list(range(75, 111, 2))
    cop_h, cop_c, q_heat = [], [], []
    for Td in T_drive_sweep:
        d = m.rate_duties(T_gen_c=float(Td))
        cop_h.append(d["cop_heating"])
        cop_c.append(d["cop_cooling"])
        q_heat.append(d["Q_heat_kW"])

    print("Transient: T_gen %.1f -> %.1f C (drive %.1f C)"
          % (r["T_gen_C"][0], r["T_gen_C"][-1], m.T_drive))
    print("Design COP_heating=%.3f, COP_cooling=%.3f, f=%.2f, Q_heat=%.1f kW"
          % (r["cop_heating_design"], r["cop_cooling_design"],
             r["f_circulation"], r["Q_heat_kW_design"]))

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Generator loop warm-up (transient ODE)",
                        "Cycle COP vs driving generator temperature"))
    fig.add_trace(go.Scatter(x=r["t"] / 60.0, y=r["T_gen_C"],
                             name="T_gen [C]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(T_drive_sweep), y=cop_h,
                             name="COP heating"), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(T_drive_sweep), y=cop_c,
                             name="COP cooling"), row=1, col=2)
    fig.update_xaxes(title_text="time [min]", row=1, col=1)
    fig.update_yaxes(title_text="T_gen [C]", row=1, col=1)
    fig.update_xaxes(title_text="generator temp [C]", row=1, col=2)
    fig.update_yaxes(title_text="COP [-]", row=1, col=2)
    fig.update_layout(title_text="EC071 Absorption Heat Pump -- F2a Physics-Lumped")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    run()
