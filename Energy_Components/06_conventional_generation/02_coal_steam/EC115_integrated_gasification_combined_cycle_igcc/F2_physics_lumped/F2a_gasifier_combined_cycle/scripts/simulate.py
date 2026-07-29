"""
EC115 -- IGCC -- F2a Physics-Lumped
Optional Plotly simulation report. Plotly import is wrapped so absence
does not crash; run with `python3 scripts/simulate.py` if plotly is present.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model
    design_coal = m.Q_coal_design / m.LHV_coal

    # Scenario A: cold start at design fuel (metal heats up to firing temp)
    cold = m.simulate(design_coal, T_metal_0=m.T_comp, dt=2.0, duration_s=3000.0)

    # Scenario B: fuel ramp (coal feed step 60% -> 100% at t=600 s)
    def coal_ramp(t):
        return 0.6 * design_coal if t < 600.0 else design_coal
    ramp = m.simulate(coal_ramp, T_metal_0=m.T_comp, dt=2.0, duration_s=2400.0)

    # Steady efficiency vs coal feed sweep
    coals = np.linspace(10.0, design_coal, 25)
    p_net = np.array([float(m.net_power_mw(c)) for c in coals])
    return cm, m, cold, ramp, coals, p_net


def main():
    cm, m, cold, ramp, coals, p_net = run_scenarios()
    print(cm.get_info()["design"])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cold start: combustor metal vs gas temperature",
            "Fuel ramp (60%->100%): metal temperature response",
            "Net power vs coal feed",
            "Efficiency stack (Carnot bound)",
        ),
    )
    fig.add_trace(go.Scatter(x=cold["t"], y=cold["T_gas"], name="T_gas"), 1, 1)
    fig.add_trace(go.Scatter(x=cold["t"], y=cold["T_metal"], name="T_metal"), 1, 1)
    fig.add_trace(go.Scatter(x=ramp["t"], y=ramp["T_metal"], name="T_metal ramp"), 1, 2)
    fig.add_trace(go.Scatter(x=ramp["t"], y=ramp["T_gas"], name="T_gas ramp"), 1, 2)
    fig.add_trace(go.Scatter(x=coals, y=p_net, name="net MW"), 2, 1)
    fig.add_trace(go.Bar(
        x=["Brayton", "Rankine", "Combined", "Net", "Carnot"],
        y=[m.eta_B, m.eta_R, m.combined_cycle_efficiency(),
           m.net_efficiency(), m.carnot_efficiency()],
        name="efficiency",
    ), 2, 2)
    fig.update_layout(title="EC115 IGCC F2a -- Physics-Lumped Simulation Report",
                      height=800)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Report written: {out}")


if __name__ == "__main__":
    main()
