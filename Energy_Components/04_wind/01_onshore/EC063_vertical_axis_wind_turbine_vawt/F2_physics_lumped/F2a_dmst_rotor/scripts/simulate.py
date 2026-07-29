"""
EC063 -- VAWT F2a DMST + Rotor Dynamics -- simulation report.

Generates an interactive Plotly report:
  (1) DMST performance curve Cp(lambda)
  (2) Rotor spin-up / dynamic response to a wind step
  (3) Operating-point sweep: steady power vs generator load

Plotly is optional -- import is wrapped so absence does not crash.
Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

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
    m = cm._model
    cpmax, lam_opt = m.cp_max()

    # (1) Cp(lambda)
    lam = np.linspace(0.5, 10.0, 80)
    cp = np.array([m.cp(l) for l in lam])

    # (2) wind-step dynamic response
    def wind(t):
        return 8.0 if t < 60.0 else 13.0
    dyn = m.simulate(wind, T_load=200.0, omega0=8.0, dt=0.5, duration_s=140.0)

    # (3) power vs load sweep at U = 10 m/s
    loads = np.linspace(50.0, 600.0, 14)
    P_elec, tsr_ss = [], []
    for TL in loads:
        r = m.simulate(10.0, T_load=TL, omega0=8.0, dt=0.5, duration_s=300.0)
        P_elec.append(r["power_elec"][-1] / 1000.0)
        tsr_ss.append(r["tip_speed_ratio"][-1])

    print(f"Peak Cp = {cpmax:.3f} at TSR = {lam_opt:.2f}")
    print(f"Wind-step final: TSR={dyn['tip_speed_ratio'][-1]:.2f}, "
          f"P_elec={dyn['power_elec'][-1]/1000:.2f} kW")
    print(f"Max P_elec over load sweep = {max(P_elec):.2f} kW")

    if not _HAVE_PLOTLY:
        print("[plotly not installed -- skipping HTML report]")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "DMST performance curve Cp(λ)",
            "Wind-step dynamic response",
            "Steady electrical power vs generator load",
            "Operating TSR vs generator load",
        ),
    )
    fig.add_trace(go.Scatter(x=lam, y=cp, name="Cp"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dyn["t"], y=dyn["rpm"], name="rpm"), row=1, col=2)
    fig.add_trace(go.Scatter(x=dyn["t"], y=dyn["power_elec"] / 1000.0,
                             name="P_elec [kW]", yaxis="y2"), row=1, col=2)
    fig.add_trace(go.Scatter(x=loads, y=P_elec, name="P_elec [kW]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=loads, y=tsr_ss, name="TSR"), row=2, col=2)

    fig.update_xaxes(title_text="λ = ωR/U", row=1, col=1)
    fig.update_yaxes(title_text="Cp", row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=1, col=2)
    fig.update_xaxes(title_text="load torque [N.m]", row=2, col=1)
    fig.update_yaxes(title_text="P_elec [kW]", row=2, col=1)
    fig.update_xaxes(title_text="load torque [N.m]", row=2, col=2)
    fig.update_yaxes(title_text="TSR", row=2, col=2)
    fig.update_layout(title="EC063 VAWT -- F2a DMST + Rotor Dynamics",
                      height=760, showlegend=True)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
