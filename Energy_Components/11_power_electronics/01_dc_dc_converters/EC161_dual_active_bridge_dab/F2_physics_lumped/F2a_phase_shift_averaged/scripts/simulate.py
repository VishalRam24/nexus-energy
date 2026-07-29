"""
EC161 -- DAB F2a -- simulation scenarios + optional Plotly HTML report.
Plotly import is guarded so its absence does not crash execution.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model
    v1, v2 = m.V1_nom, m.V2_nom

    # 1) SPS power-transfer curve vs phi (the signature DAB characteristic)
    phis = np.linspace(-np.pi / 2, np.pi / 2, 201)
    P = m.power_transfer(v1, v2, phis)
    eta = m.efficiency(v1, v2, phis)

    # 2) Averaged output-voltage transient for a 5 kW step
    phi_5k = float(m.phase_for_power(v1, v2, 5000.0))
    tr = m.simulate(phi_5k, v1=v1, v2_0=150.0, r_load=4.0, dt=2e-5, duration_s=5e-3)

    return phis, P, eta, tr, phi_5k


def main():
    phis, P, eta, tr, phi_5k = run_scenarios()
    print(f"P_max = {P.max():.1f} W at phi=+pi/2; P_min = {P.min():.1f} W at phi=-pi/2")
    print(f"5 kW phase shift phi = {phi_5k:.4f} rad; V_out {tr['v_out'][0]:.1f} -> {tr['v_out'][-1]:.1f} V")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] skipping HTML report.")
        return

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "SPS power transfer P(phi)", "Efficiency vs phi", "Averaged V_out transient"))
    fig.add_trace(go.Scatter(x=phis, y=P, name="P(phi)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=phis, y=eta, name="eta"), row=1, col=2)
    fig.add_trace(go.Scatter(x=tr["t"] * 1e3, y=tr["v_out"], name="V_out"), row=1, col=3)
    fig.update_xaxes(title_text="phi [rad]", row=1, col=1)
    fig.update_xaxes(title_text="phi [rad]", row=1, col=2)
    fig.update_xaxes(title_text="t [ms]", row=1, col=3)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
