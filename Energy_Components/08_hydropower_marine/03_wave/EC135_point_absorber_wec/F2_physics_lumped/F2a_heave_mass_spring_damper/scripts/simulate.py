"""
EC135 -- Point Absorber WEC -- F2a Heaving-Buoy Linear Hydrodynamic Model
Simulation scenarios + optional interactive Plotly report.

Generates:
  1. Time-domain heave / velocity / power under a regular wave at resonance.
  2. Resonance sweep: mean absorbed power vs wave period.
  3. PTO-damping sweep: mean absorbed power vs B_pto (shows the optimum |Z_i|).

Plotly is optional; if absent the script still prints a text summary.
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
    Tn = m.natural_period()
    H = 1.5

    # 1. Time-domain at resonance with optimal PTO
    B_opt = m.optimal_B_pto(Tn)
    ts = m.simulate(H=H, T=Tn, B_pto=B_opt, duration_s=8 * Tn, dt=0.02)

    # 2. Resonance sweep (analytic mean power)
    periods = np.linspace(0.5 * Tn, 1.8 * Tn, 60)
    P_vs_T = np.array([m.mean_power_analytic(H, T, B_pto=m.optimal_B_pto(T))
                       for T in periods]) / 1e3

    # 3. PTO-damping sweep off-resonance
    T_off = 1.2 * Tn
    B_grid = np.linspace(1e4, 6e5, 80)
    P_vs_B = np.array([m.mean_power_analytic(H, T_off, B_pto=B)
                       for B in B_grid]) / 1e3
    B_opt_off = m.optimal_B_pto(T_off)

    print(f"EC135 F2a point absorber -- T_n = {Tn:.2f} s, H = {H} m")
    print(f"  Resonant mean absorbed power : {ts['P_pto_mean']/1e3:.2f} kW")
    print(f"  Resonant capture width       : {ts['capture_width']:.2f} m "
          f"(max {ts['capture_width_max']:.2f} m)")
    print(f"  Optimal B_pto (off-res {T_off:.1f}s): {B_opt_off:.3e} N.s/m")

    if not _HAVE_PLOTLY:
        print("\n[plotly not installed -- skipping HTML report]")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Heave displacement & velocity (resonance)",
            "Instantaneous PTO power",
            "Mean absorbed power vs wave period",
            "Mean absorbed power vs PTO damping (T=1.2 T_n)",
        ),
    )
    fig.add_trace(go.Scatter(x=ts["t"], y=ts["x"], name="x [m]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts["t"], y=ts["x_dot"], name="x_dot [m/s]"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ts["t"], y=ts["P_pto_inst"] / 1e3,
                             name="P_pto [kW]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=periods, y=P_vs_T, name="P(T)"), row=2, col=1)
    fig.add_vline(x=Tn, line_dash="dash", row=2, col=1)
    fig.add_trace(go.Scatter(x=B_grid / 1e3, y=P_vs_B, name="P(B_pto)"),
                  row=2, col=2)
    fig.add_vline(x=B_opt_off / 1e3, line_dash="dash", row=2, col=2)

    fig.update_xaxes(title_text="t [s]", row=1, col=1)
    fig.update_xaxes(title_text="t [s]", row=1, col=2)
    fig.update_xaxes(title_text="wave period T [s]", row=2, col=1)
    fig.update_xaxes(title_text="B_pto [kN.s/m]", row=2, col=2)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=2)
    fig.update_layout(title="EC135 Point Absorber WEC -- F2a Linear Hydrodynamic",
                      height=800, showlegend=True)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"\nReport written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
