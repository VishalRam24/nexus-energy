"""
EC134 -- OWC -- F2a Physics-Lumped: Plotly simulation report.

Generates an interactive HTML report showing:
  - water-column displacement & chamber pressure time-series
  - power flows (excitation / pneumatic / electrical)
  - Wells turbine efficiency curve
  - resonance sweep (response amplitude vs wave period)
  - mean electrical power across a (H_s, T_e) sea-state scatter

Plotly is optional; absence does not crash (prints text summary instead).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

import numpy as np

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def main():
    cm = ComponentModel()
    m = cm._model

    r = cm.predict({"H_s": 2.0, "T_e": 9.0, "dt": 0.04, "duration_s": 100.0})

    # resonance sweep
    omega_n = m.natural_frequency()
    T_res = 2.0 * np.pi / omega_n
    periods = np.linspace(0.5 * T_res, 1.8 * T_res, 15)
    amps = []
    for T in periods:
        rr = m.simulate(1.0, float(T), dt=0.04, duration_s=100.0)
        i0 = len(rr["t"]) // 2
        amps.append(float(np.std(rr["x"][i0:])))

    # Wells efficiency curve
    phi = np.linspace(0, 3 * m.phi_stall, 100)
    eta_w = m.wells_efficiency(phi)

    print(f"Mean P_elec: {r['mean_P_elec_kW']:.2f} kW")
    print(f"CWR: {r['capture_width_ratio']:.3f}  capture_eff: {r['capture_efficiency']:.3f}")
    print(f"Natural period T_n = {T_res:.2f} s ; wave T_e = {2*np.pi/r['omega']:.2f} s")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); text summary only.")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Water-column displacement x(t)",
            "Chamber gauge pressure p(t)",
            "Power flows",
            "Wells turbine efficiency",
            "Resonance sweep (amplitude vs T)",
            "Electrical power vs time",
        ),
    )
    t = r["t"]
    fig.add_trace(go.Scatter(x=t, y=r["x"], name="x [m]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["pressure"], name="p [Pa]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=r["P_exc"] / 1e3, name="P_exc [kW]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["P_avail"] / 1e3, name="P_avail [kW]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["P_elec"] / 1e3, name="P_elec [kW]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=phi, y=eta_w, name="eta_wells"), row=2, col=2)
    fig.add_trace(go.Scatter(x=periods, y=amps, name="std(x) [m]", mode="lines+markers"), row=3, col=1)
    fig.add_vline(x=T_res, line_dash="dash", row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["P_elec"] / 1e3, name="P_elec [kW]"), row=3, col=2)

    fig.update_layout(height=1000, width=1200,
                      title_text="EC134 OWC F2a — Physics-Lumped Dynamics Report")
    fig.write_html(_OUT)
    print(f"[simulate] Wrote {_OUT}")


if __name__ == "__main__":
    main()
