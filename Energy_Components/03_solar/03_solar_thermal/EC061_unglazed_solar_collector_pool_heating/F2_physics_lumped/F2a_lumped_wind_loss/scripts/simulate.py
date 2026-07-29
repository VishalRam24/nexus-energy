"""
EC061 — Unglazed Solar Collector (Pool Heating) — F2a Physics-Lumped
Optional Plotly report. Plotly import is wrapped so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def _diurnal_G(t, peak=950.0, day_s=12 * 3600.0):
    frac = t / day_s
    return max(0.0, peak * np.sin(np.pi * frac))


def run_report(out_html=None):
    cm = ComponentModel()

    # 1) diurnal dynamic run
    r = cm.predict({"G": _diurnal_G, "Ta": 22.0, "u_wind": 1.5, "Tf_in": 25.0,
                    "Tp0": 22.0, "dt": 300.0, "duration_s": 12 * 3600.0})
    hrs = r["t"] / 3600.0

    # 2) efficiency curve vs (Tm-Ta)/G for several wind speeds
    G = 800.0
    xstar = np.linspace(0.0, 0.08, 25)   # (Tm-Ta)/G  [K m2/W]
    curves = {}
    m = cm._model
    for u in [0.0, 3.0, 6.0]:
        eta = []
        for x in xstar:
            dT = x * G
            Tm = 20.0 + dT
            eta0_eff = m.optical_efficiency(u)
            U_L = m.loss_coefficient(u)
            eta.append(max(0.0, eta0_eff - U_L * dT / G))
        curves[u] = (xstar, np.array(eta))

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); printing summary only.")
        print(f"  Diurnal peak Q_use = {r['Q_use_W'].max():.1f} W")
        print(f"  Daily energy = {np.trapz(r['Q_use_W'], r['t'])/3.6e6:.2f} kWh")
        for u, (x, eta) in curves.items():
            print(f"  u={u} m/s: eta(x=0)={eta[0]:.3f}, eta(x=0.05)={eta[min(15,len(eta)-1)]:.3f}")
        return None

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Plate temperature", "Useful heat",
                                        "Efficiency curve vs (Tm-Ta)/G",
                                        "Loss coefficient vs wind"))
    fig.add_trace(go.Scatter(x=hrs, y=r["T_plate"], name="T_plate"), 1, 1)
    fig.add_trace(go.Scatter(x=hrs, y=r["Ta"], name="Ta", line=dict(dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=hrs, y=r["Q_use_W"], name="Q_use [W]"), 1, 2)
    for u, (x, eta) in curves.items():
        fig.add_trace(go.Scatter(x=x, y=eta, name=f"u={u} m/s"), 2, 1)
    uu = np.linspace(0, 10, 30)
    fig.add_trace(go.Scatter(x=uu, y=[m.loss_coefficient(v) for v in uu],
                             name="U_L(u)"), 2, 2)
    fig.update_layout(title="EC061 Unglazed Collector F2a — Physics-Lumped", height=760)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    run_report()
