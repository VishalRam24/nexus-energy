"""
EC139 -- PRO F2a Osmotic Flux -- simulation + optional Plotly report.
Generates the characteristic power-density vs DeltaP curve (peak at ~Dpi/2)
and the module dilution time series. Plotly import is optional.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_BAR = 1.0e5


def run():
    cm = ComponentModel()
    m = cm._model

    dpi = float(m.osmotic_pressure(m.C_draw0) - m.osmotic_pressure(m.C_feed0))
    dP_grid = np.linspace(0.0, dpi, 120)
    W = np.array([m.power_density(dP) for dP in dP_grid])
    Jw = np.array([m.water_flux(dP) for dP in dP_grid]) * 3.6e6  # LMH
    dP_opt, W_max = m.optimal_delta_P()

    print(f"Delta_pi = {dpi/_BAR:.2f} bar")
    print(f"Optimal DeltaP = {dP_opt/_BAR:.2f} bar "
          f"(= {dP_opt/dpi:.3f} * Delta_pi)")
    print(f"Peak power density = {W_max:.2f} W/m2")

    r = cm.predict({"duration_s": 600.0, "dt": 5.0})
    print(f"Module net power = {r['P_net_final_W']:.1f} W over {m.A_mem:.0f} m2; "
          f"draw diluted {m.C_draw0:.1f} -> {r['C_draw_gL'][-1]:.2f} g/L")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Power density vs DeltaP",
                                            "Draw dilution (module ODE)"))
        fig.add_trace(go.Scatter(x=dP_grid/_BAR, y=W, name="W [W/m2]"),
                      row=1, col=1)
        fig.add_vline(x=dP_opt/_BAR, line_dash="dash", row=1, col=1)
        fig.add_trace(go.Scatter(x=r["t"], y=r["C_draw_gL"],
                                 name="C_draw [g/L]"), row=1, col=2)
        out = os.path.join(os.path.dirname(__file__), "..",
                           "simulation_report.html")
        fig.write_html(out)
        print(f"Report written: {out}")
    except Exception as e:
        print(f"(Plotly report skipped: {e})")


if __name__ == "__main__":
    run()
