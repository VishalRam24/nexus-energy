"""
EC049 -- Multi-Junction CPV -- F2a Physics-Lumped
Optional Plotly simulation report. Plotly import is guarded so its absence
does not crash the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # 1) Efficiency / Voc vs concentration sweep (steady, 298 K)
    dni_grid = np.linspace(50, 1050, 60)
    eff, voc, pmp = [], [], []
    for g in dni_grid:
        r = m.mpp(g, 298.15)
        eff.append(r["efficiency"])
        voc.append(r["v_oc"])
        pmp.append(r["p_mp"] * 1000.0)  # mW
    C_grid = m.concentration(dni_grid)

    # 2) Thermal transient at DNI=900
    tr = m.simulate(900.0, T0=298.15, dt=1.0, duration_s=200.0)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # noqa
        print(f"[simulate] Plotly unavailable ({e}); printing summary instead.")
        print(f"  Peak efficiency: {max(eff)*100:.2f}% @ C={C_grid[int(np.argmax(eff))]:.0f}")
        print(f"  Voc @ 1000x: {voc[-1]:.3f} V")
        print(f"  Thermal SS T: {tr['temperature'][-1]:.2f} K")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Efficiency vs Concentration", "Voc vs Concentration (log boost)",
        "MPP Power vs Concentration", "Junction Temperature Transient (DNI=900)"))
    fig.add_trace(go.Scatter(x=C_grid, y=np.array(eff) * 100, name="eta %"), 1, 1)
    fig.add_trace(go.Scatter(x=C_grid, y=voc, name="Voc"), 1, 2)
    fig.add_trace(go.Scatter(x=C_grid, y=pmp, name="P_mp mW"), 2, 1)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["temperature"], name="T [K]"), 2, 2)
    fig.update_layout(title="EC049 Multi-Junction CPV — F2a Physics-Lumped",
                      showlegend=False, height=750)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    run_report()
