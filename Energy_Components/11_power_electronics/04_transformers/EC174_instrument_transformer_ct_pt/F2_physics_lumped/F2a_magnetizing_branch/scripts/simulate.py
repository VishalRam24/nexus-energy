"""
EC174 -- Instrument Transformer (CT / PT) -- F2a Magnetizing-Branch
Optional Plotly report: error-vs-burden curves and CT saturation waveforms.
Plotly import is wrapped so its absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def run():
    cm = ComponentModel()
    m = cm._model

    # 1) Ratio & phase error vs burden
    bfs = np.linspace(0.1, 4.0, 30)
    ratio = [float(m.ct_errors(m.I_rated, bf)["ratio_error_pct"]) for bf in bfs]
    phase = [float(m.ct_errors(m.I_rated, bf)["phase_error_min"]) for bf in bfs]

    # 2) Saturation waveforms at increasing primary current
    waves = {}
    for mult in [5, 10, 20]:
        s = m.simulate_saturation(m.I_rated * np.sqrt(2.0) * mult, 1.0, n_cycles=2)
        waves[mult] = s

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); computed results only.")
        print(f"  ratio_err @rated burden = {ratio[np.argmin(np.abs(bfs-1.0))]:.4f} %")
        print(f"  distortion @20x = {waves[20]['distortion']:.3f}, saturated={waves[20]['saturated']}")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Ratio error vs burden", "Phase error vs burden",
        "Secondary current (CT saturation)", "Core flux linkage"))

    fig.add_trace(go.Scatter(x=bfs, y=ratio, name="ratio err [%]"), row=1, col=1)
    fig.add_hline(y=-m.accuracy_class, line_dash="dash", row=1, col=1)
    fig.add_trace(go.Scatter(x=bfs, y=phase, name="phase err [min]"), row=1, col=2)
    fig.add_hline(y=m.phase_class_min, line_dash="dash", row=1, col=2)

    for mult, s in waves.items():
        fig.add_trace(go.Scatter(x=s["t"] * 1e3, y=s["i_sec"],
                                 name=f"i_sec {mult}x"), row=2, col=1)
        fig.add_trace(go.Scatter(x=s["t"] * 1e3, y=s["flux"],
                                 name=f"flux {mult}x"), row=2, col=2)

    fig.update_layout(title="EC174 Instrument Transformer F2a -- Magnetizing-Branch Model",
                      height=800, width=1100)
    fig.write_html(_OUT)
    print(f"[simulate] wrote {_OUT}")


if __name__ == "__main__":
    run()
