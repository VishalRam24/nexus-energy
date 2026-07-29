"""
EC159 -- Buck-Boost Converter (Inverting) -- F2a State-Space Averaged Model
Optional Plotly simulation report. Plotly import is guarded so its absence
does not crash anything.
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


def run_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # 1) Transient start-up at d=0.5
    r = cm.predict({"duty": 0.5, "v_in": 24.0, "R_load": 4.0,
                    "dt": 2.0e-6, "duration_s": 6.0e-3})

    # 2) DC sweep: gain & efficiency vs duty
    duties = np.linspace(0.1, 0.9, 41)
    vout = np.array([m.steady_state(d, 24.0, 4.0)["vout"] for d in duties])
    ideal = np.array([m.ideal_gain(d) * 24.0 for d in duties])
    eff = np.array([m.steady_state(d, 24.0, 4.0)["efficiency"] for d in duties])

    print(f"Start-up settled Vout = {r['vout'][-1]:.3f} V")
    print(f"Peak efficiency over sweep = {eff.max()*100:.2f}% at d={duties[eff.argmax()]:.2f}")

    if not _HAVE_PLOTLY:
        print("plotly not available -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Output voltage transient", "Inductor current transient",
                        "DC gain vs duty (lossy vs ideal)", "Efficiency vs duty"),
    )
    fig.add_trace(go.Scatter(x=r["t"] * 1e3, y=r["vout"], name="Vout"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"] * 1e3, y=r["iL"], name="iL"), row=1, col=2)
    fig.add_trace(go.Scatter(x=duties, y=vout, name="Vout (lossy)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=duties, y=ideal, name="Vout (ideal)",
                             line=dict(dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=duties, y=eff * 100.0, name="efficiency %"), row=2, col=2)
    fig.update_xaxes(title_text="time [ms]", row=1, col=1)
    fig.update_xaxes(title_text="time [ms]", row=1, col=2)
    fig.update_xaxes(title_text="duty", row=2, col=1)
    fig.update_xaxes(title_text="duty", row=2, col=2)
    fig.update_layout(title="EC159 Buck-Boost F2a -- Averaged State-Space Model",
                      height=720, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"Report written to {os.path.abspath(out_html)}")


if __name__ == "__main__":
    run_report()
