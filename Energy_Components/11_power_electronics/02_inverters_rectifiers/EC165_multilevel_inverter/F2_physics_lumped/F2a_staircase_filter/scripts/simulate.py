"""
EC165 -- Multilevel Inverter -- F2a Physics-Lumped
Optional Plotly report: staircase waveforms, filtered output, THD vs levels,
efficiency vs level count. Plotly import is guarded so absence does not crash.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    cm = ComponentModel()
    m = cm._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Staircase pole voltage (3 vs 9 levels)",
            "Filtered AC output (v_C), 5-level",
            "THD vs level count",
            "Efficiency vs level count",
        ),
    )

    # (1) staircase waveforms
    for N, color in [(3, "firebrick"), (9, "royalblue")]:
        t, v = m.waveform(1.0, N, n_samples=2000)
        fig.add_trace(go.Scatter(x=t * 1000, y=v, name=f"{N}-level",
                                 line=dict(color=color)), row=1, col=1)

    # (2) filtered output
    r = m.simulate(m=1.0, n_levels=5, n_periods=6)
    fig.add_trace(go.Scatter(x=r["t"] * 1000, y=r["v_pole"], name="pole (5L)",
                             line=dict(color="lightgray")), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"] * 1000, y=r["v_out"], name="v_out (5L)",
                             line=dict(color="green")), row=1, col=2)

    # (3) THD vs levels  (4) efficiency vs levels
    levels = [2, 3, 5, 7, 9, 11, 15, 21]
    thds, effs = [], []
    for N in levels:
        thds.append(m.thd(1.0, N) * 100.0)
        rr = m.simulate(m=1.0, n_levels=N, n_periods=4)
        effs.append(rr["efficiency"] * 100.0)
    fig.add_trace(go.Scatter(x=levels, y=thds, mode="lines+markers",
                             name="THD %"), row=2, col=1)
    fig.add_trace(go.Scatter(x=levels, y=effs, mode="lines+markers",
                             name="Efficiency %"), row=2, col=2)

    fig.update_xaxes(title_text="time [ms]", row=1, col=1)
    fig.update_xaxes(title_text="time [ms]", row=1, col=2)
    fig.update_xaxes(title_text="levels N", row=2, col=1)
    fig.update_xaxes(title_text="levels N", row=2, col=2)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="V", row=1, col=2)
    fig.update_yaxes(title_text="THD %", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency %", row=2, col=2)
    fig.update_layout(title="EC165 Multilevel Inverter F2a -- Physics-Lumped",
                      height=800, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
