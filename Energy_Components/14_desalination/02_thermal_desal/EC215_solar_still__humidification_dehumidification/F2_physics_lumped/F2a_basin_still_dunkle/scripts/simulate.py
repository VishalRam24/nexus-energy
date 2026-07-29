"""
EC215 -- Solar Still / HDH -- F2a Basin Still (Dunkle)
Optional Plotly report: 2-day diurnal run showing temperatures, fluxes,
and cumulative distillate. Plotly import guarded so absence never crashes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run(out_html=None):
    cm = ComponentModel()
    r = cm.predict({"G_peak_W_m2": 900.0, "duration_s": 2 * 86400.0, "dt": 600.0})
    t_h = r["t"] / 3600.0

    print(f"2-day run: peak T_water = {r['T_water'].max()-273.15:.1f} C, "
          f"daily yield ~ {r['cumulative_distillate_kg'][-1]/2/cm._model.A:.2f} L/(m2.day)")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return r

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Temperatures & Irradiance", "Interior fluxes [W/m2]",
                        "Cumulative distillate [kg]"),
    )
    fig.add_trace(go.Scatter(x=t_h, y=r["T_water"] - 273.15, name="T_water [C]"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["T_glass"] - 273.15, name="T_glass [C]"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["G"] / 10.0, name="G/10 [W/m2]",
                             line=dict(dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["q_evap"], name="q_evap"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["q_conv"], name="q_conv"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["q_rad"], name="q_rad"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["cumulative_distillate_kg"],
                             name="distillate [kg]"), 3, 1)
    fig.update_xaxes(title_text="time [h]", row=3, col=1)
    fig.update_layout(title="EC215 F2a Single-Basin Solar Still (Dunkle)",
                      height=900)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return r


if __name__ == "__main__":
    run()
