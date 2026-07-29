"""
EC013 -- LH2 Storage -- F2a Two-Phase Cryogenic Tank
Optional Plotly simulation report.  Plotly import is wrapped so its absence
does not crash; running this file is not required for the build.

Generates a 4-panel report of a sealed dormancy run:
  (1) self-pressurization P(t),  (2) saturation temperature T(t),
  (3) liquid / vapor mass split,  (4) boil-off rate (BOR %/day).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()

    # Sealed 30-day dormancy: self-pressurize, hit vent, then regulate.
    sealed = cm.predict({"fill_fraction": 0.90, "T_ambient_K": 298.15,
                         "P0_bar": 1.01325, "duration_s": 30 * 86400.0,
                         "n_steps": 600, "sealed": True})
    # Open-vent boil-off reference (constant-pressure NBP loss).
    vented = cm.predict({"fill_fraction": 0.90, "T_ambient_K": 298.15,
                         "P0_bar": 1.01325, "duration_s": 30 * 86400.0,
                         "n_steps": 600, "sealed": False})

    days_s = sealed["t"] / 86400.0
    days_v = vented["t"] / 86400.0

    print("Sealed 30-day dormancy (90% fill, T_amb=298 K):")
    print(f"  P:   {sealed['pressure'][0]:.3f} -> {sealed['pressure'][-1]:.3f} bar")
    print(f"  T:   {sealed['temperature'][0]:.3f} -> {sealed['temperature'][-1]:.3f} K")
    print(f"  m_L: {sealed['m_liquid'][0]:.3f} -> {sealed['m_liquid'][-1]:.3f} kg")
    print(f"Open-vent: BOR={vented['BOR_pct_day'][0]:.3f} %/day, "
          f"lost {vented['m_total'][0]-vented['m_total'][-1]:.3f} kg in 30 d")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Self-pressurization P(t)", "Saturation temperature T(t)",
                        "Liquid / vapor mass split", "Boil-off rate (BOR)"))

    fig.add_trace(go.Scatter(x=days_s, y=sealed["pressure"], name="P sealed",
                             line=dict(color="crimson")), row=1, col=1)
    fig.add_hline(y=cm._model.P_vent, line_dash="dash", line_color="gray",
                  annotation_text="vent set-point", row=1, col=1)

    fig.add_trace(go.Scatter(x=days_s, y=sealed["temperature"], name="T sealed",
                             line=dict(color="darkorange")), row=1, col=2)

    fig.add_trace(go.Scatter(x=days_s, y=sealed["m_liquid"], name="m_liquid",
                             line=dict(color="navy")), row=2, col=1)
    fig.add_trace(go.Scatter(x=days_s, y=sealed["m_vapor"], name="m_vapor",
                             line=dict(color="steelblue")), row=2, col=1)

    fig.add_trace(go.Scatter(x=days_v, y=vented["BOR_pct_day"], name="BOR (open vent)",
                             line=dict(color="green")), row=2, col=2)

    fig.update_xaxes(title_text="time [days]")
    fig.update_yaxes(title_text="P [bar]", row=1, col=1)
    fig.update_yaxes(title_text="T [K]", row=1, col=2)
    fig.update_yaxes(title_text="mass [kg]", row=2, col=1)
    fig.update_yaxes(title_text="BOR [%/day]", row=2, col=2)
    fig.update_layout(title="EC013 LH2 Storage F2a -- Two-Phase Cryogenic Tank",
                      height=750, width=1100, template="plotly_white")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written: {os.path.abspath(out_html)}")


if __name__ == "__main__":
    run_report()
