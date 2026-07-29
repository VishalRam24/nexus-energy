"""
EC149 -- Biodiesel Transesterification -- F2a Physics-Lumped Kinetics
Optional Plotly report: concentration profiles, yield/conversion vs time,
methanol-ratio sweep, temperature effect. Plotly import is guarded.
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

    base = cm.predict({"methanol_ratio": 6.0, "T0_K": 333.15, "duration_min": 120.0})

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Species concentrations vs time (6:1, 60C)",
            "FAME yield & TG conversion vs time",
            "Final FAME yield vs methanol:oil ratio",
            "Conversion vs temperature (isothermal, 20 min)",
        ),
    )

    # 1. species profiles
    for sp, name in [("TG", "Triglyceride"), ("DG", "Diglyceride"),
                     ("MG", "Monoglyceride"), ("FAME", "FAME (biodiesel)"),
                     ("glycerol", "Glycerol")]:
        fig.add_trace(go.Scatter(x=base["t"], y=base[sp], name=name), row=1, col=1)

    # 2. yield / conversion
    fig.add_trace(go.Scatter(x=base["t"], y=base["FAME_yield"], name="FAME yield"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=base["t"], y=base["conversion"], name="TG conversion"),
                  row=1, col=2)

    # 3. methanol ratio sweep
    ratios = [3.0, 4.5, 6.0, 7.5, 9.0, 12.0]
    yields = [cm.predict({"methanol_ratio": r, "T0_K": 333.15,
                          "duration_min": 120.0})["FAME_yield_final"] for r in ratios]
    fig.add_trace(go.Scatter(x=ratios, y=yields, mode="lines+markers",
                             name="final FAME yield"), row=2, col=1)

    # 4. conversion vs temperature
    temps = [303.15, 313.15, 323.15, 333.15, 343.15]
    conv = [cm.predict({"methanol_ratio": 6.0, "T0_K": T, "duration_min": 20.0,
                        "isothermal": True})["conversion_final"] for T in temps]
    fig.add_trace(go.Scatter(x=[t - 273.15 for t in temps], y=conv,
                             mode="lines+markers", name="conversion @20min"),
                  row=2, col=2)

    fig.update_xaxes(title_text="time [min]", row=1, col=1)
    fig.update_yaxes(title_text="conc [mol/L]", row=1, col=1)
    fig.update_xaxes(title_text="time [min]", row=1, col=2)
    fig.update_xaxes(title_text="MeOH:oil [mol/mol]", row=2, col=1)
    fig.update_xaxes(title_text="temperature [C]", row=2, col=2)
    fig.update_layout(title="EC149 Biodiesel Transesterification — F2a Kinetics", height=820)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
