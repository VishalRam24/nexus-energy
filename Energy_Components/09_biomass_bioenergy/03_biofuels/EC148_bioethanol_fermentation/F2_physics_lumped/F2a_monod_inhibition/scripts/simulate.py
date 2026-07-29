"""
EC148 -- Bioethanol Fermentation -- F2a Monod + Luong Inhibition
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash; run with `python3 scripts/simulate.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    r = cm.predict({"duration_h": 60.0, "dt_h": 0.25})

    print("EC148 F2a Bioethanol Fermentation -- batch summary")
    print(f"  Final ethanol     : {r['ethanol_final_g_L']:.2f} g/L")
    print(f"  Final biomass     : {r['biomass_final_g_L']:.2f} g/L")
    print(f"  Glucose consumed  : {r['glucose_consumed_g_L']:.2f} g/L")
    print(f"  Ethanol yield     : {r['ethanol_yield_g_g']:.3f} g/g (<= 0.511)")
    print(f"  Ferment efficiency: {r['ferment_efficiency']*100:.1f} %")
    print(f"  Productivity      : {r['productivity_g_L_h']:.3f} g/(L.h)")
    print(f"  Peak temperature  : {max(r['temperature']):.2f} K")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -- skip plotting
        print(f"  [plotly unavailable: {e}; skipping HTML report]")
        return

    t = r["t"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Concentrations", "Specific growth rate mu",
                        "Broth temperature", "Ethanol yield vs glucose consumed"),
    )
    fig.add_trace(go.Scatter(x=t, y=r["glucose"], name="Glucose g/L"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["biomass"], name="Biomass g/L"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["ethanol"], name="Ethanol g/L"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["mu"], name="mu 1/h"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["temperature"], name="T K"), 2, 1)
    fig.add_trace(go.Scatter(x=r["glucose"], y=r["ethanol"], name="EtOH vs glucose"), 2, 2)
    fig.update_layout(title="EC148 F2a Bioethanol Fermentation (Monod + Luong inhibition)",
                      height=750)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"  Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
