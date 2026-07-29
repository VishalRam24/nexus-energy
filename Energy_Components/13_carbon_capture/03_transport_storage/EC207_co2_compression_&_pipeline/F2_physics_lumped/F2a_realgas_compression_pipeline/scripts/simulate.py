"""
EC207 -- CO2 Compression & Pipeline -- F2a
Optional Plotly HTML simulation report (import wrapped; absence won't crash).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# Scenario 1: nominal compression + transient
r1 = model.predict({"mass_flow_kg_s": 100.0, "P_outlet_bar": 150.0, "duration_s": 600.0})

# Scenario 2: SEC vs discharge pressure sweep
P_outs = np.linspace(100.0, 200.0, 25)
secs = [model._model.compress(P_out=p)["SEC_kWh_per_tCO2"] for p in P_outs]

# Scenario 3: pipeline dP vs length sweep
lengths = np.linspace(10.0, 400.0, 25)
dps = [model._model.pipeline_pressure_drop_bar(100.0, length_km=L, diameter_m=0.508)
       for L in lengths]


def main():
    if not _HAS_PLOTLY:
        print("plotly not installed -- skipping HTML report.")
        print(f"SEC nominal = {r1['SEC_kWh_per_tCO2']:.1f} kWh/tCO2, "
              f"P_disch = {r1['P_discharge_bar']:.0f} bar")
        return
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Per-stage discharge T (intercooled)",
        "SEC vs discharge pressure",
        "Pipeline dP vs length",
        "Lumped pressure transient"))
    stages = np.arange(1, len(r1["stage_T_discharge_K"]) + 1)
    fig.add_trace(go.Scatter(x=stages, y=r1["stage_T_discharge_K"],
                             mode="lines+markers", name="T_discharge"), row=1, col=1)
    fig.add_trace(go.Scatter(x=P_outs, y=secs, mode="lines", name="SEC"), row=1, col=2)
    fig.add_trace(go.Scatter(x=lengths, y=dps, mode="lines", name="dP"), row=2, col=1)
    tr = r1["transient"]
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["P_discharge_bar"],
                             mode="lines", name="P(t)"), row=2, col=2)
    out = os.path.join(OUTPUT_DIR, "simulation_report.html")
    fig.write_html(out)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
