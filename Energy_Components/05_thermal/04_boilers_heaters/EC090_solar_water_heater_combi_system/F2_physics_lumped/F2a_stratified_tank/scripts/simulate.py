"""
EC090 -- Solar Water Heater Combi System -- F2a Stratified Tank
Optional Plotly HTML simulation report (import guarded; not required to run).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly not installed -- skipping HTML report (model still usable).")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
r = model.predict({"dt": 300.0, "duration_s": 86400.0})
hrs = r["t"] / 3600.0
N = model._model.N

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Stratified node temperatures (top..bottom)",
        "Collector useful gain & auxiliary fuel",
        "Combined load + pump state",
        f"Daily solar fraction = {r['solar_fraction']:.3f}",
    ],
    vertical_spacing=0.13, horizontal_spacing=0.10,
)

for i in range(N):
    fig.add_trace(go.Scatter(x=hrs, y=r["T_nodes"][i] - 273.15,
                  name=f"node {i+1}"), row=1, col=1)
fig.add_trace(go.Scatter(x=hrs, y=r["Q_solar"], name="Q_solar [W]",
              line=dict(color="#d62728")), row=1, col=2)
fig.add_trace(go.Scatter(x=hrs, y=r["Q_aux_fuel"], name="Q_aux fuel [W]",
              line=dict(color="#ff7f0e")), row=1, col=2)
fig.add_trace(go.Scatter(x=hrs, y=r["Q_load"], name="Q_load [W]",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=hrs, y=r["pump_on"], name="pump on",
              line=dict(color="#1f77b4", dash="dot")), row=2, col=1)

labels = ["solar", "aux (delivered)"]
vals = [r["E_solar_J"] / 3.6e6, r["E_aux_delivered_J"] / 3.6e6]
fig.add_trace(go.Bar(x=labels, y=vals, marker_color=["#d62728", "#ff7f0e"],
              showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="hour", row=1, col=1)
fig.update_yaxes(title_text="T [C]", row=1, col=1)
fig.update_yaxes(title_text="kWh", row=2, col=2)
fig.update_layout(title="<b>EC090 Solar Combi -- F2a Stratified Tank (Hottel-Whillier + N-node ODE)</b>",
                  height=850, template="plotly_white")

out = os.path.join(OUTPUT_DIR, "EC090_F2a_stratified_tank_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
