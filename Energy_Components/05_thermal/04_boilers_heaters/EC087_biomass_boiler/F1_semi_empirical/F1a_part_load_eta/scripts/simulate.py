"""
EC087 — Biomass Boiler — F1a Part-Load Efficiency
Plotly HTML simulation report generator.
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
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")

model  = ComponentModel()
info   = model.get_info()
params = info["active_parameters"]
Q_rated = params["Q_rated"]

plr_vals = np.linspace(0.0, 1.0, 200)
eta, q_out, q_fuel, m_fuel = [], [], [], []
for plr in plr_vals:
    out = model.predict({"part_load_ratio": float(plr)})
    eta.append(out["efficiency"])
    q_out.append(out["thermal_output_kw"])
    q_fuel.append(out["fuel_input_kw"])
    m_fuel.append(out["fuel_mass_flow_kg_h"])

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Efficiency vs Part-Load Ratio",
        "Thermal Output vs PLR",
        "Fuel Input vs PLR",
        "Biomass Mass Flow vs PLR",
    ],
    vertical_spacing=0.13,
    horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=plr_vals, y=eta,    line=dict(color="#1f77b4", width=2),
                         name="Efficiency"), row=1, col=1)
fig.add_hline(y=params["eta_nom"], line_dash="dash", line_color="grey",
              annotation_text=f"eta_nom = {params['eta_nom']}", row=1, col=1)
fig.add_vline(x=params["PLR_min"], line_dash="dot", line_color="red",
              annotation_text=f"PLR_min = {params['PLR_min']}", row=1, col=1)

fig.add_trace(go.Scatter(x=plr_vals, y=q_out,  line=dict(color="#ff7f0e", width=2),
                         name="Q_out", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=plr_vals, y=q_fuel, line=dict(color="#2ca02c", width=2),
                         name="Q_fuel", showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=plr_vals, y=m_fuel, line=dict(color="#8c564b", width=2),
                         name="m_fuel", showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=1, col=1)
fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=1, col=2)
fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=2, col=1)
fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=2, col=2)
fig.update_yaxes(title_text="Efficiency (-)",        row=1, col=1)
fig.update_yaxes(title_text="Thermal Output (kW)",   row=1, col=2)
fig.update_yaxes(title_text="Fuel Input (kW)",       row=2, col=1)
fig.update_yaxes(title_text="Fuel Mass Flow (kg/h)", row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            f"<b>EC087 — Biomass Boiler — F1a Part-Load Efficiency</b><br>"
            f"<sup>Q_rated={Q_rated:.0f} kW, η_nom={params['eta_nom']*100:.0f}%, "
            f"LHV_dry={params['LHV_fuel_MJ_kg']} MJ/kg, "
            f"moisture={params['moisture_content']*100:.0f}%</sup>"
        ),
        x=0.5,
    ),
    height=820,
    template="plotly_white",
)

out_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
