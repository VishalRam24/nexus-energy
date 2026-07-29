"""
EC086 — Electric Boiler / Resistance Heater — F1a Constant Efficiency
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

model  = ComponentModel()
info   = model.get_info()
params = info["active_parameters"]
P_rated = params["P_rated_kw"]

plr_vals = np.linspace(0.0, 1.0, 200)

eta, q_out, p_in = [], [], []
for plr in plr_vals:
    out = model.predict({"part_load_ratio": float(plr)})
    eta.append(out["efficiency"])
    q_out.append(out["thermal_output_kw"])
    p_in.append(out["electrical_input_kw"])

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Efficiency vs Part-Load Ratio",
        "Thermal Output vs PLR",
        "Electrical Input vs PLR",
        "Q_out vs P_elec (linearity check)",
    ],
    vertical_spacing=0.13,
    horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=plr_vals, y=eta,    line=dict(color="#1f77b4", width=2),
                         name="Efficiency"), row=1, col=1)
fig.add_hline(y=params["eta_nom"], line_dash="dash", line_color="grey",
              annotation_text=f"eta_nom = {params['eta_nom']}", row=1, col=1)

fig.add_trace(go.Scatter(x=plr_vals, y=q_out,  line=dict(color="#ff7f0e", width=2),
                         name="Q_out", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=plr_vals, y=p_in,   line=dict(color="#2ca02c", width=2),
                         name="P_in",  showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=p_in,     y=q_out,  line=dict(color="#d62728", width=2),
                         name="Q vs P", showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="Part-Load Ratio (PLR)",  row=1, col=1)
fig.update_xaxes(title_text="Part-Load Ratio (PLR)",  row=1, col=2)
fig.update_xaxes(title_text="Part-Load Ratio (PLR)",  row=2, col=1)
fig.update_xaxes(title_text="Electrical Input (kW)",  row=2, col=2)
fig.update_yaxes(title_text="Efficiency (-)",         row=1, col=1)
fig.update_yaxes(title_text="Thermal Output (kW)",    row=1, col=2)
fig.update_yaxes(title_text="Electrical Input (kW)",  row=2, col=1)
fig.update_yaxes(title_text="Thermal Output (kW)",    row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            f"<b>EC086 — Electric Boiler / Resistance Heater — F1a Constant Eta</b><br>"
            f"<sup>P_rated={P_rated:.0f} kW, η_nom={params['eta_nom']*100:.1f}%, "
            f"P_standby={params['P_standby_kw']} kW</sup>"
        ),
        x=0.5,
    ),
    height=820,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
