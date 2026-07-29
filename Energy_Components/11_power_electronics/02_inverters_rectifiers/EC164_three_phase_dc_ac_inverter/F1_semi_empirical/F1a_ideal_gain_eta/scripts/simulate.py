"""
EC164 — Three-Phase DC-AC Inverter — F1a Ideal Gain + Efficiency
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model  = ComponentModel()
info   = model.get_info()
params = info["active_parameters"]
P_rated = params["P_rated"]

# -------------------------------------------------------------------------
# Sweep 1: Efficiency vs PLR
# -------------------------------------------------------------------------

plr_vals = np.linspace(0.01, 1.0, 200)
eff_vals  = []
ploss_vals = []
for plr in plr_vals:
    out = model.predict({"v_dc": 800.0, "p_load": plr * P_rated, "modulation_index": 0.9})
    eff_vals.append(out["efficiency"])
    ploss_vals.append(out["p_loss_W"])

# -------------------------------------------------------------------------
# Sweep 2: V_ac vs modulation index (various V_dc)
# -------------------------------------------------------------------------

m_vals = np.linspace(0.0, 1.0, 200)
v_dc_levels = [600.0, 700.0, 800.0, 900.0, 1000.0]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# -------------------------------------------------------------------------
# Sweep 3: Power loss breakdown vs load
# -------------------------------------------------------------------------

p_load_vals = np.linspace(1000.0, P_rated, 200)
p_in_vals   = []
for p in p_load_vals:
    out = model.predict({"v_dc": 800.0, "p_load": float(p), "modulation_index": 0.9})
    p_in_vals.append(out["p_in_W"])

# -------------------------------------------------------------------------
# Build figure
# -------------------------------------------------------------------------

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Efficiency vs Part-Load Ratio",
        "AC Output Voltage vs Modulation Index",
        "Power Loss vs Output Power",
        "Power Input vs Output Power",
        "AC Current vs Output Power (m=0.9, PF=1)",
        "Efficiency Map: PLR vs V_dc",
    ],
    vertical_spacing=0.13,
    horizontal_spacing=0.10,
)

# Panel 1: Efficiency vs PLR
fig.add_trace(go.Scatter(
    x=list(plr_vals), y=eff_vals,
    line=dict(color="#1f77b4", width=2),
    name="Efficiency", showlegend=True,
), row=1, col=1)

# Panel 2: V_ac vs m for different V_dc
for i, v_dc in enumerate(v_dc_levels):
    v_ac_vals = [model.predict({"v_dc": v_dc, "p_load": P_rated * 0.8,
                                 "modulation_index": float(m)})["v_ac_rms_V"]
                  for m in m_vals]
    fig.add_trace(go.Scatter(
        x=list(m_vals), y=v_ac_vals,
        line=dict(color=colors[i]),
        name=f"V_dc={v_dc:.0f}V", legendgroup=f"vdc{v_dc}",
        showlegend=True,
    ), row=1, col=2)

# Panel 3: Power loss vs output
fig.add_trace(go.Scatter(
    x=list(p_load_vals / 1000), y=ploss_vals,
    line=dict(color="#d62728", width=2),
    name="P_loss", showlegend=False,
), row=2, col=1)

# Panel 4: Power in vs power out (Sankey-style)
fig.add_trace(go.Scatter(
    x=list(p_load_vals / 1000), y=p_in_vals,
    line=dict(color="#2ca02c", width=2),
    name="P_in", showlegend=False,
), row=2, col=2)
fig.add_trace(go.Scatter(
    x=list(p_load_vals / 1000), y=list(p_load_vals),
    line=dict(color="#aec7e8", width=1.5, dash="dash"),
    name="P_out (ideal)", showlegend=False,
), row=2, col=2)

# Panel 5: AC current vs load
i_ac_vals = []
for p in p_load_vals:
    out = model.predict({"v_dc": 800.0, "p_load": float(p), "modulation_index": 0.9})
    i_ac_vals.append(out["i_ac_rms_A"])
fig.add_trace(go.Scatter(
    x=list(p_load_vals / 1000), y=i_ac_vals,
    line=dict(color="#ff7f0e", width=2),
    name="I_ac_rms", showlegend=False,
), row=3, col=1)

# Panel 6: Efficiency heatmap (PLR vs V_dc)
plr_heat = np.linspace(0.05, 1.0, 40)
vdc_heat = np.linspace(600.0, 1000.0, 30)
Z = np.zeros((len(vdc_heat), len(plr_heat)))
for i, v in enumerate(vdc_heat):
    for j, plr in enumerate(plr_heat):
        out = model.predict({"v_dc": float(v), "p_load": plr * P_rated,
                              "modulation_index": 0.9})
        Z[i, j] = out["efficiency"]

fig.add_trace(go.Heatmap(
    x=list(plr_heat), y=list(vdc_heat), z=Z,
    colorscale="RdYlGn", zmin=0.90, zmax=0.98,
    colorbar=dict(title="eta", len=0.3, y=0.12),
    showlegend=False,
), row=3, col=2)

# Axis labels
axis_labels = [
    (1, 1, "Part-Load Ratio (PLR)", "Efficiency (-)"),
    (1, 2, "Modulation Index (m)", "V_ac_rms (V)"),
    (2, 1, "Output Power (kW)", "Power Loss (W)"),
    (2, 2, "Output Power (kW)", "Power (W)"),
    (3, 1, "Output Power (kW)", "I_ac_rms (A)"),
    (3, 2, "PLR", "V_dc (V)"),
]
for row, col, xl, yl in axis_labels:
    fig.update_xaxes(title_text=xl, row=row, col=col)
    fig.update_yaxes(title_text=yl, row=row, col=col)

fig.update_layout(
    title=dict(
        text=(
            f"<b>EC164 — Three-Phase DC-AC Inverter — F1a Ideal Gain + Efficiency</b><br>"
            f"<sup>V_dc_rated={params['V_dc_rated']:.0f} V, "
            f"P_rated={P_rated/1000:.0f} kW, "
            f"η_rated={params['eta_rated']*100:.1f}%, "
            f"f_sw={params['f_sw']/1000:.0f} kHz, IGBT-based</sup>"
        ),
        x=0.5,
    ),
    height=1050,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC164_inverter_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
