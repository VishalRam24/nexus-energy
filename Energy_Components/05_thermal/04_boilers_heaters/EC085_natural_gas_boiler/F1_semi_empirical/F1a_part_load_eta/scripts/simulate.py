"""
EC085 — Natural Gas Boiler — F1a Part-Load Efficiency
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
Q_rated = params["Q_rated"]

# -------------------------------------------------------------------------
# Sweep data
# -------------------------------------------------------------------------

plr_vals    = np.linspace(0.0, 1.0, 200)
supply_temps = [35.0, 45.0, 55.0, 65.0, 75.0]
colors       = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

data_by_T = {}
for T_sup in supply_temps:
    rows = []
    for plr in plr_vals:
        try:
            out = model.predict({"part_load_ratio": float(plr), "supply_temp": T_sup})
        except Exception:
            out = None
        rows.append(out)
    data_by_T[T_sup] = rows

# -------------------------------------------------------------------------
# Build figure
# -------------------------------------------------------------------------

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Efficiency vs Part-Load Ratio",
        "Thermal Output vs PLR",
        "Fuel Input vs PLR",
        "Gas Consumption vs PLR",
        "Efficiency vs Supply Temperature (at PLR=1)",
        "Condensing Correction Factor vs Supply Temperature",
    ],
    vertical_spacing=0.13,
    horizontal_spacing=0.10,
)

for i, T_sup in enumerate(supply_temps):
    rows = data_by_T[T_sup]
    valid_plr = [plr_vals[k] for k, r in enumerate(rows) if r is not None]
    eta       = [r["efficiency"]          for r in rows if r]
    q_out     = [r["thermal_output_kw"]   for r in rows if r]
    q_fuel    = [r["fuel_input_kw"]       for r in rows if r]
    v_gas     = [r["gas_consumption_m3h"] for r in rows if r]

    label = f"T_sup = {T_sup:.0f}°C"
    kw = dict(line=dict(color=colors[i]), name=label, legendgroup=label, showlegend=True)

    fig.add_trace(go.Scatter(x=valid_plr, y=eta,   **kw),                            row=1, col=1)
    fig.add_trace(go.Scatter(x=valid_plr, y=q_out,  **{**kw, "showlegend": False}),  row=1, col=2)
    fig.add_trace(go.Scatter(x=valid_plr, y=q_fuel, **{**kw, "showlegend": False}),  row=2, col=1)
    fig.add_trace(go.Scatter(x=valid_plr, y=v_gas,  **{**kw, "showlegend": False}),  row=2, col=2)

# Panel 5: Efficiency vs supply temperature at PLR=1
T_range = np.linspace(30.0, 80.0, 100)
for plr_val, col_name, c in [(1.0, "PLR=1.0", "#1f77b4"), (0.5, "PLR=0.5", "#ff7f0e"), (0.2, "PLR=0.2", "#2ca02c")]:
    eta_T = []
    for T in T_range:
        out = model.predict({"part_load_ratio": plr_val, "supply_temp": float(T)})
        eta_T.append(out["efficiency"])
    fig.add_trace(go.Scatter(
        x=list(T_range), y=eta_T,
        line=dict(color=c), name=col_name,
        legendgroup=col_name, showlegend=True,
    ), row=3, col=1)

# Panel 6: Condensing correction factor vs T_supply
from model import NaturalGasBoilerModel
m_raw = NaturalGasBoilerModel(params)
corr_vals = [m_raw.condensing_correction(T) for T in T_range]
fig.add_trace(go.Scatter(
    x=list(T_range), y=corr_vals,
    line=dict(color="#d62728", width=2),
    name="Condensing Factor", showlegend=False,
), row=3, col=2)
fig.add_hline(y=1.0, line_dash="dash", line_color="grey",
              annotation_text="Factor=1.0 (T=55°C)", row=3, col=2)

# Axis labels
axis_labels = [
    (1, 1, "Part-Load Ratio (PLR)", "Efficiency (-)"),
    (1, 2, "Part-Load Ratio (PLR)", "Thermal Output (kW)"),
    (2, 1, "Part-Load Ratio (PLR)", "Fuel Input (kW)"),
    (2, 2, "Part-Load Ratio (PLR)", "Gas Consumption (m³/h)"),
    (3, 1, "Supply Temperature (°C)", "Efficiency (-)"),
    (3, 2, "Supply Temperature (°C)", "Correction Factor (-)"),
]
for row, col, xl, yl in axis_labels:
    fig.update_xaxes(title_text=xl, row=row, col=col)
    fig.update_yaxes(title_text=yl, row=row, col=col)

fig.update_layout(
    title=dict(
        text=(
            f"<b>EC085 — Natural Gas Boiler (Condensing) — F1a Part-Load Efficiency</b><br>"
            f"<sup>Q_rated={Q_rated:.0f} kW, η_nom={params['eta_nom']*100:.0f}%, "
            f"LHV={params['LHV_gas']} MJ/m³, "
            f"Curve: a0={params['a0']}, a1={params['a1']}, a2={params['a2']}</sup>"
        ),
        x=0.5,
    ),
    height=1050,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC085_boiler_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
