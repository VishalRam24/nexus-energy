"""
EC065 -- Offshore Fixed-Bottom Wind Turbine -- F2a BEM Steady -- Plotly HTML report.
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
    print("ERROR: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cm = ComponentModel()
m = cm._model

# ---- 1. Power curve with pitch control ----
V_range = np.arange(3, 25.5, 0.5)
pc = m.power_curve(V_range, pitch_control=True)

# ---- 2. Cp vs TSR sweep ----
tsr_range = np.arange(3, 12, 0.25)
Cp_tsr = []
for tsr in tsr_range:
    r = m.solve(10.0, pitch_deg=0.0, tip_speed_ratio=tsr)
    Cp_tsr.append(r["Cp"])

# ---- 3. Blade element distribution at rated wind ----
r_rated = m.solve(m.rated_wind)
r_pos = [bl["r"] for bl in r_rated["blade_loads"]]
a_arr = [bl["a"] for bl in r_rated["blade_loads"]]
ap_arr = [bl["a_prime"] for bl in r_rated["blade_loads"]]
alpha_arr = [bl["alpha_deg"] for bl in r_rated["blade_loads"]]
dT_arr = [bl["dThrust_N"] / 1000.0 for bl in r_rated["blade_loads"]]
dP_arr = [bl["dPower_W"] / 1000.0 for bl in r_rated["blade_loads"]]

# ---- Build figure (3 rows x 2 cols) ----
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Power Curve (with Pitch Control)",
        "Thrust Curve",
        "Cp vs Tip-Speed Ratio",
        "Blade Pitch Schedule",
        "Blade Element Forces (at Rated Wind)",
        "Induction & AoA Distribution (at Rated Wind)",
    ],
)

# Row 1, Col 1 -- Power curve
fig.add_trace(
    go.Scatter(x=V_range, y=pc["power_kw"], name="Power",
               line=dict(color="#1f77b4", width=2)),
    row=1, col=1,
)
fig.add_hline(y=m.rated_power_kw, line_dash="dash", line_color="gray",
              annotation_text=f"Rated {m.rated_power_kw:.0f} kW",
              row=1, col=1)

# Row 1, Col 2 -- Thrust curve
fig.add_trace(
    go.Scatter(x=V_range, y=pc["thrust_kN"], name="Thrust",
               line=dict(color="#d62728", width=2)),
    row=1, col=2,
)

# Row 2, Col 1 -- Cp vs TSR
fig.add_trace(
    go.Scatter(x=tsr_range, y=Cp_tsr, name="Cp",
               line=dict(color="#ff7f0e", width=2)),
    row=2, col=1,
)
fig.add_hline(y=16.0 / 27.0, line_dash="dash", line_color="gray",
              annotation_text="Betz limit (0.593)",
              row=2, col=1)

# Row 2, Col 2 -- Pitch schedule
fig.add_trace(
    go.Scatter(x=V_range, y=pc["pitch_deg"], name="Pitch",
               line=dict(color="#9467bd", width=2)),
    row=2, col=2,
)

# Row 3, Col 1 -- Blade element forces
fig.add_trace(
    go.Scatter(x=r_pos, y=dT_arr, name="dThrust (kN)",
               line=dict(color="#2ca02c", width=2)),
    row=3, col=1,
)
fig.add_trace(
    go.Scatter(x=r_pos, y=dP_arr, name="dPower (kW)",
               line=dict(color="#1f77b4", width=2, dash="dash")),
    row=3, col=1,
)

# Row 3, Col 2 -- Induction factors and AoA
fig.add_trace(
    go.Scatter(x=r_pos, y=a_arr, name="a (axial)",
               line=dict(color="#e377c2", width=2)),
    row=3, col=2,
)
fig.add_trace(
    go.Scatter(x=r_pos, y=alpha_arr, name="AoA (deg)",
               line=dict(color="#8c564b", width=2, dash="dot")),
    row=3, col=2,
)

# Axis labels
axis_labels = [
    (1, 1, "Wind Speed (m/s)", "Power (kW)"),
    (1, 2, "Wind Speed (m/s)", "Thrust (kN)"),
    (2, 1, "Tip-Speed Ratio", "Cp"),
    (2, 2, "Wind Speed (m/s)", "Pitch (deg)"),
    (3, 1, "Radial Position (m)", "Force (kN) / Power (kW)"),
    (3, 2, "Radial Position (m)", "a / AoA (deg)"),
]
for r, c, xl, yl in axis_labels:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC065 Offshore Fixed-Bottom Wind Turbine -- F2a BEM Steady</b>",
    height=1100,
    template="plotly_white",
    showlegend=True,
    legend=dict(orientation="h", y=-0.05),
)

out = os.path.join(OUTPUT_DIR, "EC065_F2a_bem_steady_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
