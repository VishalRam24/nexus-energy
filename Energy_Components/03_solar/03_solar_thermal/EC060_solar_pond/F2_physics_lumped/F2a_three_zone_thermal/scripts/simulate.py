"""
EC060 -- Solar Pond (Salinity-Gradient) -- F2a Three-Zone Lumped
Plotly HTML simulation report generator (optional; degrades gracefully).
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
    print("plotly not installed; skipping HTML report (pip install plotly).")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: multi-day charge-up from cold start (steady daily-mean insolation)
r1 = model.predict({"G": 200.0, "T_lcz_init": 20.0, "T_amb": 20.0,
                    "duration_days": 365.0, "dt_hours": 12.0})

# Scenario 2: diurnal forcing on an already-warm pond (shows night Q_solar=0)
r2 = model.predict({"G": 700.0, "T_lcz_init": 80.0, "T_amb": 20.0,
                    "duration_days": 4.0, "dt_hours": 0.25, "diurnal": True})

# Scenario 3: charge then extract heat
r3 = model.predict({"G": 250.0, "T_lcz_init": 85.0, "T_amb": 20.0,
                    "duration_days": 90.0, "dt_hours": 12.0, "Q_extract_W": 3.0e5})

# Scenario 4: Beer-Lambert depth profile
depths = np.linspace(0.0, m.h_ucz + m.h_ncz + m.h_lcz, 100)
frac = m.transmitted_fraction(depths)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Annual LCZ Charge-up (G=200 W/m2)",
        "Diurnal Solar to LCZ (night = 0)",
        "Charge + Heat Extraction (300 kW)",
        "Beer-Lambert Solar Attenuation vs Depth",
    ],
    vertical_spacing=0.13, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t_days"], y=r1["T_lcz"], name="T_LCZ",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t_days"], y=r1["T_ucz"], name="T_UCZ",
              line=dict(color="#1f77b4")), row=1, col=1)

fig.add_trace(go.Scatter(x=r2["t_days"], y=r2["Q_solar_W"] / 1e3, name="Q_solar (kW)",
              line=dict(color="#ff7f0e"), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=r3["t_days"], y=r3["T_lcz"], name="T_LCZ extract",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)

# Depth axis pointing downward (depth increases as we go down)
fig.add_trace(go.Scatter(x=frac, y=depths, name="transmitted f",
              line=dict(color="#9C27B0"), showlegend=False), row=2, col=2)
# Mark zone boundaries
fig.add_hline(y=m.h_ucz, line_dash="dot", row=2, col=2,
              annotation_text="UCZ/NCZ")
fig.add_hline(y=m.h_ucz + m.h_ncz, line_dash="dot", row=2, col=2,
              annotation_text="NCZ/LCZ")

fig.update_xaxes(title_text="Time (days)", row=1, col=1)
fig.update_yaxes(title_text="Temperature (C)", row=1, col=1)
fig.update_xaxes(title_text="Time (days)", row=1, col=2)
fig.update_yaxes(title_text="Solar to LCZ (kW)", row=1, col=2)
fig.update_xaxes(title_text="Time (days)", row=2, col=1)
fig.update_yaxes(title_text="LCZ Temperature (C)", row=2, col=1)
fig.update_xaxes(title_text="Transmitted fraction", row=2, col=2)
fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=2, col=2)

fig.update_layout(
    title="<b>EC060 Solar Pond -- F2a Three-Zone Lumped Energy Balance</b>",
    height=850, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC060_F2a_three_zone_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
