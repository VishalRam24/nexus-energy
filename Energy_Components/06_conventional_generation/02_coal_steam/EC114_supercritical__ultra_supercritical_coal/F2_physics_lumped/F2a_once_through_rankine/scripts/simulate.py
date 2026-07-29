"""
EC114 -- Supercritical / Ultra-Supercritical Coal Plant -- F2a Physics-Lumped
Plotly HTML simulation report (optional; plotly import is guarded).
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
    print("plotly not installed -- skipping HTML report (pip install plotly).")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario A: efficiency / power / CO2 vs part-load ---
plrs = np.linspace(0.30, 1.0, 30)
eta_net = np.array([m.compute_cycle(p)["eta_net"] for p in plrs])
P_net = np.array([m.compute_cycle(p)["P_net_MW"] for p in plrs])
co2 = np.array([m.compute_cycle(p)["co2_intensity_g_per_kwh"] for p in plrs])
coal = np.array([m.compute_cycle(p)["coal_rate_kgs"] for p in plrs])

# --- Scenario B: T-s style state-point ladder at full load ---
c = m.compute_cycle(1.0)
sp_h = c["state_points"]["h"]
sp_T = c["state_points"]["T"]
labels = ["1 cond out", "2 pump out", "2b feedwater", "3 HP in", "4 HP out",
          "5 RH1 in", "6 IP out", "7 RH2 in", "8 LP out"]

# --- Scenario C: evaporator thermal transient (load step down then up) ---
def load(t):
    if t < 600:
        return 1.0
    elif t < 1800:
        return 0.5
    return 1.0

r = m.simulate(load, dt=10.0, duration_s=2700.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Net Efficiency vs Part-Load (USC band)",
        "Net Power & Coal Rate vs Part-Load",
        "CO2 Intensity vs Part-Load",
        "Cycle State Points (h vs T, full load)",
        "Evaporator Metal Temperature Transient",
        "Furnace vs Steam Heat Duty (transient)",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.12,
    specs=[[{}, {"secondary_y": True}], [{}, {}], [{}, {}]],
)

fig.add_trace(go.Scatter(x=plrs, y=eta_net * 100, name="eta_net %",
              line=dict(color="#2ca02c")), row=1, col=1)
fig.add_hline(y=m.eta_subcritical_ref * 100, line_dash="dash",
              line_color="grey", row=1, col=1,
              annotation_text="subcritical ref")

fig.add_trace(go.Scatter(x=plrs, y=P_net, name="P_net MW",
              line=dict(color="#1f77b4")), row=1, col=2, secondary_y=False)
fig.add_trace(go.Scatter(x=plrs, y=coal, name="coal kg/s",
              line=dict(color="#8c564b")), row=1, col=2, secondary_y=True)

fig.add_trace(go.Scatter(x=plrs, y=co2, name="CO2 g/kWh",
              line=dict(color="#d62728"), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=sp_h, y=sp_T, mode="markers+lines+text",
              text=labels, textposition="top center", name="cycle",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=r["t"], y=r["T_evap"], name="T_evap C",
              line=dict(color="#ff7f0e"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=r["t"], y=r["Q_furnace_MW"], name="Q_furnace",
              line=dict(color="#d62728")), row=3, col=2)
fig.add_trace(go.Scatter(x=r["t"], y=r["Q_steam_MW"], name="Q_steam",
              line=dict(color="#1f77b4", dash="dot")), row=3, col=2)

for r_, c_, xl, yl in [
    (1, 1, "Part-load ratio", "Net efficiency (%)"),
    (1, 2, "Part-load ratio", "Net power (MW)"),
    (2, 1, "Part-load ratio", "CO2 (g/kWh)"),
    (2, 2, "Enthalpy h (kJ/kg)", "Temperature (C)"),
    (3, 1, "Time (s)", "Evaporator T (C)"),
    (3, 2, "Time (s)", "Heat duty (MW)"),
]:
    fig.update_xaxes(title_text=xl, row=r_, col=c_)
    fig.update_yaxes(title_text=yl, row=r_, col=c_)

fig.update_layout(
    title="<b>EC114 Supercritical/USC Coal -- F2a Once-Through Rankine + Evaporator ODE</b>",
    height=1050, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC114_F2a_once_through_rankine_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
