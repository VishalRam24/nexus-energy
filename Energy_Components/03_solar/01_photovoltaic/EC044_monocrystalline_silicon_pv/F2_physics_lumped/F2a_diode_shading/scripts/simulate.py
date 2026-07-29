"""
EC044 -- Mono-Si PV -- F2a Diode Shading -- Plotly HTML report.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: pip install plotly"); sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)
cm = ComponentModel()
m = cm._model

fig = make_subplots(rows=2, cols=2,
    subplot_titles=["I-V Uniform Irradiance", "P-V Uniform Irradiance",
                    "I-V Partial Shading", "P-V Partial Shading"])

# Uniform at various G
for G_val, color in [(1000, "#1f77b4"), (800, "#ff7f0e"), (600, "#2ca02c"), (400, "#d62728")]:
    G = np.full(60, float(G_val))
    r = m.iv_curve(G, 25.0)
    lab = f"G={G_val}"
    fig.add_trace(go.Scatter(x=r["V"], y=r["I"], name=lab, line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["V"], y=r["P"], name=lab, showlegend=False,
                  line=dict(color=color)), row=1, col=2)

# Shading scenarios
scenarios = [
    ("No shade", np.full(60, 1000.0), "#1f77b4"),
    ("10 cells @ 200", np.concatenate([np.full(10, 200.0), np.full(50, 1000.0)]), "#ff7f0e"),
    ("20 cells @ 300", np.concatenate([np.full(20, 300.0), np.full(40, 1000.0)]), "#2ca02c"),
    ("30 cells @ 100", np.concatenate([np.full(30, 100.0), np.full(30, 1000.0)]), "#d62728"),
]

for name, G, color in scenarios:
    r = m.iv_curve(G, 25.0, N_points=300)
    fig.add_trace(go.Scatter(x=r["V"], y=r["I"], name=name, line=dict(color=color)), row=2, col=1)
    fig.add_trace(go.Scatter(x=r["V"], y=r["P"], name=name, showlegend=False,
                  line=dict(color=color)), row=2, col=2)

for r, c, xl, yl in [(1,1,"V (V)","I (A)"),(1,2,"V (V)","P (W)"),
                       (2,1,"V (V)","I (A)"),(2,2,"V (V)","P (W)")]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(title="<b>EC044 Mono-Si PV -- F2a Diode + Partial Shading</b>",
                  height=800, template="plotly_white")
out = os.path.join(OUTPUT_DIR, "EC044_F2a_diode_shading_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
