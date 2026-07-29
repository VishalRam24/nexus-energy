"""
EC002 -- SOFC -- F2a Electrochemical -- Plotly HTML report.
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
model = ComponentModel()
m = model._model

# Polarization at various T
j_sw = np.linspace(0.01, 2.0, 200)
temps = [973.15, 1023.15, 1073.15, 1123.15]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Dynamic sim
r1 = m.simulate(0.5, 1073.15, dt=5, duration_s=3000)

fig = make_subplots(rows=2, cols=2,
    subplot_titles=["Polarization Curves", "Power Density",
                    "Thermal Transient (j=0.5)", "Efficiency Transient"])

for i, T in enumerate(temps):
    V = [m.cell_voltage(j, T) for j in j_sw]
    P = [j * v for j, v in zip(j_sw, V)]
    label = f"T={T-273.15:.0f}C"
    fig.add_trace(go.Scatter(x=j_sw, y=V, name=label, line=dict(color=colors[i])), row=1, col=1)
    fig.add_trace(go.Scatter(x=j_sw, y=P, name=label, showlegend=False,
                  line=dict(color=colors[i])), row=1, col=2)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              showlegend=False, line=dict(color="#d62728")), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["efficiency"], name="eta(t)",
              showlegend=False, line=dict(color="#2ca02c")), row=2, col=2)

for r, c, xl, yl in [(1,1,"j (A/cm2)","V (V)"),(1,2,"j (A/cm2)","P (W/cm2)"),
                       (2,1,"Time (s)","T (K)"),(2,2,"Time (s)","Efficiency")]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(title="<b>EC002 SOFC -- F2a Electrochemical + Thermal</b>",
                  height=800, template="plotly_white")
out = os.path.join(OUTPUT_DIR, "EC002_F2a_electrochemical_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
