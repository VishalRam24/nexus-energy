"""
EC008 -- PEMEL -- F2a Electrochemical -- Plotly HTML report.
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

j_sw = np.linspace(0.01, 2.5, 200)
temps = [333.15, 353.15, 373.15]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

r1 = m.simulate(1.5, 333.15, 30.0, 0.5, 300.0)

fig = make_subplots(rows=2, cols=2,
    subplot_titles=["V-I Curves", "H2 Production vs j", "Thermal Transient", "Efficiency Transient"])

for i, T in enumerate(temps):
    V = [m.cell_voltage(j, T, 30) for j in j_sw]
    h2 = [m.h2_production_rate(j, T) * m.N_cells * 2.016e-3 * 3600 for j in j_sw]
    lab = f"T={T-273.15:.0f}C"
    fig.add_trace(go.Scatter(x=j_sw, y=V, name=lab, line=dict(color=colors[i])), row=1, col=1)
    fig.add_trace(go.Scatter(x=j_sw, y=h2, name=lab, showlegend=False,
                  line=dict(color=colors[i])), row=1, col=2)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], showlegend=False,
              line=dict(color="#d62728")), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["efficiency"], showlegend=False,
              line=dict(color="#2ca02c")), row=2, col=2)

for r, c, xl, yl in [(1,1,"j (A/cm2)","V (V)"),(1,2,"j (A/cm2)","H2 (kg/hr)"),
                       (2,1,"Time (s)","T (K)"),(2,2,"Time (s)","Efficiency")]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(title="<b>EC008 PEMEL -- F2a Electrochemical + Thermal</b>",
                  height=800, template="plotly_white")
out = os.path.join(OUTPUT_DIR, "EC008_F2a_electrochemical_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
