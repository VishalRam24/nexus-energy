"""
EC062 -- HAWT Onshore -- F2a BEM Steady -- Plotly HTML report.
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

V_range = np.arange(4, 25, 0.5)
P, Cp_arr, Ct_arr, T_arr = [], [], [], []
for V in V_range:
    r = m.solve(V)
    P.append(r["power_kw"]); Cp_arr.append(r["Cp"]); Ct_arr.append(r["Ct"])
    T_arr.append(r["thrust_kN"])

# Blade loads at V=10
r10 = m.solve(10.0)
r_pos = m.r
a_arr = [bl["a"] for bl in r10["blade_loads"]]
alpha_arr = [bl["alpha_deg"] for bl in r10["blade_loads"]]

fig = make_subplots(rows=2, cols=2,
    subplot_titles=["Power Curve", "Cp vs Wind Speed",
                    "Axial Induction Factor", "AoA Distribution"])

fig.add_trace(go.Scatter(x=V_range, y=P, line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=V_range, y=Cp_arr, line=dict(color="#ff7f0e")), row=1, col=2)
fig.add_trace(go.Scatter(x=r_pos, y=a_arr, line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r_pos, y=alpha_arr, line=dict(color="#d62728")), row=2, col=2)

for r, c, xl, yl in [(1,1,"V (m/s)","Power (kW)"),(1,2,"V (m/s)","Cp"),
                       (2,1,"r (m)","a"),(2,2,"r (m)","AoA (deg)")]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(title="<b>EC062 HAWT -- F2a BEM Steady</b>",
                  height=800, template="plotly_white", showlegend=False)
out = os.path.join(OUTPUT_DIR, "EC062_F2a_bem_steady_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
