"""
EC077 -- Microchannel Heat Exchanger -- F2a Lumped Transient
Optional Plotly HTML report. plotly import is guarded so absence is non-fatal.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except ImportError:
    _HAVE_PLOTLY = False
    print("plotly not installed -- skipping HTML report (model unaffected).")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: cold-start transient
r1 = model.predict({"T_h_in": 80.0, "T_c_in": 20.0, "dt": 1.0, "duration_s": 180.0})

# Scenario 2: steady profiles along the core (final-time node profiles)
nodes = np.arange(m.N)

# Scenario 3: effectiveness vs hot mass-flow sweep
mdots = np.linspace(0.02, 0.30, 12)
eps_sweep, dP_sweep, ntu_sweep = [], [], []
for md in mdots:
    rr = m.simulate(mdot_h=md, mdot_c=0.08, dt=20.0, duration_s=400.0)
    eps_sweep.append(rr["effectiveness"][-1])
    dP_sweep.append(rr["dP_h_Pa"])
    e, ntu, _, _ = m.epsilon_ntu_counterflow(md, 0.08)
    ntu_sweep.append(ntu)

print(f"Steady: T_h_out={r1['T_h_out'][-1]:.2f} C  T_c_out={r1['T_c_out'][-1]:.2f} C  "
      f"Q={r1['Q_kW'][-1]:.3f} kW  eps={r1['effectiveness'][-1]:.4f}  "
      f"UA={r1['UA']:.0f} W/K  h_h={r1['h_h']:.0f} W/m2K  Re_h={r1['Re_h']:.1f}  "
      f"dP_h={r1['dP_h_Pa']:.0f} Pa")

if _HAVE_PLOTLY:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cold-start transient (outlet temps)",
            "Steady temperature profile along core",
            "Effectiveness vs hot mass flow",
            "Hot-side pressure drop vs mass flow",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_h_out"], name="T_h_out",
                  line=dict(color="#d62728")), row=1, col=1)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_c_out"], name="T_c_out",
                  line=dict(color="#1f77b4")), row=1, col=1)

    fig.add_trace(go.Scatter(x=nodes, y=r1["T_h_profile"], name="hot",
                  line=dict(color="#d62728")), row=1, col=2)
    fig.add_trace(go.Scatter(x=nodes, y=r1["T_wall_profile"], name="wall",
                  line=dict(color="#7f7f7f")), row=1, col=2)
    fig.add_trace(go.Scatter(x=nodes, y=r1["T_c_profile"], name="cold",
                  line=dict(color="#1f77b4")), row=1, col=2)

    fig.add_trace(go.Scatter(x=mdots, y=eps_sweep, name="eps(ODE)",
                  line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=mdots, y=dP_sweep, name="dP_h",
                  line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)

    for r, c, xl, yl in [
        (1, 1, "Time (s)", "Temp (C)"),
        (1, 2, "Node index", "Temp (C)"),
        (2, 1, "mdot_h (kg/s)", "Effectiveness"),
        (2, 2, "mdot_h (kg/s)", "dP hot (Pa)"),
    ]:
        fig.update_xaxes(title_text=xl, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)

    fig.update_layout(
        title="<b>EC077 Microchannel HX -- F2a Lumped Transient (N-CV + wall ODEs)</b>",
        height=800, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    out_path = os.path.join(OUTPUT_DIR, "EC077_F2a_lumped_dynamic_report.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Report saved: {out_path}")
