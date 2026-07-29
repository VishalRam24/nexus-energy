"""
EC039 -- Organic Flow Battery (OFB) -- F2a Physics-Lumped Stack Model
Plotly HTML simulation report generator (optional; plotly import guarded).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
    print("plotly not installed -- skipping HTML report (model unaffected).")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: full discharge from SOC=0.9 ---
r_dis = model.predict({"current_A": 20.0, "soc0": 0.9, "T0_K": 298.15,
                       "dt": 10.0, "duration_s": 1500.0})

# --- Scenario 2: thermal transient at higher current ---
r_th = model.predict({"current_A": 40.0, "soc0": 0.8, "T0_K": 298.15,
                      "dt": 30.0, "duration_s": 3600.0})

# --- Scenario 3: long-horizon capacity fade (90 days, resting) ---
r_fade = model.predict({"current_A": 0.0, "soc0": 0.5, "T0_K": 305.0,
                        "dt": 3600.0, "duration_s": 90 * 24 * 3600.0})

if _HAS_PLOTLY:
    # polarization-style sweep (cell V vs current at fixed SOC)
    I_sweep = np.linspace(-45.0, 45.0, 200)
    V_chg_dis = [m.cell_voltage(0.6, I, 298.15) for I in I_sweep]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Discharge: SOC(t) (I=20 A)",
            "Discharge: Stack Voltage(t)",
            "Thermal Transient T(t) (I=40 A)",
            "V-I Curve at SOC=0.6 (charge<-/->discharge)",
            "Capacity Fade (90 days, resting @ 32C)",
            "Round-trip Energy Efficiency(t)",
        ],
        vertical_spacing=0.10, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["soc"], name="SOC",
                  line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["voltage"], name="V_stack",
                  line=dict(color="#2ca02c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=r_th["t"], y=r_th["temperature"], name="T",
                  line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=I_sweep, y=V_chg_dis, name="V_cell(I)",
                  line=dict(color="#9467bd")), row=2, col=2)
    fig.add_trace(go.Scatter(x=r_fade["t"] / 86400.0, y=r_fade["capacity"],
                  name="capacity", line=dict(color="#ff7f0e")), row=3, col=1)
    fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["efficiency"], name="eta",
                  line=dict(color="#17becf")), row=3, col=2)

    for r, c, xl, yl in [
        (1, 1, "Time (s)", "SOC (-)"),
        (1, 2, "Time (s)", "Stack Voltage (V)"),
        (2, 1, "Time (s)", "Temperature (K)"),
        (2, 2, "Current (A)", "Cell Voltage (V)"),
        (3, 1, "Time (days)", "Capacity (frac)"),
        (3, 2, "Time (s)", "Energy Efficiency (-)"),
    ]:
        fig.update_xaxes(title_text=xl, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)

    fig.update_layout(
        title="<b>EC039 Organic Flow Battery -- F2a Physics-Lumped Stack "
              "(electrochemical + SOC + fade + thermal ODE)</b>",
        height=1000, template="plotly_white", showlegend=False,
    )
    out_path = os.path.join(OUTPUT_DIR, "EC039_F2a_stack_model_report.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Report saved: {out_path}")
else:
    print(f"Discharge end SOC={r_dis['soc'][-1]:.3f}, "
          f"thermal end T={r_th['temperature'][-1]:.2f} K, "
          f"90-day capacity={r_fade['capacity'][-1]:.4f}")
