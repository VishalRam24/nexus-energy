"""
EC176 -- PMSM -- F2a dq-Frame Dynamic Model -- Simulation report generator.

Generates an interactive Plotly HTML report with:
  1. Speed step response (command 1500 rpm)
  2. Current waveforms (i_d, i_q vs time)
  3. Torque vs time
  4. Load step response
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import PMSMF2a

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly is required. Install with: pip install plotly")
    sys.exit(1)

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def load_model():
    with open(_PARAMS_PATH) as f:
        params = json.load(f)
    return PMSMF2a(params)


def build_report():
    m = load_model()

    # ---- Scenario 1: Speed step response at 1000 rpm, 5 Nm load ----
    r1 = m.simulate_speed_control(1000.0, T_load_Nm=5.0, dt=1e-4, duration_s=3.0)

    # ---- Scenario 2: Load step -- 2 Nm then 10 Nm at t=2s ----
    def T_load_step(t):
        return 2.0 if t < 2.0 else 10.0

    r2 = m.simulate_speed_control(1000.0, T_load_Nm=T_load_step, dt=1e-4, duration_s=5.0)

    # ---- Build figures ----
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "1. Speed Step Response (Ref = 1000 rpm, Load = 5 Nm)",
            "2. Current Waveforms (i_d, i_q) -- Speed Control",
            "3. Electromagnetic Torque -- Speed Control",
            "4. Load Step Response (2 Nm -> 10 Nm at t=2s)",
        ],
        vertical_spacing=0.08,
    )

    # Plot 1: Speed step response
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["speed_rpm"], name="Speed",
                             line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r1["t"][0], r1["t"][-1]], y=[1000, 1000],
                             name="Reference", line=dict(color="red", dash="dash")),
                  row=1, col=1)
    fig.update_yaxes(title_text="Speed [rpm]", row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=1, col=1)

    # Plot 2: Currents
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["i_d"], name="i_d",
                             line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["i_q"], name="i_q",
                             line=dict(color="orange")), row=2, col=1)
    fig.update_yaxes(title_text="Current [A]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)

    # Plot 3: Torque
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["torque"], name="T_e",
                             line=dict(color="purple")), row=3, col=1)
    fig.add_trace(go.Scatter(x=[r1["t"][0], r1["t"][-1]], y=[5, 5],
                             name="T_load", line=dict(color="gray", dash="dot")),
                  row=3, col=1)
    fig.update_yaxes(title_text="Torque [Nm]", row=3, col=1)
    fig.update_xaxes(title_text="Time [s]", row=3, col=1)

    # Plot 4: Load step response
    fig.add_trace(go.Scatter(x=r2["t"], y=r2["speed_rpm"], name="Speed (load step)",
                             line=dict(color="blue"), showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=[r2["t"][0], r2["t"][-1]], y=[1000, 1000],
                             name="Reference", line=dict(color="red", dash="dash"),
                             showlegend=False), row=4, col=1)
    # Show load profile
    t_load_vis = np.array([0, 1.999, 2.0, 5.0])
    T_load_vis = np.array([2.0, 2.0, 10.0, 10.0])
    fig.add_trace(go.Scatter(x=t_load_vis, y=T_load_vis, name="T_load step",
                             line=dict(color="gray", dash="dot"), yaxis="y8"),
                  row=4, col=1)
    fig.update_yaxes(title_text="Speed [rpm]", row=4, col=1)
    fig.update_xaxes(title_text="Time [s]", row=4, col=1)

    fig.update_layout(
        title_text="EC176 PMSM -- F2a dq-Frame Dynamic Model -- Simulation Report",
        height=1400,
        template="plotly_white",
        showlegend=True,
    )

    fig.write_html(_OUTPUT_PATH, include_plotlyjs="cdn")
    print(f"Report saved to: {os.path.abspath(_OUTPUT_PATH)}")


if __name__ == "__main__":
    build_report()
