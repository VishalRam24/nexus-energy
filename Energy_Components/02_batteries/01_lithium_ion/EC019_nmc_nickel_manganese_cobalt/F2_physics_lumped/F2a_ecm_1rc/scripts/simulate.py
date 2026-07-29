"""
EC019 -- NMC Battery -- F2a ECM 1-RC -- Simulation & HTML Report Generator

Generates interactive Plotly charts showing:
1. Voltage vs time for 1C discharge
2. SOC vs time
3. V_rc (polarization) dynamics -- RC relaxation after current step
4. Comparison: ECM voltage vs F1a static voltage at same current
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required: pip install plotly")
    sys.exit(1)


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    Q = model.params["cell"]["capacity"]["value"]
    I_1C = Q

    # --- Panel 1 & 2: 1C discharge ---
    dt = 1.0
    n_steps = int(4000)  # slightly more than 1h to see cutoff
    current_profile = np.full(n_steps, I_1C)
    result = model.predict({"current": current_profile, "dt": dt, "soc_init": 1.0})

    # Trim to where voltage hits cutoff
    v_min = model.params["cell"]["voltage_min"]["value"]
    valid = result["voltage"] > v_min + 0.001
    if not np.all(valid):
        cutoff_idx = np.argmin(valid)
    else:
        cutoff_idx = n_steps
    t_min = result["time"][:cutoff_idx] / 60.0  # convert to minutes

    # --- Panel 3: Step response showing RC relaxation ---
    # Apply current for 60s, then rest for 120s
    step_current = np.concatenate([
        np.full(60, I_1C),     # 60s discharge
        np.zeros(120),          # 120s rest
        np.full(60, I_1C),     # 60s discharge
        np.zeros(120),          # 120s rest
    ])
    model._model.reset(0.8)
    step_result = model._model.simulate(step_current, dt)

    # --- Panel 4: ECM vs static comparison ---
    model._model.reset(1.0)
    ecm_result = model._model.simulate(current_profile[:cutoff_idx], dt)
    soc_trace = ecm_result["soc"]
    static_v = model._model.static_voltage(soc_trace, I_1C)

    # --- Build figure ---
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "1C Discharge: Voltage vs Time",
            "1C Discharge: SOC vs Time",
            "Step Response: V_rc Polarization Dynamics",
            "ECM Dynamic vs F1a Static Voltage",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Panel 1: Voltage vs time
    fig.add_trace(
        go.Scatter(x=t_min, y=result["voltage"][:cutoff_idx],
                   name="V_terminal", line=dict(color="#d62728", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_min, y=model._model.ocv(result["soc"][:cutoff_idx]),
                   name="OCV", line=dict(color="#1f77b4", width=1, dash="dash")),
        row=1, col=1,
    )

    # Panel 2: SOC vs time
    fig.add_trace(
        go.Scatter(x=t_min, y=result["soc"][:cutoff_idx],
                   name="SOC", line=dict(color="#2ca02c", width=2), showlegend=False),
        row=1, col=2,
    )

    # Panel 3: V_rc dynamics
    step_time_min = step_result["time"] / 60.0
    fig.add_trace(
        go.Scatter(x=step_time_min, y=step_result["v_rc"],
                   name="V_rc", line=dict(color="#9467bd", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=step_time_min, y=np.concatenate([
            np.full(60, I_1C), np.zeros(120), np.full(60, I_1C), np.zeros(120)
        ]) / I_1C * max(abs(step_result["v_rc"])) * 0.5,
                   name="Current (scaled)", line=dict(color="#ff7f0e", width=1, dash="dot")),
        row=2, col=1,
    )

    # Panel 4: ECM vs static
    ecm_t_min = ecm_result["time"] / 60.0
    fig.add_trace(
        go.Scatter(x=ecm_t_min, y=ecm_result["voltage"],
                   name="ECM (F2a)", line=dict(color="#d62728", width=2)),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(x=ecm_t_min, y=static_v,
                   name="Static (F1a)", line=dict(color="#1f77b4", width=2, dash="dash")),
        row=2, col=2,
    )

    # Axis labels
    fig.update_xaxes(title_text="Time (min)", row=1, col=1)
    fig.update_xaxes(title_text="Time (min)", row=1, col=2)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_xaxes(title_text="Time (min)", row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="SOC", row=1, col=2)
    fig.update_yaxes(title_text="V_rc (V)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}: {info['description']}",
        height=800,
        template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
