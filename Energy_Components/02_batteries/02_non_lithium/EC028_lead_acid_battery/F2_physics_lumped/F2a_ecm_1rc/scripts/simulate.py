"""
EC028 -- Lead-Acid Battery -- F2a ECM 1-RC -- Simulation & HTML Report Generator
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
    v_min = model.params["cell"]["voltage_min"]["value"]

    # Lead-acid at 1C: use C/10 rate for more realistic discharge
    I_discharge = Q / 10.0  # C/10 rate for lead-acid
    dt = 1.0
    n_steps = int(40000)  # 10+ hours for C/10
    current_profile = np.full(n_steps, I_discharge)
    result = model.predict({"current": current_profile, "dt": dt, "soc_init": 1.0})

    valid = result["voltage"] > v_min + 0.001
    cutoff_idx = np.argmin(valid) if not np.all(valid) else n_steps
    t_hr = result["time"][:cutoff_idx] / 3600.0

    # Step response
    step_current = np.concatenate([
        np.full(120, I_discharge), np.zeros(300),
        np.full(120, I_discharge), np.zeros(300),
    ])
    model._model.reset(0.8)
    step_result = model._model.simulate(step_current, dt)

    # ECM vs static
    model._model.reset(1.0)
    ecm_result = model._model.simulate(current_profile[:cutoff_idx], dt)
    static_v = model._model.static_voltage(ecm_result["soc"], I_discharge)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"C/10 Discharge ({I_discharge:.1f}A): Voltage vs Time",
            f"C/10 Discharge: SOC vs Time",
            "Step Response: V_rc Polarization Dynamics",
            "ECM Dynamic vs F1a Static Voltage",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(x=t_hr, y=result["voltage"][:cutoff_idx],
                  name="V_terminal", line=dict(color="#d62728", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_hr, y=model._model.ocv(result["soc"][:cutoff_idx]),
                  name="OCV", line=dict(color="#1f77b4", width=1, dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=t_hr, y=result["soc"][:cutoff_idx],
                  name="SOC", line=dict(color="#2ca02c", width=2), showlegend=False), row=1, col=2)

    step_t = step_result["time"] / 60.0
    fig.add_trace(go.Scatter(x=step_t, y=step_result["v_rc"],
                  name="V_rc", line=dict(color="#9467bd", width=2)), row=2, col=1)

    ecm_t = ecm_result["time"] / 3600.0
    fig.add_trace(go.Scatter(x=ecm_t, y=ecm_result["voltage"],
                  name="ECM (F2a)", line=dict(color="#d62728", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=ecm_t, y=static_v,
                  name="Static (F1a)", line=dict(color="#1f77b4", width=2, dash="dash")), row=2, col=2)

    fig.update_xaxes(title_text="Time (h)", row=1, col=1)
    fig.update_xaxes(title_text="Time (h)", row=1, col=2)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="SOC", row=1, col=2)
    fig.update_yaxes(title_text="V_rc (V)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}: {info['description']}",
        height=800, template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
