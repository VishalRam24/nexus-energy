"""EC036 -- VRFB -- F2a Stack Model -- Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Discharge at 50A (SOC & Voltage)",
            "Power Breakdown (Stack, Pump, Net)",
            "Charge at -50A (SOC & Voltage)",
            "Flow Rate Sensitivity on Voltage",
            "Efficiency vs Flow Rate",
            "Current Step Response",
        ],
        vertical_spacing=0.10,
    )

    # Panel 1: Discharge
    r1 = model.predict({
        "current_A": 50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 3600.0, "soc_init": 0.9,
    })
    fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["voltage"], name="Voltage (V)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["soc"], name="SOC", line=dict(dash="dash")), row=1, col=1)

    # Panel 2: Power
    fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["power_stack"], name="P_stack (W)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["power_pump"], name="P_pump (W)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["net_power"], name="P_net (W)"), row=1, col=2)

    # Panel 3: Charge
    r2 = model.predict({
        "current_A": -50.0, "flow_rate_L_min": 10.0,
        "dt": 10.0, "duration_s": 3600.0, "soc_init": 0.2,
    })
    fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["voltage"], name="V (charge)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["soc"], name="SOC (charge)", line=dict(dash="dash")), row=2, col=1)

    # Panel 4: Flow rate sensitivity
    for Q in [3, 5, 10, 20]:
        r = model.predict({
            "current_A": 60.0, "flow_rate_L_min": Q,
            "dt": 10.0, "duration_s": 1800.0, "soc_init": 0.8,
        })
        fig.add_trace(go.Scatter(x=r["t"]/60, y=r["voltage"], name=f"Q={Q} L/min"), row=2, col=2)

    # Panel 5: Efficiency vs flow rate
    Q_range = np.linspace(2, 25, 30)
    eff_arr = []
    for Q in Q_range:
        r = model.predict({
            "current_A": 50.0, "flow_rate_L_min": float(Q),
            "dt": 60.0, "duration_s": 60.0, "soc_init": 0.5,
        })
        eff_arr.append(r["efficiency"][0])
    fig.add_trace(go.Scatter(x=Q_range, y=eff_arr, name="Efficiency", mode="lines+markers"), row=3, col=1)

    # Panel 6: Current step
    def I_step(t):
        return 30.0 if t < 300 else 80.0

    r3 = model.predict({
        "current_A": I_step, "flow_rate_L_min": 10.0,
        "dt": 1.0, "duration_s": 600.0, "soc_init": 0.7,
    })
    fig.add_trace(go.Scatter(x=r3["t"], y=r3["voltage"], name="V (step)"), row=3, col=2)

    for r in range(1, 4):
        for c in [1, 2]:
            fig.update_xaxes(title_text="Time (min)" if r < 3 else "Flow (L/min)" if c == 1 else "Time (s)", row=r, col=c)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Stack Model",
        height=1000, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
