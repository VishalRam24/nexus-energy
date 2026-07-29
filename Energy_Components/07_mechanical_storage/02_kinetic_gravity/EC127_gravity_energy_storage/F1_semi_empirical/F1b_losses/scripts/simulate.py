"""EC127 — Gravity Energy Storage — F1b Losses — Simulation Scenarios"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def run_scenarios():
    model = ComponentModel()
    m = model._model

    # Scenario 1: Loss breakdown vs velocity
    v_arr = np.linspace(0.01, 3.0, 100)
    friction = []
    drag = []
    bearing = []
    total = []
    for v in v_arr:
        r = model.predict({"velocity_mps": v, "mode": "losses"})
        friction.append(float(r["friction_kw"]))
        drag.append(float(r["drag_kw"]))
        bearing.append(float(r["bearing_kw"]))
        total.append(float(r["total_mech_loss_kw"]))

    # Scenario 2: RTE vs velocity
    rtes = [float(model.predict({"soc": 0.5, "velocity_mps": v})["round_trip_efficiency"])
            for v in v_arr]

    # Scenario 3: Motor/generator efficiency vs PLF
    plf_arr = np.linspace(0.05, 1.0, 50)
    eta_motor = [float(m.motor_efficiency(p)) for p in plf_arr]
    eta_gen = [float(m.generator_efficiency(p)) for p in plf_arr]
    eta_drive = [float(m.drive_efficiency(p)) for p in plf_arr]

    # Scenario 4: Discharge and charge power vs velocity
    P_charge = [float(model.predict({"soc": 0.5, "velocity_mps": v, "mode": "charge"})["power_kw"])
                for v in v_arr]
    P_discharge = [float(model.predict({"soc": 0.5, "velocity_mps": v, "mode": "discharge"})["power_kw"])
                   for v in v_arr]
    P_grav = [m.m * m.g * v / 1000.0 for v in v_arr]   # ideal gravitational power

    print("=== EC127 F1b Gravity Storage Losses — Simulation Report ===\n")
    print("Loss breakdown at v=2 m/s:")
    r = model.predict({"velocity_mps": 2.0, "mode": "losses"})
    print(f"  Friction: {float(r['friction_kw']):.1f} kW")
    print(f"  Drag:     {float(r['drag_kw']):.4f} kW")
    print(f"  Bearing:  {float(r['bearing_kw']):.1f} kW")
    print(f"  Total:    {float(r['total_mech_loss_kw']):.1f} kW")

    print("\nRTE vs velocity:")
    for v in [0.5, 1.0, 2.0, 3.0]:
        print(f"  v={v:.1f} m/s: RTE={float(model.predict({'soc': 0.5, 'velocity_mps': v})['round_trip_efficiency']):.4f}")

    print("\nEfficiency vs PLF:")
    for plf in [0.1, 0.25, 0.5, 0.75, 1.0]:
        print(f"  PLF={plf:.2f}: eta_motor={float(m.motor_efficiency(plf)):.4f}  "
              f"eta_gen={float(m.generator_efficiency(plf)):.4f}")

    if not HAS_PLOTLY:
        print("\nPlotly not available — skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=["Mechanical Loss Breakdown vs Velocity",
                                          "Round-Trip Efficiency vs Velocity",
                                          "Motor/Generator/Drive Efficiency vs PLF",
                                          "Charge & Discharge Power vs Velocity"])

    fig.add_trace(go.Scatter(x=v_arr, y=friction, name="Friction", fill="tozeroy",
                              line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=v_arr, y=drag, name="Drag",
                              line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=v_arr, y=bearing, name="Bearing",
                              line=dict(color="purple")), row=1, col=1)
    fig.add_trace(go.Scatter(x=v_arr, y=total, name="Total",
                              line=dict(color="black", dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=v_arr, y=rtes, name="RTE", line=dict(color="green")), row=1, col=2)

    fig.add_trace(go.Scatter(x=plf_arr, y=eta_motor, name="Motor", line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=plf_arr, y=eta_gen, name="Generator", line=dict(color="teal")), row=2, col=1)
    fig.add_trace(go.Scatter(x=plf_arr, y=eta_drive, name="Drive", line=dict(color="olive", dash="dash")), row=2, col=1)

    fig.add_trace(go.Scatter(x=v_arr, y=P_charge, name="P_charge (in)",
                              line=dict(color="red")), row=2, col=2)
    fig.add_trace(go.Scatter(x=v_arr, y=P_discharge, name="P_discharge (out)",
                              line=dict(color="blue")), row=2, col=2)
    fig.add_trace(go.Scatter(x=v_arr, y=P_grav, name="P_grav (ideal)",
                              line=dict(color="gray", dash="dot")), row=2, col=2)

    fig.update_xaxes(title_text="Velocity [m/s]", row=1, col=1)
    fig.update_xaxes(title_text="Velocity [m/s]", row=1, col=2)
    fig.update_xaxes(title_text="PLF [-]", row=2, col=1)
    fig.update_xaxes(title_text="Velocity [m/s]", row=2, col=2)
    fig.update_yaxes(title_text="Loss [kW]", row=1, col=1)
    fig.update_yaxes(title_text="RTE [-]", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=2, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=2, col=2)

    fig.update_layout(height=700, width=1100,
                       title_text="EC127 F1b — Gravity Storage Losses Model")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    run_scenarios()
