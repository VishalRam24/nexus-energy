"""EC124 — LAES — F1b Thermal — Simulation Scenarios"""
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

    # Scenario 1: BOR vs T_amb
    T_amb_arr = np.linspace(243.15, 333.15, 50)
    bors = [float(m.boil_off_rate_per_day(T)) * 100 for T in T_amb_arr]

    # Scenario 2: SOC decay over 1 week at different T_amb
    times = np.linspace(0, 168, 200)   # hours
    T_scenarios = [253.15, 273.15, 298.15, 313.15]
    soc_decay = {}
    for T in T_scenarios:
        soc_decay[T] = [float(m.soc_after_standby(0.9, t, T)) for t in times]

    # Scenario 3: RTE, cold recycle, discharge work vs T_amb
    rtes = [float(m.round_trip_efficiency(T)) for T in T_amb_arr]
    eps = [float(m.cold_recycle_effectiveness(T)) for T in T_amb_arr]
    w_disch = [float(m.effective_discharge_work(T)) for T in T_amb_arr]

    # Scenario 4: Power output vs m_dot at two T_amb
    m_dot_arr = np.linspace(0, 150, 50)
    P_cold = [float(m.discharge_power(md, 253.15)) for md in m_dot_arr]
    P_warm = [float(m.discharge_power(md, 313.15)) for md in m_dot_arr]

    print("=== EC124 F1b LAES Thermal — Simulation Report ===\n")
    print(f"BOR at T_amb_ref: {m.bor_ref * 100:.2f} %/day")
    print("BOR at select T:")
    for T in [253.15, 273.15, 298.15, 313.15, 323.15]:
        print(f"  {T - 273.15:+.0f}°C: {float(m.boil_off_rate_per_day(T)) * 100:.4f} %/day")
    print("\nRTE at select T:")
    for T in [253.15, 273.15, 298.15, 313.15]:
        print(f"  {T - 273.15:+.0f}°C: RTE={float(m.round_trip_efficiency(T)):.4f}  "
              f"eps={float(m.cold_recycle_effectiveness(T)):.3f}  "
              f"w_disch={float(m.effective_discharge_work(T)):.4f} kWh/kg")

    if not HAS_PLOTLY:
        print("\nPlotly not available — skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=["Boil-Off Rate vs Ambient Temperature",
                                          "SOC Decay During Storage",
                                          "RTE & Cold Recycle Effectiveness vs T_amb",
                                          "Discharge Power vs Mass Flow"])

    fig.add_trace(go.Scatter(x=T_amb_arr - 273.15, y=bors,
                              name="BOR [%/day]", line=dict(color="red")), row=1, col=1)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, T in enumerate(T_scenarios):
        label = f"T_amb={T - 273.15:.0f}°C"
        fig.add_trace(go.Scatter(x=times, y=soc_decay[T], name=label,
                                  line=dict(color=colors[i])), row=1, col=2)

    fig.add_trace(go.Scatter(x=T_amb_arr - 273.15, y=rtes, name="RTE",
                              line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_amb_arr - 273.15, y=eps, name="Cold recycle eff",
                              line=dict(color="cyan", dash="dash")), row=2, col=1)

    fig.add_trace(go.Scatter(x=m_dot_arr, y=P_cold, name="T_amb=-20°C",
                              line=dict(color="steelblue")), row=2, col=2)
    fig.add_trace(go.Scatter(x=m_dot_arr, y=P_warm, name="T_amb=+40°C",
                              line=dict(color="tomato")), row=2, col=2)

    fig.update_xaxes(title_text="T_amb [°C]", row=1, col=1)
    fig.update_xaxes(title_text="Time [h]", row=1, col=2)
    fig.update_xaxes(title_text="T_amb [°C]", row=2, col=1)
    fig.update_xaxes(title_text="Mass flow [kg/s]", row=2, col=2)
    fig.update_yaxes(title_text="BOR [%/day]", row=1, col=1)
    fig.update_yaxes(title_text="SOC [-]", row=1, col=2)
    fig.update_yaxes(title_text="RTE / eff [-]", row=2, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=2, col=2)

    fig.update_layout(height=700, width=1100,
                       title_text="EC124 F1b — LAES Thermal Model")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    run_scenarios()
