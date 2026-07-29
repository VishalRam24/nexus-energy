"""EC123 — CAES — F1b Thermal — Simulation Scenarios"""
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

    # Scenario 1: Cavern thermal drift
    tau = m.tau_cav
    t_arr = np.linspace(0, 5 * tau, 200)
    T0 = m.T_cav_nominal + 20.0
    T_drift = [float(model.predict({"mode": "thermal", "soc": 0.5,
                                     "T_cav_0_K": T0, "t_idle_s": float(t)})["T_cav_K"])
               for t in t_arr]

    # Scenario 2: Compression work vs T_amb
    T_amb_arr = np.linspace(243.15, 323.15, 50)
    w_comp_arr = [float(m.specific_compression_work(T)) for T in T_amb_arr]
    rte_arr = [float(m.round_trip_efficiency(T)) for T in T_amb_arr]

    # Scenario 3: Cavern pressure vs SOC at two T_cav
    soc_arr = np.linspace(0, 1, 50)
    p_cold = [float(model.predict({"soc": s, "m_dot_air": 0.0,
                                    "T_cav_K": 295.0})["cavern_pressure_Pa"]) / 1e6
              for s in soc_arr]
    p_hot = [float(model.predict({"soc": s, "m_dot_air": 0.0,
                                   "T_cav_K": 325.0})["cavern_pressure_Pa"]) / 1e6
             for s in soc_arr]

    # Scenario 4: Energy capacity vs T_cav
    T_cav_range = np.linspace(290.0, 335.0, 30)
    E_cap = [m.energy_capacity_kwh(T) / 1000.0 for T in T_cav_range]   # MWh

    print("=== EC123 F1b CAES Thermal — Simulation Report ===\n")
    print(f"Cavern tau = {tau / 3600:.1f} h,  T_rock = {m.T_rock:.1f} K")
    print(f"T_cav after 24h idle (from {T0:.1f} K): {T_drift[np.argmin(np.abs(t_arr - 86400))]:.2f} K")
    print("\nCompression work vs T_amb:")
    for T in [253.15, 273.15, 288.15, 303.15, 313.15]:
        print(f"  {T - 273.15:+.0f}°C: {float(m.specific_compression_work(T)):.1f} kJ/kg, "
              f"RTE={float(m.round_trip_efficiency(T)):.4f}")

    if not HAS_PLOTLY:
        print("\nPlotly not available — skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=2,
                         subplot_titles=["Cavern Temperature Drift to Rock",
                                          "Compression Work & RTE vs T_amb",
                                          "Cavern Pressure vs SOC at Two Temperatures",
                                          "Usable Energy Capacity vs Cavern Temperature"])

    fig.add_trace(go.Scatter(x=t_arr / 3600, y=[T - 273.15 for T in T_drift],
                              name="T_cav", line=dict(color="orange")), row=1, col=1)
    fig.add_hline(y=m.T_rock - 273.15, line_dash="dash", line_color="brown",
                  annotation_text="T_rock", row=1, col=1)

    fig.add_trace(go.Scatter(x=T_amb_arr - 273.15, y=w_comp_arr,
                              name="w_comp [kJ/kg]", line=dict(color="red")), row=1, col=2)
    fig.add_trace(go.Scatter(x=T_amb_arr - 273.15, y=rte_arr,
                              name="RTE", line=dict(color="blue"),
                              yaxis="y2"), row=1, col=2)

    fig.add_trace(go.Scatter(x=soc_arr, y=p_cold, name="T_cav=22°C",
                              line=dict(color="steelblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=soc_arr, y=p_hot, name="T_cav=52°C",
                              line=dict(color="tomato")), row=2, col=1)

    fig.add_trace(go.Scatter(x=T_cav_range - 273.15, y=E_cap,
                              name="E_cap [MWh]", line=dict(color="green")), row=2, col=2)

    fig.update_xaxes(title_text="Time [h]", row=1, col=1)
    fig.update_xaxes(title_text="T_amb [°C]", row=1, col=2)
    fig.update_xaxes(title_text="SOC [-]", row=2, col=1)
    fig.update_xaxes(title_text="T_cav [°C]", row=2, col=2)
    fig.update_yaxes(title_text="T_cav [°C]", row=1, col=1)
    fig.update_yaxes(title_text="kJ/kg or RTE", row=1, col=2)
    fig.update_yaxes(title_text="Pressure [MPa]", row=2, col=1)
    fig.update_yaxes(title_text="Energy [MWh]", row=2, col=2)

    fig.update_layout(height=700, width=1100,
                       title_text="EC123 F1b — CAES Thermal Model")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    run_scenarios()
