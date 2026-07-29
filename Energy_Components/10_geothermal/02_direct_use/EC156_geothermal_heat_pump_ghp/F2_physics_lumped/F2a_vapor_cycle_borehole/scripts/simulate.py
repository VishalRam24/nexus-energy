"""
EC156 -- GHP F2a -- simulation report.
Generates an interactive Plotly HTML report (ground-source depletion + COP).
Plotly import is wrapped so its absence does not crash the build.
Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # Scenario: 30-day continuous heating draw, watch the borehole deplete.
    r = cm.predict({"T_supply_c": 45.0, "Q_demand_kW": 8.0,
                    "dt": 600.0, "duration_s": 30 * 86400})
    days = r["t"] / 86400.0

    # COP vs source temperature sweep (steady cycle).
    sweep_Tloop = list(range(0, 19))
    cop_sweep = [cm.operating_point(T, 45.0)["COP"] for T in sweep_Tloop]
    carnot_sweep = [cm.operating_point(T, 45.0)["COP_carnot"] for T in sweep_Tloop]

    print("=== EC156 GHP F2a simulation ===")
    print(f"30-day draw: T_loop {r['T_loop'][0]:.2f} -> {r['T_loop'][-1]:.2f} degC")
    print(f"            T_ground {r['T_ground'][0]:.2f} -> {r['T_ground'][-1]:.2f} degC")
    print(f"            COP {r['COP'][0]:.2f} -> {r['COP'][-1]:.2f}")
    print(f"            Q_cond {r['Q_cond_kW'][0]:.2f} -> {r['Q_cond_kW'][-1]:.2f} kW")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Ground-loop & ground temperature (30-day draw)",
                        "Heating COP over time",
                        "COP vs ground-loop temperature",
                        "Heat duties"),
    )
    fig.add_trace(go.Scatter(x=days, y=r["T_loop"], name="T_loop"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=r["T_ground"], name="T_ground"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=r["COP"], name="COP"), row=1, col=2)
    fig.add_trace(go.Scatter(x=sweep_Tloop, y=cop_sweep, name="COP"), row=2, col=1)
    fig.add_trace(go.Scatter(x=sweep_Tloop, y=carnot_sweep, name="Carnot ceiling",
                             line=dict(dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=days, y=r["Q_cond_kW"], name="Q_cond"), row=2, col=2)
    fig.add_trace(go.Scatter(x=days, y=r["Q_evap_kW"], name="Q_evap"), row=2, col=2)
    fig.add_trace(go.Scatter(x=days, y=r["W_elec_kW"], name="W_elec"), row=2, col=2)
    fig.update_layout(title="EC156 Geothermal Heat Pump (GHP) -- F2a Vapor-Cycle + Borehole",
                      height=800)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
