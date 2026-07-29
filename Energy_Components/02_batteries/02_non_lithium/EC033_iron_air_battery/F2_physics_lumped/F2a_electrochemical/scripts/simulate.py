"""
EC033 -- Iron-Air Battery (Fe-Air) -- F2a Physics-Lumped Electrochemical
Optional Plotly simulation report. Plotly import is guarded so absence of the
package does not crash; the script still prints a text summary.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


def run_report(out_html="simulation_report.html"):
    cm = ComponentModel()
    m = cm._model

    # --- Charge/discharge polarization sweep (voltage gap) ---
    j_sweep = np.linspace(0.001, 0.055, 60)
    V_dis = np.array([m.cell_voltage(+j, 298.15) for j in j_sweep])
    V_chg = np.array([m.cell_voltage(-j, 298.15) for j in j_sweep])
    rte = np.array([m.round_trip_efficiency(j, 298.15) for j in j_sweep])
    ce = np.array([m.coulombic_efficiency_charge(j, 298.15)[0] for j in j_sweep])

    # --- A full charge then discharge transient ---
    r_chg = cm.predict({"current_density_A_cm2": -0.02, "duration_s": 7200.0,
                        "dt": 60.0, "soc_init": 0.2})
    r_dis = cm.predict({"current_density_A_cm2": 0.02, "duration_s": 7200.0,
                        "dt": 60.0, "soc_init": 0.9})

    print("=== EC033 Iron-Air F2a simulation summary ===")
    print(f"OCV @298K            : {m.ocv(298.15):.4f} V")
    print(f"V_discharge @0.02    : {m.cell_voltage(0.02, 298.15):.4f} V")
    print(f"V_charge   @0.02     : {m.cell_voltage(-0.02, 298.15):.4f} V")
    print(f"Voltaic eff @0.02    : {m.voltaic_efficiency(0.02, 298.15):.4f}")
    print(f"Coulombic eff @0.02  : {m.coulombic_efficiency_charge(0.02, 298.15)[0]:.4f}")
    print(f"Round-trip eff @0.02 : {m.round_trip_efficiency(0.02, 298.15):.4f}")
    print(f"Charge end SOC/T     : {r_chg['soc'][-1]:.3f} / {r_chg['temperature'][-1]:.2f} K")
    print(f"Discharge end SOC/T  : {r_dis['soc'][-1]:.3f} / {r_dis['temperature'][-1]:.2f} K")

    if not _HAVE_PLOTLY:
        print("(plotly not installed -- skipping HTML report)")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Charge/discharge polarization (voltage gap)",
            "Efficiencies vs current density",
            "Transient: temperature",
            "Transient: state of charge",
        ),
    )
    fig.add_trace(go.Scatter(x=j_sweep, y=V_chg, name="V charge"), row=1, col=1)
    fig.add_trace(go.Scatter(x=j_sweep, y=V_dis, name="V discharge"), row=1, col=1)
    fig.add_trace(go.Scatter(x=j_sweep, y=rte, name="round-trip eff"), row=1, col=2)
    fig.add_trace(go.Scatter(x=j_sweep, y=ce, name="coulombic eff"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r_dis["t"] / 3600.0, y=r_dis["temperature"],
                             name="T (discharge)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r_chg["t"] / 3600.0, y=r_chg["soc"],
                             name="SOC (charge)"), row=2, col=2)
    fig.add_trace(go.Scatter(x=r_dis["t"] / 3600.0, y=r_dis["soc"],
                             name="SOC (discharge)"), row=2, col=2)
    fig.update_layout(height=800, title_text="EC033 Iron-Air F2a Physics-Lumped Report")

    out_path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(out_path)
    print(f"Report written to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    run_report()
