"""
EC042 -- Pseudocapacitor -- F2a RC-Faradaic
Optional Plotly report. Plotly import is guarded so absence never crashes.
Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_HTML = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # Scenario A: galvanostatic discharge at several rates
    rates = [10.0, 30.0, 60.0, 90.0]
    discharges = {}
    for I in rates:
        discharges[I] = cm.predict({"current_A": I, "v_cap0_V": 1.0,
                                    "dt": 0.02, "duration_s": 8.0})

    # Scenario B: differential capacitance vs voltage
    vv = np.linspace(0, 1.0, 200)
    C_v = m.differential_capacitance(vv, m.T_ref)
    C_far = m.faradaic_capacitance(vv, m.T_ref)

    # Scenario C: efficiency vs rate
    eff_rates = [5, 10, 20, 40, 60, 80]
    effs = [m.round_trip_efficiency(I, 0.2, m.T_ref) for I in eff_rates]

    # Scenario D: self-discharge
    sd = cm.predict({"current_A": 0.0, "v_cap0_V": 1.0, "dt": 2.0, "duration_s": 600.0})

    return discharges, (vv, C_v, C_far), (eff_rates, effs), sd


def main():
    discharges, capv, effv, sd = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        # still print a numeric summary
        vv, C_v, C_far = capv
        print(f"  peak C_diff = {C_v.max():.1f} F at V={vv[np.argmax(C_v)]:.2f}")
        eff_rates, effs = effv
        print("  efficiency vs rate:", [f"{r}A:{e:.3f}" for r, e in zip(eff_rates, effs)])
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Galvanostatic discharge (V_term)",
        "Differential capacitance C(V)",
        "Round-trip efficiency vs rate",
        "Open-circuit self-discharge"))

    for I, r in discharges.items():
        fig.add_trace(go.Scatter(x=r["t"], y=r["terminal_voltage"],
                                 name=f"{I:.0f} A"), row=1, col=1)
    vv, C_v, C_far = capv
    fig.add_trace(go.Scatter(x=vv, y=C_v, name="C_total"), row=1, col=2)
    fig.add_trace(go.Scatter(x=vv, y=C_far, name="C_faradaic"), row=1, col=2)
    eff_rates, effs = effv
    fig.add_trace(go.Scatter(x=eff_rates, y=effs, name="eta", mode="lines+markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=sd["t"], y=sd["v_cap"], name="V_cap"), row=2, col=2)

    fig.update_layout(title="EC042 Pseudocapacitor F2a RC-Faradaic", height=800)
    fig.write_html(_HTML)
    print(f"[simulate] wrote {_HTML}")


if __name__ == "__main__":
    main()
