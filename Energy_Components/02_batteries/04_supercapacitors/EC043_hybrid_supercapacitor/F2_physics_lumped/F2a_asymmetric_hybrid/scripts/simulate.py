"""
EC043 -- Hybrid Supercapacitor (Lithium-Ion Capacitor) -- F2a Asymmetric Hybrid
Simulation scenarios + optional interactive Plotly HTML report.

Run: python3 scripts/simulate.py
Plotly is optional; absence does not crash this script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def scenario_galvanostatic_discharge(cm):
    """Constant-current discharge from near-full -> shows sloping 2.2-3.8 V curve."""
    Q_max = cm._model.Q_max
    return cm.predict({"current_A": 60.0, "q0_C": 0.98 * Q_max,
                       "T0_K": 298.15, "dt": 0.5, "duration_s": 65.0})


def scenario_pulse_profile(cm):
    """Charge/discharge pulse train (regen-braking-like duty)."""
    Q_max = cm._model.Q_max

    def pulse(t):
        phase = int(t // 5) % 2
        return 250.0 if phase == 0 else -250.0

    return cm._model.simulate(pulse, 0.5 * Q_max, 298.15, 0.1, 40.0)


def build_report(path="simulation_report.html"):
    cm = ComponentModel()
    dis = scenario_galvanostatic_discharge(cm)
    pul = scenario_pulse_profile(cm)

    print("Galvanostatic discharge: "
          f"V {dis['v_terminal'][0]:.2f} -> {dis['v_terminal'][-1]:.2f} V, "
          f"SOC {dis['soc'][0]:.2f} -> {dis['soc'][-1]:.2f}, "
          f"E {dis['energy_J'][0]:.0f} -> {dis['energy_J'][-1]:.0f} J")
    print("Pulse profile: T "
          f"{pul['temperature'][0]:.3f} -> {pul['temperature'][-1]:.3f} K, "
          f"peak |I*V| power {np.max(np.abs(pul['power'])):.0f} W")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Discharge V_terminal & OCV vs SOC",
                        "Discharge temperature",
                        "Pulse current power",
                        "Pulse temperature"),
    )
    fig.add_trace(go.Scatter(x=dis["soc"], y=dis["v_terminal"], name="V_term"), 1, 1)
    fig.add_trace(go.Scatter(x=dis["soc"], y=dis["v_oc"], name="OCV"), 1, 1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["temperature"], name="T discharge"), 1, 2)
    fig.add_trace(go.Scatter(x=pul["t"], y=pul["power"], name="P pulse"), 2, 1)
    fig.add_trace(go.Scatter(x=pul["t"], y=pul["temperature"], name="T pulse"), 2, 2)
    fig.update_layout(title="EC043 Hybrid Supercapacitor (LIC) -- F2a Asymmetric Hybrid",
                      height=750)
    out = os.path.join(os.path.dirname(__file__), "..", path)
    fig.write_html(out)
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    build_report()
