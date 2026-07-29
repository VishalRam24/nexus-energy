"""
EC027 -- Solid-State Lithium Battery -- F2a Thevenin ECM
Optional Plotly simulation report. Plotly import is guarded so its absence
does not crash. Generates simulation_report.html one level up.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    scenarios = {}

    # 1) Constant 1C discharge at three temperatures (cold penalty)
    for label, T in [("-10C", 263.15), ("25C", 298.15), ("60C", 333.15)]:
        scenarios[f"discharge_{label}"] = cm.predict(
            {"current_A": 4.0, "soc0": 0.95, "T0": T, "T_amb": T,
             "dt": 5.0, "duration_s": 3000.0})

    # 2) Pulse profile (RC relaxation) at 25C
    def pulse(t):
        return 12.0 if (t % 200) < 100 else 0.0
    scenarios["pulse_25C"] = cm.predict(
        {"current_A": pulse, "soc0": 0.8, "T0": 298.15, "dt": 1.0, "duration_s": 1000.0})

    return cm, scenarios


def main():
    cm, scenarios = run_scenarios()
    here = os.path.dirname(__file__)
    out_html = os.path.join(here, "..", "simulation_report.html")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> print summary only
        print(f"[simulate] Plotly unavailable ({e}); printing text summary.")
        for name, r in scenarios.items():
            print(f"  {name}: V_end={r['voltage'][-1]:.3f} V, "
                  f"SOC_end={r['soc'][-1]:.3f}, T_end={r['temperature'][-1]:.2f} K, "
                  f"R0_end={r['R0'][-1]*1000:.2f} mOhm, eff={r['coulombic_efficiency']:.4f}")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Discharge voltage vs SOC (T sweep)", "Cell temperature",
        "Series SE resistance R0(T)", "Pulse terminal voltage"))

    for label in ["-10C", "25C", "60C"]:
        r = scenarios[f"discharge_{label}"]
        fig.add_trace(go.Scatter(x=r["soc"], y=r["voltage"], name=f"V {label}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["t"], y=r["temperature"], name=f"T {label}"), row=1, col=2)
        fig.add_trace(go.Scatter(x=r["t"], y=np.array(r["R0"]) * 1000, name=f"R0 {label} (mOhm)"), row=2, col=1)

    rp = scenarios["pulse_25C"]
    fig.add_trace(go.Scatter(x=rp["t"], y=rp["voltage"], name="V pulse"), row=2, col=2)

    fig.update_layout(title="EC027 Solid-State Li -- F2a Thevenin ECM", height=800)
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")


if __name__ == "__main__":
    main()
