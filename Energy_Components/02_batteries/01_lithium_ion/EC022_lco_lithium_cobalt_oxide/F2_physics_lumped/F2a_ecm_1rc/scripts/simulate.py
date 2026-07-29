"""
EC022 -- LCO Battery -- F2a Thevenin 1-RC ECM
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # Scenario: 1C discharge from full, then a charge pulse, with self-heating.
    def current_profile(t):
        if t < 2400.0:
            return 2.6          # 1C discharge
        elif t < 3000.0:
            return 0.0          # rest
        else:
            return -2.6         # 1C charge

    r = cm.predict({
        "current_A": current_profile,
        "soc0": 0.95,
        "T_K": 298.15,
        "dt": 5.0,
        "duration_s": 3600.0,
    })

    print(f"Sim points: {len(r['t'])}")
    print(f"SOC: {r['soc'][0]:.3f} -> {r['soc'][-1]:.3f}")
    print(f"V:   {r['voltage'][0]:.3f} -> {r['voltage'][-1]:.3f} V")
    print(f"T:   {r['temperature'][0]:.2f} -> {r['temperature'][-1]:.2f} K "
          f"(peak {r['temperature'].max():.2f} K)")
    print(f"Peak Q_gen: {r['heat_gen'].max():.3f} W")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # plotly absent -> skip plot, do not crash
        print(f"[simulate] Plotly unavailable ({exc}); skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Terminal Voltage & OCV-SOC", "SOC",
                        "Cell Temperature", "Heat Generation"),
    )
    fig.add_trace(go.Scatter(x=r["t"], y=r["voltage"], name="V_term"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["soc"], name="SOC"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["temperature"], name="T [K]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["heat_gen"], name="Q_gen [W]"), row=2, col=2)
    fig.update_layout(title_text="EC022 LCO F2a Thevenin 1-RC ECM", showlegend=True)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
