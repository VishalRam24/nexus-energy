"""
EC029 -- NiMH Battery -- F2a Thevenin 2-RC Electrothermal
Optional Plotly simulation report. Plotly import is wrapped so absence
does not crash. Generates simulation_report.html one level up.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()

    # Scenario A: 2 A discharge from 90% SOC
    dis = cm.predict({"current_A": 2.0, "soc0": 0.9, "dt": 5.0, "duration_s": 2400.0})
    # Scenario B: 4 A overcharge from 93% SOC (oxygen recombination exotherm)
    oc = cm.predict({"current_A": -4.0, "soc0": 0.93, "dt": 5.0, "duration_s": 1200.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); printing summary only.")
        print(f"  Discharge: SOC {dis['soc'][0]:.3f}->{dis['soc'][-1]:.3f}, "
              f"V_end={dis['voltage'][-1]:.3f} V, dT={dis['temperature'][-1]-dis['temperature'][0]:.2f} K")
        print(f"  Overcharge: SOC {oc['soc'][0]:.3f}->{oc['soc'][-1]:.3f}, "
              f"dT={oc['temperature'][-1]-oc['temperature'][0]:.2f} K, "
              f"f_oc_end={oc['overcharge_fraction'][-1]:.3f}")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Discharge: voltage & SOC", "Discharge: temperature",
                        "Overcharge: SOC saturates", "Overcharge: heat terms (W)"),
        specs=[[{"secondary_y": True}, {}], [{}, {}]],
    )

    fig.add_trace(go.Scatter(x=dis["t"], y=dis["voltage"], name="V_term [V]"), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["soc"], name="SOC"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["temperature"], name="T [K]"), row=1, col=2)

    fig.add_trace(go.Scatter(x=oc["t"], y=oc["soc"], name="SOC (overcharge)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=oc["t"], y=oc["overcharge_fraction"], name="f_oc"), row=2, col=1)
    fig.add_trace(go.Scatter(x=oc["t"], y=oc["heat"]["recombination"], name="Q_recomb"), row=2, col=2)
    fig.add_trace(go.Scatter(x=oc["t"], y=oc["heat"]["irreversible"], name="Q_irrev"), row=2, col=2)

    fig.update_layout(title="EC029 NiMH F2a -- Thevenin 2-RC Electrothermal", height=800)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Report written to {out}")


if __name__ == "__main__":
    main()
