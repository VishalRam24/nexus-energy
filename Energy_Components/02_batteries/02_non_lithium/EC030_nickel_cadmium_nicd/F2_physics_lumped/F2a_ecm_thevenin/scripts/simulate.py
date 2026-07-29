"""
EC030 -- NiCd F2a Thevenin ECM -- simulation scenarios + optional Plotly report.
Plotly import is wrapped so absence does not crash. Not required to run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    m = ComponentModel()

    # Scenario 1: 1C discharge from full
    dis = m.predict({"current_A": 10.0, "soc0": 1.0, "dt": 2.0, "duration_s": 3000.0})
    # Scenario 2: pulse profile (discharge / rest / charge)
    def pulse(t):
        if t < 600.0:
            return 30.0      # 3C discharge
        elif t < 1200.0:
            return 0.0       # rest
        else:
            return -10.0     # 1C charge
    pul = m.predict({"current_A": pulse, "soc0": 0.9, "dt": 2.0, "duration_s": 1800.0})
    return dis, pul


def build_report(path="simulation_report.html"):
    dis, pul = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> skip gracefully
        print(f"[simulate] Plotly unavailable ({e}); scenarios computed, no HTML written.")
        print(f"  1C discharge: V {dis['voltage'][0]:.3f}->{dis['voltage'][-1]:.3f} V, "
              f"T {dis['temperature'][0]:.1f}->{dis['temperature'][-1]:.1f} K")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "1C Discharge: Voltage & SOC", "1C Discharge: Temperature",
        "Pulse Profile: Voltage", "Pulse Profile: Current & SOC"))
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["voltage"], name="V_term"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["ocv"], name="OCV"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["temperature"], name="T [K]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=pul["t"], y=pul["voltage"], name="V pulse"), row=2, col=1)
    fig.add_trace(go.Scatter(x=pul["t"], y=pul["current"], name="I [A]"), row=2, col=2)
    fig.add_trace(go.Scatter(x=pul["t"], y=pul["soc"], name="SOC", yaxis="y2"), row=2, col=2)
    fig.update_layout(title="EC030 NiCd -- F2a Thevenin ECM", height=800)
    out = os.path.join(os.path.dirname(__file__), "..", path)
    fig.write_html(out)
    print(f"[simulate] report written: {out}")
    return out


if __name__ == "__main__":
    build_report()
