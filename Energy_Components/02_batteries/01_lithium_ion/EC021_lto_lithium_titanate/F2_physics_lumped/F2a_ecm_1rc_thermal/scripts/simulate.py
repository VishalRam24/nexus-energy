"""
EC021 -- LTO Battery -- F2a Thevenin 1-RC ECM + Thermal
Optional Plotly simulation report. Plotly import is guarded so its absence
does not crash the module.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # Scenario: 1C discharge pulse train (load 5C, rest), then full 1C discharge
    def profile(t):
        if t < 120.0:
            return 14.5      # ~5C pulse
        elif t < 300.0:
            return 0.0       # rest -> RC relaxation visible
        else:
            return 2.9       # 1C steady discharge

    r = cm.predict({"current_A": profile, "soc0": 0.95, "T0": 298.15,
                    "dt": 1.0, "duration_s": 1800.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # noqa: BLE001
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  Final SOC={r['soc'][-1]:.3f}, V={r['voltage'][-1]:.3f} V, "
              f"T={r['temperature'][-1]:.2f} K")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Terminal voltage & OCV", "SOC", "RC polarization V_RC", "Cell temperature"))
    ocv = m.ocv(r["soc"])
    fig.add_trace(go.Scatter(x=r["t"], y=r["voltage"], name="V_terminal"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=ocv, name="OCV(SOC)"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["soc"], name="SOC"), 1, 2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["v_rc"], name="V_RC"), 2, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["temperature"], name="T"), 2, 2)
    fig.update_layout(title="EC021 LTO F2a -- 1-RC ECM + Thermal", height=720)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    run_report()
