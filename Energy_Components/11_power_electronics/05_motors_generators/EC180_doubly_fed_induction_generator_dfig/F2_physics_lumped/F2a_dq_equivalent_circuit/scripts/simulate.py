"""
EC180 -- DFIG F2a -- simulation scenarios + optional Plotly HTML report.

Scenarios:
  A) Slip sweep at fixed grid power -> shows P_rotor = -s*P_stator
     (sub-synchronous draws / super-synchronous delivers rotor power).
  B) Power-control step response at fixed super-synchronous slip.

Run:  python3 scripts/simulate.py
Plotly is optional; absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def scenario_slip_sweep():
    cm = ComponentModel()
    slips = np.linspace(-0.30, 0.30, 13)
    Ps, Pr, Qs = [], [], []
    for s in slips:
        r = cm.predict({"mode": "power_control", "P_stator_ref_W": -1.5e6,
                        "Q_stator_ref_VAr": 0.0, "slip": float(s),
                        "duration_s": 1.0, "dt": 2e-4})
        n = max(1, int(len(r["t"]) * 0.2))
        Ps.append(np.mean(r["P_stator"][-n:]) / 1e6)
        Pr.append(np.mean(r["P_rotor"][-n:]) / 1e6)
        Qs.append(np.mean(r["Q_stator"][-n:]) / 1e6)
    return slips, np.array(Ps), np.array(Pr), np.array(Qs)


def scenario_step():
    cm = ComponentModel()

    def Pref(t):
        return -0.6e6 if t < 0.3 else -1.6e6

    r = cm.predict({"mode": "power_control", "P_stator_ref_W": Pref,
                    "Q_stator_ref_VAr": 0.0, "slip": -0.2,
                    "duration_s": 0.6, "dt": 1e-4})
    return r


def main():
    slips, Ps, Pr, Qs = scenario_slip_sweep()
    step = scenario_step()

    print("DFIG slip sweep (P_stator_ref = -1.5 MW, super/sub-synchronous):")
    for s, ps, pr in zip(slips, Ps, Pr):
        tag = "super-sync" if s < 0 else ("sync" if abs(s) < 1e-9 else "sub-sync")
        print(f"  s={s:+.2f} ({tag:10s})  P_stator={ps:+.3f} MW  P_rotor={pr:+.3f} MW")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Slip-power relation: P_rotor = -s·P_stator",
                            "Power-control step response (slip = -0.2)"))

        fig.add_trace(go.Scatter(x=slips, y=Ps, name="P_stator [MW]",
                                 mode="lines+markers"), row=1, col=1)
        fig.add_trace(go.Scatter(x=slips, y=Pr, name="P_rotor [MW]",
                                 mode="lines+markers"), row=1, col=1)
        fig.add_trace(go.Scatter(x=slips, y=-slips * Ps,
                                 name="-s·P_stator (theory)",
                                 line=dict(dash="dash")), row=1, col=1)
        fig.update_xaxes(title_text="slip s [-]", row=1, col=1)
        fig.update_yaxes(title_text="Power [MW]", row=1, col=1)

        fig.add_trace(go.Scatter(x=step["t"], y=step["P_stator"] / 1e6,
                                 name="P_stator [MW]"), row=2, col=1)
        fig.add_trace(go.Scatter(x=step["t"], y=step["Q_stator"] / 1e6,
                                 name="Q_stator [MVAr]"), row=2, col=1)
        fig.update_xaxes(title_text="time [s]", row=2, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)

        fig.update_layout(title_text="EC180 DFIG F2a -- dq-Frame Dynamic Model",
                          height=800)
        out = os.path.join(os.path.dirname(__file__), "..",
                           "simulation_report.html")
        fig.write_html(out)
        print(f"\nWrote report: {os.path.abspath(out)}")
    except Exception as e:
        print(f"\n[plotly unavailable, skipping HTML report: {e}]")


if __name__ == "__main__":
    main()
