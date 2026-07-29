"""
EC214 -- MVC F2a -- simulation scenarios + optional Plotly HTML report.
Plotly is imported lazily; its absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # 1) startup transient
    trans = cm.predict({"T0_brine_C": 45.0, "duration_s": 4000.0, "dt": 10.0})

    # 2) SEC vs compressor lift sweep
    lifts = np.linspace(2.0, 6.0, 15)
    sec_lift = [cm._model.specific_energy(dT_lift=l) for l in lifts]

    # 3) SEC vs brine temperature sweep
    Tbs = np.linspace(50.0, 72.0, 15)
    sec_T = [cm._model.specific_energy(T_brine_C=t) for t in Tbs]

    return cm, trans, (lifts, sec_lift), (Tbs, sec_T)


def make_report(path="simulation_report.html"):
    cm, trans, (lifts, sec_lift), (Tbs, sec_T) = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> text summary only
        d = cm._model.design_point()
        print("[simulate] Plotly unavailable, text summary only:", e)
        print(f"  SEC={d['SEC_kWh_m3']:.2f} kWh/m3  distillate={trans['distillate_m3_day_final']:.1f} m3/day")
        print(f"  T_steam={d['T_steam_C']:.2f} C  BPE={d['BPE_C']:.3f} K  GOR_eq={d['GOR_equiv']:.1f}")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Startup: brine temperature", "Startup: distillate production",
        "SEC vs compressor lift", "SEC vs brine temperature"))
    fig.add_trace(go.Scatter(x=trans["t"], y=trans["T_brine_C_series"],
                             name="T_brine"), row=1, col=1)
    fig.add_trace(go.Scatter(x=trans["t"], y=trans["distillate_m3_day_series"],
                             name="distillate"), row=1, col=2)
    fig.add_trace(go.Scatter(x=lifts, y=sec_lift, name="SEC vs lift"), row=2, col=1)
    fig.add_trace(go.Scatter(x=Tbs, y=sec_T, name="SEC vs Tb"), row=2, col=2)
    fig.update_layout(title="EC214 MVC F2a -- Physics-Lumped Simulation", height=720)
    out = os.path.join(os.path.dirname(__file__), "..", path)
    fig.write_html(out)
    print(f"[simulate] wrote {out}")
    return out


if __name__ == "__main__":
    make_report()
