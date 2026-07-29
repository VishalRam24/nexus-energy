"""
EC112 -- Micro Gas Turbine -- F2a Recuperated Brayton
Optional Plotly report. Plotly import is guarded so absence never crashes.
Run: python3 scripts/simulate.py  (writes ../simulation_report.html if plotly present)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report():
    cm = ComponentModel()
    m = cm._model

    # 1) part-load efficiency sweep
    plrs = [round(0.3 + 0.05 * i, 2) for i in range(15)]
    eta_pl = [cm.predict({"PLR": p})["eta_electrical"] for p in plrs]

    # 2) recuperator-effectiveness sweep
    eps_vals = [round(0.05 * i, 2) for i in range(19)]
    eta_eps = []
    for e in eps_vals:
        m.eps_rec = e
        eta_eps.append(m.cycle()["eta_electrical"])
    m.eps_rec = cm._raw["unit"]["recup_effectiveness"]["value"]

    # 3) cold-start transient
    tr = cm.predict({"mode": "transient", "fuel_fraction": 1.0,
                     "T_rec0_K": 288.15, "duration_s": 300.0, "dt": 2.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] plotly unavailable ({e}); printing summary instead.")
        print(f"  full-load eta = {eta_pl[-1]*100:.1f}%")
        print(f"  eta(eps=0) = {eta_eps[0]*100:.1f}%  eta(eps=0.85) = "
              f"{m.cycle()['eta_electrical']*100:.1f}%")
        print(f"  warm-up T_recup {tr['T_recup'][0]:.0f} -> {tr['T_recup'][-1]:.0f} K")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Electrical efficiency vs part-load",
        "Efficiency vs recuperator effectiveness",
        "Cold-start: recuperator metal temperature",
        "Cold-start: electrical power"))
    fig.add_trace(go.Scatter(x=plrs, y=[e*100 for e in eta_pl], name="eta_el"), 1, 1)
    fig.add_trace(go.Scatter(x=eps_vals, y=[e*100 for e in eta_eps], name="eta vs eps"), 1, 2)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["T_recup"], name="T_recup"), 2, 1)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["P_el_kw"], name="P_el"), 2, 2)
    fig.update_layout(title="EC112 Micro Gas Turbine -- F2a Recuperated Brayton", height=720)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    build_report()
