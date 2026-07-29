"""
EC108 -- Steam Turbine CHP -- F2a Physics-Lumped
Optional Plotly report: part-load performance map + boiler thermal transient.
Plotly is imported lazily so its absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()

    # part-load sweep
    plr = np.linspace(0.3, 1.0, 30)
    P_el, Q_use, eta_tot, eta_el, p2h = [], [], [], [], []
    for x in plr:
        s = cm.predict({"PLR": float(x)})
        P_el.append(s["P_el_kw"]); Q_use.append(s["Q_useful_kw"])
        eta_tot.append(s["eta_total"]); eta_el.append(s["eta_el"])
        p2h.append(s["power_to_heat"])

    # warm-up transient
    tr = cm.predict({"PLR": 1.0, "transient": True, "T0_C": 80.0,
                     "duration_s": 2400.0, "dt": 10.0})["transient"]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Plotly not available; printing summary instead.")
        s = cm.predict({"PLR": 1.0})
        print(f"Rated: P_el={s['P_el_kw']:.0f} kW, Q_useful={s['Q_useful_kw']:.0f} kW, "
              f"eta_total={s['eta_total']:.3f}, power-to-heat={s['power_to_heat']:.3f}")
        print(f"Transient: T_boiler {tr['T_boiler_C'][0]:.0f} -> {tr['T_boiler_C'][-1]:.0f} degC")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Electrical & Useful Heat vs PLR",
                        "Efficiencies vs PLR",
                        "Power-to-Heat Ratio vs PLR",
                        "Boiler Thermal Transient (cold start)"),
    )
    fig.add_trace(go.Scatter(x=plr, y=P_el, name="P_el [kW_e]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=Q_use, name="Q_useful [kW_th]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=eta_tot, name="eta_total"), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=eta_el, name="eta_el"), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=p2h, name="power/heat"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["T_boiler_C"], name="T_boiler [degC]"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["P_el_kw"], name="P_el(t) [kW]",
                             yaxis="y2"), row=2, col=2)

    fig.update_layout(title="EC108 Steam Turbine CHP -- F2a Physics-Lumped",
                      height=800, width=1100)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
