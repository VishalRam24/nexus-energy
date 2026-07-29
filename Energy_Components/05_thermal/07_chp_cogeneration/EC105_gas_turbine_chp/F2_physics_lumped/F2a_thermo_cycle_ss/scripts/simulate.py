"""
EC105 -- Gas Turbine CHP -- F2a Physics-Lumped Thermo Cycle
Optional Plotly simulation report (safe if plotly absent).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # Efficiency vs part-load sweep
    plr = np.linspace(0.4, 1.0, 25)
    eta_el, eta_th, eta_tot, hpr = [], [], [], []
    for p in plr:
        s = cm._model.cycle_state(p)
        eta_el.append(s["eta_electrical"])
        eta_th.append(s["eta_thermal"])
        eta_tot.append(s["eta_total"])
        hpr.append(s["heat_to_power_ratio"])

    # HRSG cold-start transient at full load
    sim = cm._model.simulate(1.0, dt=2.0, duration_s=1500.0)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] plotly unavailable ({e}); skipping HTML report.")
        s = cm._model.cycle_state(1.0)
        print(f"[simulate] full load: eta_el={s['eta_electrical']*100:.1f}% "
              f"eta_th={s['eta_thermal']*100:.1f}% eta_tot={s['eta_total']*100:.1f}% "
              f"HPR={s['heat_to_power_ratio']:.2f}")
        return

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Efficiency vs Part-Load",
                                        "HRSG Cold-Start Transient (full load)"))
    fig.add_trace(go.Scatter(x=plr, y=eta_el, name="eta_electrical"), 1, 1)
    fig.add_trace(go.Scatter(x=plr, y=eta_th, name="eta_thermal"), 1, 1)
    fig.add_trace(go.Scatter(x=plr, y=eta_tot, name="eta_total (CHP)"), 1, 1)
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["T_hrsg_K"],
                             name="T_HRSG"), 1, 2)
    fig.add_trace(go.Scatter(x=sim["t"],
                             y=np.full_like(sim["t"], sim["T_hrsg_steady_K"]),
                             name="T_steady", line=dict(dash="dash")), 1, 2)
    fig.update_xaxes(title_text="Part-load ratio", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency", row=1, col=1)
    fig.update_xaxes(title_text="Time [s]", row=1, col=2)
    fig.update_yaxes(title_text="HRSG temperature [K]", row=1, col=2)
    fig.update_layout(title="EC105 Gas Turbine CHP -- F2a Physics-Lumped",
                      template="plotly_white")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    run()
