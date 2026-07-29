"""
EC086 -- Electric Boiler / Resistance Heater -- F2a Dynamic Thermal Mass
Optional Plotly simulation report. Plotly import is guarded so its absence
does not break the build/test workflow.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    cm = ComponentModel()
    m = cm._model

    # Scenario: cold start with a midday draw spike.
    def draw(t):
        return 0.08 if 1800.0 <= t <= 3600.0 else 0.01

    r_onoff = m.simulate(293.15, draw, dt=2.0, duration_s=7200.0,
                         control="onoff")
    r_mod = m.simulate(293.15, draw, dt=2.0, duration_s=7200.0,
                       control="modulating")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # plotly absent -- print summary instead
        print(f"[simulate] Plotly unavailable ({exc}); text summary:")
        for name, r in [("on/off", r_onoff), ("modulating", r_mod)]:
            e = r["energy"]
            print(f"  {name:11s}: T_final={r['temperature'][-1]-273.15:6.2f} C "
                  f"E_elec={e['E_elec_J']/3.6e6:6.3f} kWh "
                  f"E_load={e['E_load_J']/3.6e6:6.3f} kWh "
                  f"resid={e['E_residual_J']:.2e} J")
        return

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Water temperature",
                                        "Firing fraction",
                                        "Power flows (on/off)"))
    for name, r, dash in [("on/off", r_onoff, None),
                          ("modulating", r_mod, "dash")]:
        fig.add_trace(go.Scatter(x=r["t"]/60.0, y=r["temperature"]-273.15,
                                 name=f"T ({name})",
                                 line=dict(dash=dash)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["t"]/60.0, y=r["firing_fraction"],
                                 name=f"u ({name})",
                                 line=dict(dash=dash)), row=2, col=1)
    fig.add_hline(y=m.T_set-273.15, line_dash="dot", row=1, col=1)
    fig.add_trace(go.Scatter(x=r_onoff["t"]/60.0, y=r_onoff["Q_elec_W"]/1000.0,
                             name="Q_elec [kW]"), row=3, col=1)
    fig.add_trace(go.Scatter(x=r_onoff["t"]/60.0, y=r_onoff["Q_loss_W"],
                             name="Q_loss [W]"), row=3, col=1)
    fig.add_trace(go.Scatter(x=r_onoff["t"]/60.0, y=r_onoff["Q_load_W"]/1000.0,
                             name="Q_load [kW]"), row=3, col=1)
    fig.update_xaxes(title_text="time [min]", row=3, col=1)
    fig.update_layout(title="EC086 Electric Boiler F2a -- Dynamic Thermal Mass",
                      height=850)
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")


if __name__ == "__main__":
    run_report()
