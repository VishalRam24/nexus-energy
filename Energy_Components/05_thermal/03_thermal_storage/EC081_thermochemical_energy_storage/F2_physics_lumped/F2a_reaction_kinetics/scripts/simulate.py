"""
EC081 -- Thermochemical Energy Storage (CaO/Ca(OH)2) -- F2a Reaction Kinetics
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash the module.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    m = ComponentModel()
    scenarios = {
        "charge": m.predict({"mode": "charge", "X0": 0.0, "T0_K": 723.15,
                             "T_source_K": 873.15, "duration_s": 7200.0, "dt": 30.0}),
        "discharge": m.predict({"mode": "discharge", "X0": 1.0, "T0_K": 873.15,
                                "T_source_K": 623.15, "duration_s": 7200.0, "dt": 30.0}),
        "hold": m.predict({"mode": "hold", "X0": 0.7, "T0_K": 700.0,
                           "T_source_K": 700.0, "duration_s": 86400.0, "dt": 600.0}),
    }
    return m, scenarios


def build_report(path="simulation_report.html"):
    m, scen = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> print summary only
        print(f"[simulate] Plotly unavailable ({e}); printing text summary.")
        for name, r in scen.items():
            print(f"  {name:9s}: SOC {r['SOC'][0]:.3f}->{r['SOC'][-1]:.3f}, "
                  f"T {r['temperature'][0]-273.15:.0f}->{r['temperature'][-1]-273.15:.0f} C")
        return None

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Conversion / SOC", "Bed temperature [C]",
                                        "Reaction heat Q_rxn [kW]"))
    for name, r in scen.items():
        t_h = r["t"] / 3600.0
        fig.add_trace(go.Scatter(x=t_h, y=r["SOC"], name=f"{name} SOC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_h, y=r["temperature"] - 273.15,
                                name=f"{name} T"), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_h, y=r["Q_rxn_W"] / 1e3,
                                name=f"{name} Q"), row=3, col=1)
    fig.update_xaxes(title_text="time [h]", row=3, col=1)
    fig.update_layout(height=900, title_text="EC081 F2a Thermochemical Storage")
    out = os.path.join(os.path.dirname(__file__), "..", path)
    fig.write_html(out)
    print(f"[simulate] Report written to {out}")
    return out


if __name__ == "__main__":
    build_report()
