"""
EC145 -- Pyrolysis Reactor -- F2a
Optional Plotly report: dynamic product evolution + yield-vs-temperature sweep.
Plotly import is wrapped so its absence does not crash.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html="simulation_report.html"):
    cm = ComponentModel()

    # 1) Dynamic run: heated reactor
    dyn = cm.predict({"Q_ext_W": 6000.0, "T0_K": 550.0, "dt": 0.5, "duration_s": 180.0})

    # 2) Equilibrium yield vs temperature sweep
    temps_C = np.linspace(300.0, 800.0, 26)
    oil, char, gas = [], [], []
    for Tc in temps_C:
        eq = cm.predict({"mode": "isothermal", "T_isothermal": Tc + 273.15})
        oil.append(eq["bio_oil_yield"])
        char.append(eq["char_yield"])
        gas.append(eq["gas_yield"])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  Dynamic final: bio-oil {dyn['bio_oil_yield']*100:.1f}%  "
              f"char {dyn['char_yield']*100:.1f}%  gas {dyn['gas_yield']*100:.1f}%")
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Dynamic product evolution", "Equilibrium yield vs temperature"),
    )
    t = dyn["t"]
    fig.add_trace(go.Scatter(x=t, y=dyn["y_bio_oil"], name="bio-oil"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=dyn["y_char"], name="char"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=dyn["y_gas"], name="gas"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=dyn["y_biomass"], name="biomass"), 1, 1)

    fig.add_trace(go.Scatter(x=temps_C, y=oil, name="bio-oil (eq)"), 1, 2)
    fig.add_trace(go.Scatter(x=temps_C, y=char, name="char (eq)"), 1, 2)
    fig.add_trace(go.Scatter(x=temps_C, y=gas, name="gas (eq)"), 1, 2)

    fig.update_xaxes(title_text="time [s]", row=1, col=1)
    fig.update_xaxes(title_text="temperature [degC]", row=1, col=2)
    fig.update_yaxes(title_text="mass fraction", row=1, col=1)
    fig.update_yaxes(title_text="yield fraction", row=1, col=2)
    fig.update_layout(title="EC145 Pyrolysis Reactor F2a — Arrhenius kinetics + energy balance")

    out_path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(out_path)
    print(f"[simulate] report written to {out_path}")
    return out_path


if __name__ == "__main__":
    build_report()
