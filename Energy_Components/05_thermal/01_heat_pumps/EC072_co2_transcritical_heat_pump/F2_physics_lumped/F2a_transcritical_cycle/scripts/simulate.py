"""
EC072 -- CO2 Transcritical Heat Pump (R744) -- F2a Transcritical Cycle
Optional Plotly report: COP vs P_high (optimum), gas-cooler glide, water charge
transient. Plotly is wrapped in try/except so its absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def main():
    cm = ComponentModel()
    m = cm._model

    # 1) COP vs high-side pressure (the optimum).
    P = np.linspace(75.0, 130.0, 60)
    cop_P = np.array([m.cop(0.0, 15.0, p) for p in P])
    P_opt, cop_opt = m.optimum_high_pressure_search(0.0, 15.0)

    # 2) Gas-cooler glide profiles at several pressures.
    x = np.linspace(0.0, 1.0, 20)
    glides = {p: m.glide(0.0, 15.0, p, n=20) for p in [80.0, 95.0, 115.0]}

    # 3) Water charge transient.
    r = cm.predict({"T_source_c": 5.0, "T_water_in_c": 15.0,
                    "T_water_target_c": 65.0, "duration_s": 2400.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); printing summary only.")
        print(f"  P_opt={P_opt:.1f} bar, COP_opt={cop_opt:.3f}")
        print(f"  Water {r['T_water'][0]:.1f}->{r['T_water'][-1]:.1f} C")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Heating COP vs High-Side Pressure (optimum)",
        "Gas-Cooler CO2 Temperature Glide",
        "Water Charge Transient (lumped ODE)",
        "Instantaneous COP during charge"))

    fig.add_trace(go.Scatter(x=P, y=cop_P, name="COP", mode="lines"), 1, 1)
    fig.add_trace(go.Scatter(x=[P_opt], y=[cop_opt], name="optimum",
                             mode="markers", marker=dict(size=10)), 1, 1)
    for p, g in glides.items():
        fig.add_trace(go.Scatter(x=x, y=g, name=f"{p:.0f} bar",
                                 mode="lines"), 1, 2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_water"], name="T_water",
                             mode="lines"), 2, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["cop"], name="COP(t)",
                             mode="lines"), 2, 2)

    fig.update_xaxes(title_text="P_high [bar]", row=1, col=1)
    fig.update_yaxes(title_text="COP [-]", row=1, col=1)
    fig.update_xaxes(title_text="gas-cooler position [-]", row=1, col=2)
    fig.update_yaxes(title_text="CO2 T [degC]", row=1, col=2)
    fig.update_xaxes(title_text="time [s]", row=2, col=1)
    fig.update_yaxes(title_text="water T [degC]", row=2, col=1)
    fig.update_xaxes(title_text="time [s]", row=2, col=2)
    fig.update_yaxes(title_text="COP [-]", row=2, col=2)
    fig.update_layout(title="EC072 CO2 Transcritical Heat Pump -- F2a",
                      height=800, showlegend=True)
    fig.write_html(_OUT)
    print(f"[simulate] wrote {_OUT}")


if __name__ == "__main__":
    main()
