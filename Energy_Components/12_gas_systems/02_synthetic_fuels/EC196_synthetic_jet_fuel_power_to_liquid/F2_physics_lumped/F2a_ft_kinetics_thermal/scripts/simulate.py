"""
EC196 -- Synthetic Jet Fuel (PtL) -- F2a
Optional Plotly simulation report. Plotly import is guarded so its absence
does not crash. Produces simulation_report.html when plotly is available.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # 1) Cold-start transient to steady state
    transient = cm.predict({"T0_C": 200.0, "dt": 30.0, "duration_s": 10800.0})

    # 2) Temperature sweep (steady map) of selectivity / conversion / efficiency
    m = cm._model
    T_sweep = np.linspace(180.0, 290.0, 60)
    X = np.array([m.conversion(T) for T in T_sweep])
    S = np.array([m.asf_selectivity_jet(T) for T in T_sweep])
    eta = np.array([m.ptl_efficiency(T) for T in T_sweep])
    alpha = m.alpha(T_sweep)

    # 3) ASF distribution at design alpha
    n, W = m.asf_weight_fractions(T_C=m.T_opt, n_max=40)

    return cm, transient, (T_sweep, X, S, eta, alpha), (n, W)


def build_report():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] plotly unavailable ({e}); skipping HTML report.")
        cm, *_ = run_scenarios()
        print("[simulate] scenarios computed successfully (no plot).")
        return

    cm, transient, sweep, asf = run_scenarios()
    T_sweep, X, S, eta, alpha = sweep
    n, W = asf

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Reactor thermal transient (cold start)",
            "CO conversion & jet selectivity vs T",
            "Power-to-Liquid efficiency & alpha vs T",
            "ASF weight distribution (jet cut shaded)",
        ),
    )

    fig.add_trace(go.Scatter(x=transient["t"] / 3600.0, y=transient["temperature"],
                             name="T_reactor [degC]"), row=1, col=1)

    fig.add_trace(go.Scatter(x=T_sweep, y=X, name="X_CO"), row=1, col=2)
    fig.add_trace(go.Scatter(x=T_sweep, y=S, name="S_jet"), row=1, col=2)

    fig.add_trace(go.Scatter(x=T_sweep, y=eta, name="eta_PtL"), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_sweep, y=alpha, name="alpha"), row=2, col=1)

    fig.add_trace(go.Bar(x=n, y=W, name="W_n",
                         marker_color=["crimson" if 8 <= c <= 16 else "steelblue"
                                       for c in n]),
                  row=2, col=2)

    fig.update_xaxes(title_text="time [h]", row=1, col=1)
    fig.update_xaxes(title_text="T [degC]", row=1, col=2)
    fig.update_xaxes(title_text="T [degC]", row=2, col=1)
    fig.update_xaxes(title_text="carbon number n", row=2, col=2)
    fig.update_layout(title_text="EC196 PtL Jet Fuel F2a -- FT Kinetics + Thermal ODE",
                      height=820, width=1180)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    build_report()
