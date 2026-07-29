"""
EC138 -- OTEC F2a -- simulation scenarios + optional Plotly report.

Generates: (1) a transient warm-water step response, (2) a net-power vs warm-water
temperature sweep showing the dT cutoff where net power goes positive.
Plotly is imported lazily so its absence does not crash the script.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from predict import ComponentModel

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # Scenario 1: transient warm-water step 22 -> 28 degC
    def Tw_step(t):
        return 22.0 if t < 3600.0 else 28.0
    transient = m.simulate(Tw_step, 5.0, dt=30.0, duration_s=7200.0)

    # Scenario 2: net power vs warm-water temperature
    Tw_sweep = np.linspace(8.0, 30.0, 45)
    P_net = np.array([m.steady_state(float(Tw), 5.0)["P_net_kw"] for Tw in Tw_sweep])
    P_gross = np.array([m.steady_state(float(Tw), 5.0)["P_gross_kw"] for Tw in Tw_sweep])
    eta_net = np.array([m.steady_state(float(Tw), 5.0)["eta_net"] for Tw in Tw_sweep])

    return transient, (Tw_sweep, P_gross, P_net, eta_net)


def main():
    transient, sweep = run_scenarios()
    Tw_sweep, P_gross, P_net, eta_net = sweep

    print("OTEC F2a simulation scenarios")
    print(f"  transient final net power : {transient['P_net_kw'][-1]:.1f} kW")
    cutoff = Tw_sweep[np.argmax(P_net > 0)] if np.any(P_net > 0) else None
    print(f"  net-power-positive warm-T cutoff (cold=5C): "
          f"{cutoff:.1f} degC" if cutoff is not None else "  no positive net power")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Transient: WF sat temps after warm-T step",
                        "Transient: net power response",
                        "Net & gross power vs warm-water T",
                        "Net efficiency vs warm-water T"),
    )
    fig.add_trace(go.Scatter(x=transient["t"]/3600, y=transient["T_evap_c"],
                             name="T_evap"), row=1, col=1)
    fig.add_trace(go.Scatter(x=transient["t"]/3600, y=transient["T_cond_c"],
                             name="T_cond"), row=1, col=1)
    fig.add_trace(go.Scatter(x=transient["t"]/3600, y=transient["P_net_kw"],
                             name="P_net"), row=1, col=2)
    fig.add_trace(go.Scatter(x=Tw_sweep, y=P_gross, name="P_gross"), row=2, col=1)
    fig.add_trace(go.Scatter(x=Tw_sweep, y=P_net, name="P_net"), row=2, col=1)
    fig.add_trace(go.Scatter(x=Tw_sweep, y=eta_net*100, name="eta_net %"), row=2, col=2)
    fig.update_layout(title="EC138 OTEC F2a -- Physics-Lumped Thermal Cycle",
                      height=800, showlegend=True)
    fig.write_html(_OUT)
    print(f"  report written: {_OUT}")


if __name__ == "__main__":
    main()
