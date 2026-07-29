"""
EC151 -- Dry Steam Geothermal Plant -- F2a Physics-Lumped
Simulation scenarios + optional Plotly HTML report.
Plotly is wrapped in try/except so its absence never crashes the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # Scenario A: pressure-vs-power sweep (quasi-steady)
    P_wh = np.linspace(0.4, 1.3, 30)
    P_net = np.array([m.power(50.0, p, 0.012, 0.0)["P_net_kW"] for p in P_wh])
    eta_u = np.array([m.power(50.0, p, 0.012, 0.0)["eta_utilization"] for p in P_wh])
    eta_c = np.array([m.carnot_efficiency(p, 0.012) for p in P_wh])

    # Scenario B: wellhead pressure step transient
    def P_step(t):
        return 0.6 if t < 60.0 else 1.0
    trans = m.simulate(P_step, m_dot0=m._mdot_target(0.6), dt=2.0, duration_s=300.0)

    # Scenario C: NCG sensitivity
    x_ncg = np.linspace(0.0, 0.10, 20)
    P_ncg = np.array([m.power(50.0, 0.8, 0.012, 0.0, x)["P_net_kW"] for x in x_ncg])

    return {
        "sweep": {"P_wh": P_wh, "P_net": P_net, "eta_u": eta_u, "eta_c": eta_c},
        "transient": trans,
        "ncg": {"x": x_ncg, "P_net": P_ncg},
    }


def make_report(results, out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Net power vs wellhead pressure", "Efficiency vs pressure (util < Carnot)",
        "Wellhead pressure-step transient", "NCG fraction vs net power"))

    s = results["sweep"]
    fig.add_trace(go.Scatter(x=s["P_wh"], y=s["P_net"], name="P_net (kW)"), 1, 1)
    fig.add_trace(go.Scatter(x=s["P_wh"], y=s["eta_u"], name="eta_util"), 1, 2)
    fig.add_trace(go.Scatter(x=s["P_wh"], y=s["eta_c"], name="eta_Carnot"), 1, 2)

    t = results["transient"]
    fig.add_trace(go.Scatter(x=t["t"], y=t["P_net_kW"], name="P_net (kW)"), 2, 1)
    fig.add_trace(go.Scatter(x=t["t"], y=t["m_dot"], name="m_dot (kg/s)", yaxis="y2"), 2, 1)

    n = results["ncg"]
    fig.add_trace(go.Scatter(x=n["x"], y=n["P_net"], name="P_net vs NCG"), 2, 2)

    fig.update_layout(title="EC151 Dry Steam Geothermal -- F2a Physics-Lumped",
                      height=800, showlegend=True)
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    res = run_scenarios()
    print("Sweep P_net range:", res["sweep"]["P_net"].min(), "->",
          res["sweep"]["P_net"].max(), "kW")
    print("Transient final P_net:", res["transient"]["P_net_kW"][-1], "kW")
    make_report(res)
