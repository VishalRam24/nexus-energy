"""
EC124 -- LAES F2a Cryo-Tank Thermo -- simulation scenarios + Plotly report.
Plotly is optional; absence does not crash the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    m = ComponentModel()

    # Scenario A: cold-recycle sweep
    eps_vals = np.linspace(0.0, 0.9, 19)
    rte_eps = [m.predict({"mode": "round_trip", "cold_recycle_eff": float(e)})["eta_RT"]
               for e in eps_vals]

    # Scenario B: hot (waste-heat) recycle sweep
    dT_vals = np.linspace(0.0, 250.0, 26)
    rte_hot = [m.predict({"mode": "round_trip", "cold_recycle_eff": 0.60,
                          "hot_recycle_dT_K": float(d)})["eta_RT"] for d in dT_vals]

    # Scenario C: charge then 48 h storage boil-off
    rc = m.predict({"mode": "charge", "duration_s": 1.5e6 / 100.0 / 4,
                    "m_dot_kgs": 100.0})
    rs = m.predict({"mode": "store", "duration_s": 48 * 3600.0,
                    "m_liq0_kg": float(rc["m_liq"][-1]), "T_amb_K": 298.15})

    return eps_vals, rte_eps, dT_vals, rte_hot, rs


def build_report():
    eps_vals, rte_eps, dT_vals, rte_hot, rs = run_scenarios()
    out_path = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Plotly not available -- skipping HTML report.")
        print(f"Cold-recycle RTE range: {min(rte_eps)*100:.1f}% .. {max(rte_eps)*100:.1f}%")
        print(f"Hot-recycle RTE range:  {min(rte_hot)*100:.1f}% .. {max(rte_hot)*100:.1f}%")
        return

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "RTE vs cold recycle", "RTE vs waste-heat dT", "Boil-off storage (48 h)"))
    fig.add_trace(go.Scatter(x=eps_vals, y=np.array(rte_eps) * 100, mode="lines+markers",
                             name="cold recycle"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dT_vals, y=np.array(rte_hot) * 100, mode="lines+markers",
                             name="hot recycle"), row=1, col=2)
    fig.add_trace(go.Scatter(x=rs["t"] / 3600.0, y=rs["m_liq"] / 1e3, mode="lines",
                             name="liquid mass"), row=1, col=3)
    fig.update_xaxes(title_text="eps_cr [-]", row=1, col=1)
    fig.update_yaxes(title_text="RTE [%]", row=1, col=1)
    fig.update_xaxes(title_text="dT [K]", row=1, col=2)
    fig.update_xaxes(title_text="time [h]", row=1, col=3)
    fig.update_yaxes(title_text="liquid air [t]", row=1, col=3)
    fig.update_layout(title="EC124 LAES F2a -- Cryo-Tank Thermodynamic Model")
    fig.write_html(out_path)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    build_report()
