"""
EC102 -- Kalina Cycle -- F2a Physics-Lumped
Simulation scenarios + optional Plotly HTML report.
Plotly import is wrapped so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # 1) efficiency & power vs source temperature
    T_src = np.linspace(90.0, 210.0, 40)
    eta, eta_c, P, glide_h = [], [], [], []
    for T in T_src:
        r = cm.predict({"T_source_c": float(T), "T_sink_c": 25.0, "Q_in_kw": 1000.0})
        eta.append(r["eta_thermal"]); eta_c.append(r["eta_carnot"])
        P.append(r["P_net_kW"]); glide_h.append(r["glide_hot_K"])

    # 2) glide vs ammonia fraction
    x_arr = np.linspace(0.30, 0.95, 40)
    glide_x = [cm._model.glide_width(30.0, float(x)) for x in x_arr]

    # 3) transient drum temperature
    tr = cm.predict({"T_source_c": 150.0, "transient": True,
                     "duration_s": 1800.0, "Q_source_kw": 1000.0})["transient"]

    return {
        "T_src": T_src, "eta": eta, "eta_c": eta_c, "P": P, "glide_h": glide_h,
        "x_arr": x_arr, "glide_x": glide_x, "tr": tr,
    }


def build_report(data, out_html="simulation_report.html"):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Efficiency vs Source T (eta < Carnot)",
                        "Net Power vs Source T",
                        "Temperature Glide vs NH3 fraction",
                        "Transient Drum Temperature"))
    fig.add_trace(go.Scatter(x=data["T_src"], y=data["eta"], name="eta_thermal"), 1, 1)
    fig.add_trace(go.Scatter(x=data["T_src"], y=data["eta_c"], name="eta_Carnot",
                             line=dict(dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=data["T_src"], y=data["P"], name="P_net [kW]"), 1, 2)
    fig.add_trace(go.Scatter(x=data["x_arr"], y=data["glide_x"], name="glide [K]"), 2, 1)
    fig.add_trace(go.Scatter(x=data["tr"]["t"], y=data["tr"]["T_drum_K"],
                             name="T_drum [K]"), 2, 2)
    fig.update_layout(title="EC102 Kalina Cycle F2a -- Physics-Lumped", height=800)
    path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(path)
    print(f"[simulate] Report written to {path}")
    return path


if __name__ == "__main__":
    data = run_scenarios()
    print(f"eta range: {min(data['eta']):.3f}..{max(data['eta']):.3f}")
    print(f"P_net range: {min(data['P']):.1f}..{max(data['P']):.1f} kW")
    print(f"max glide vs x: {max(data['glide_x']):.1f} K")
    build_report(data)
