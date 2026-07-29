"""
EC064 -- Small / Micro Wind Turbine -- F2a BEM Rotor-Dynamics
Optional Plotly report. Plotly import is guarded so absence never crashes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # 1) Spin-up transients at several wind speeds
    winds = [4.0, 6.0, 8.0, 10.0]
    transients = {U: cm.predict({"wind_speed_ms": U, "omega0_rad_s": 1.0,
                                 "dt": 0.05, "duration_s": 60.0}) for U in winds}

    # 2) Cp(lambda) curve
    lam = np.linspace(0.0, 14.0, 300)
    cp = m.Cp(lam, 0.0)

    # 3) Steady-state power curve from torque-balance roots
    U_sweep = np.linspace(0.0, m.v_cut_out + 2.0, 60)
    P_ss = []
    for U in U_sweep:
        w = m.steady_state(U)
        P_ss.append(m.gen_power_elec(w) / 1000.0)
    P_ss = np.array(P_ss)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> text summary only
        print(f"[simulate] Plotly unavailable ({e}); text summary:")
        for U in winds:
            r = transients[U]
            print(f"  U={U:>4} m/s -> omega={r['omega'][-1]:6.2f} rad/s, "
                  f"Cp={r['Cp'][-1]:.3f}, P_elec={r['P_elec'][-1]/1000:.2f} kW")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Rotor speed spin-up", "Cp(lambda)",
        "Steady-state power curve", "TSR transient"))

    for U in winds:
        r = transients[U]
        fig.add_trace(go.Scatter(x=r["t"], y=r["rpm"], name=f"{U} m/s"), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["t"], y=r["tsr"], name=f"TSR {U} m/s",
                                 showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=lam, y=cp, name="Cp"), row=1, col=2)
    fig.add_trace(go.Scatter(x=U_sweep, y=P_ss, name="P_elec [kW]"), row=2, col=1)

    fig.update_xaxes(title_text="t [s]", row=1, col=1)
    fig.update_yaxes(title_text="rpm", row=1, col=1)
    fig.update_xaxes(title_text="lambda", row=1, col=2)
    fig.update_yaxes(title_text="Cp", row=1, col=2)
    fig.update_xaxes(title_text="wind [m/s]", row=2, col=1)
    fig.update_yaxes(title_text="P_elec [kW]", row=2, col=1)
    fig.update_xaxes(title_text="t [s]", row=2, col=2)
    fig.update_yaxes(title_text="TSR", row=2, col=2)
    fig.update_layout(title="EC064 Small Wind Turbine -- F2a BEM Rotor-Dynamics",
                      height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
