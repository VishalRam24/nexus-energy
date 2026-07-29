"""EC041 — EDLC Supercapacitor — F1a RC — Simulation & HTML Report Generator"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def integrate(model, v0, current_fn, t_end, dt):
    """Forward-Euler integration of dV_cap/dt for a time-varying current."""
    n = int(t_end / dt) + 1
    t = np.linspace(0.0, n * dt, n + 1)
    v_cap = np.zeros(n + 1)
    v_term = np.zeros(n + 1)
    i_arr = np.zeros(n + 1)
    v_cap[0] = v0
    for k in range(n):
        I = current_fn(t[k])
        i_arr[k] = I
        r = model.predict({"v_cap": v_cap[k], "current": I})
        v_term[k] = float(r["voltage"])
        v_cap[k + 1] = v_cap[k] + float(r["dvcap_dt"]) * dt
        v_cap[k + 1] = max(0.0, min(model._model.v_max, v_cap[k + 1]))
    i_arr[-1] = current_fn(t[-1])
    r = model.predict({"v_cap": v_cap[-1], "current": i_arr[-1]})
    v_term[-1] = float(r["voltage"])
    return t, v_cap, v_term, i_arr


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    v_max = model.params["cell"]["v_max"]["value"]
    C = model.params["cell"]["capacitance"]["value"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "V-Q characteristic (linear capacitor)",
            "Constant-current discharge from V_max",
            "Self-discharge (leakage only)",
            "Stored energy vs voltage",
        ],
        vertical_spacing=0.13,
    )

    # 1) V vs Q
    v_cap_arr = np.linspace(0.0, v_max, 100)
    r = model.predict({"v_cap": v_cap_arr, "current": 0.0})
    fig.add_trace(
        go.Scatter(x=r["charge"], y=v_cap_arr, name="V_cap = Q/C", line=dict(color="black")),
        row=1, col=1,
    )

    # 2) Constant-current discharge
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, I in enumerate([20.0, 50.0, 100.0, 200.0]):
        # tau ~ C * V / I  -> simulate to ~95% drain
        t_end = 0.95 * C * v_max / I
        dt = max(t_end / 500.0, 0.01)
        t, vc, vt, _ = integrate(model, v_max, lambda tt, II=I: II, t_end, dt)
        fig.add_trace(go.Scatter(x=t, y=vt, name=f"I={I:.0f} A", line=dict(color=colors[i])),
                      row=1, col=2)

    # 3) Self-discharge (I=0) — long timescale
    t, vc, vt, _ = integrate(model, v_max, lambda tt: 0.0, t_end=72 * 3600.0, dt=60.0)
    fig.add_trace(
        go.Scatter(x=t / 3600.0, y=vc, name="Leakage (I=0)", line=dict(color="purple"), showlegend=False),
        row=2, col=1,
    )

    # 4) Stored energy vs voltage
    e = model.predict({"v_cap": v_cap_arr, "current": 0.0})["stored_energy"]
    fig.add_trace(
        go.Scatter(x=v_cap_arr, y=e, name="0.5*C*V^2", line=dict(color="green"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Charge (C)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_xaxes(title_text="V_cap (V)", row=2, col=2)
    fig.update_yaxes(title_text="V_cap (V)", row=1, col=1)
    fig.update_yaxes(title_text="V_terminal (V)", row=1, col=2)
    fig.update_yaxes(title_text="V_cap (V)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (J)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}: {info['description']}",
        height=800, template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
