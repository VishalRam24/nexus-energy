"""
EC107 -- Micro-CHP (Stirling-based) -- F2a Physics-Lumped
Optional Plotly report. Plotly import is wrapped so absence never crashes.
Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()

    # Cold-start warm-up at full fire
    r = cm.predict({"load_fraction": 1.0, "dt": 10.0, "duration_s": 3600.0})

    # Steady-state load sweep
    loads = [i / 20.0 for i in range(1, 21)]
    eta_e = [cm._model.steady_state(L)["eta_elec"] for L in loads]
    eta_th = [cm._model.steady_state(L)["eta_th"] for L in loads]
    eta_tot = [cm._model.steady_state(L)["eta_total"] for L in loads]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> print summary only
        ss = r["steady_state"]
        print("[simulate] plotly unavailable (%s); text summary:" % e)
        print(f"  Final head T : {r['temperature'][-1]:.1f} K")
        print(f"  P_elec       : {r['P_elec_W'][-1]:.0f} W")
        print(f"  Q_th         : {r['Q_th_W'][-1]:.0f} W")
        print(f"  eta_e/th/tot : {ss['eta_elec']:.3f} / {ss['eta_th']:.3f} / {ss['eta_total']:.3f}")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Head temperature warm-up",
            "Electrical & thermal output (warm-up)",
            "Efficiency vs burner load",
            "Warm-up gate factor",
        ),
    )
    fig.add_trace(go.Scatter(x=r["t"], y=r["temperature"], name="T_head [K]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["P_elec_W"], name="P_elec [W]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["Q_th_W"], name="Q_th [W]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=loads, y=eta_e, name="eta_elec"), row=2, col=1)
    fig.add_trace(go.Scatter(x=loads, y=eta_th, name="eta_th"), row=2, col=1)
    fig.add_trace(go.Scatter(x=loads, y=eta_tot, name="eta_total"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["warmup_factor"], name="warm-up gate"), row=2, col=2)
    fig.update_layout(title="EC107 Stirling micro-CHP F2a — physics-lumped report", height=750)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
