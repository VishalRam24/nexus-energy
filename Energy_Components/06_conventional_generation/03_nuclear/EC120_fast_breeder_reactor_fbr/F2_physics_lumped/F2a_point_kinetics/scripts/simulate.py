"""
EC120 -- FBR F2a -- simulation scenarios + optional Plotly HTML report.
Plotly is imported lazily; its absence does not crash the script.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    scenarios = {
        "steady": cm.predict({"rho_ext": 0.0, "dt": 0.1, "duration_s": 40.0}),
        "step_+100pcm": cm.predict_step({"rho_step": 0.001, "dt": 0.02, "duration_s": 60.0}),
        "step_-100pcm": cm.predict_step({"rho_step": -0.001, "dt": 0.02, "duration_s": 60.0}),
        "ramp": cm.predict_ramp({"rho_rate": 5e-5, "rho_max": 0.001,
                                 "dt": 0.1, "duration_s": 100.0}),
    }
    return cm, scenarios


def build_report(out_html=None):
    cm, scenarios = run_scenarios()
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        for name, r in scenarios.items():
            print(f"  {name:14s}: n_final={r['n'][-1]:.4f}, "
                  f"T_f={r['T_f'][-1]:.1f} K, T_Na={r['T_Na'][-1]:.1f} K, "
                  f"BR={r['breeding_ratio'][-1]:.3f}")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Normalized power n(t)", "Fuel & sodium temperature",
        "Total reactivity rho(t)", "Cumulative net fissile bred"))
    for name, r in scenarios.items():
        fig.add_trace(go.Scatter(x=r["t"], y=r["n"], name=f"n {name}"), 1, 1)
    rk = scenarios["step_+100pcm"]
    fig.add_trace(go.Scatter(x=rk["t"], y=rk["T_f"], name="T_f"), 1, 2)
    fig.add_trace(go.Scatter(x=rk["t"], y=rk["T_Na"], name="T_Na"), 1, 2)
    fig.add_trace(go.Scatter(x=rk["t"], y=rk["rho"]*1e5, name="rho [pcm]"), 2, 1)
    fig.add_trace(go.Scatter(x=rk["t"], y=rk["net_fissile_bred_kg"]*1e3,
                             name="net bred [g]"), 2, 2)
    fig.update_layout(title="EC120 FBR F2a -- Fast-Spectrum Point Kinetics",
                      height=800, width=1100)
    fig.write_html(out_html)
    print(f"[simulate] Report written: {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
