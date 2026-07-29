"""
EC119 -- MSR F2a -- simulation scenarios + optional Plotly HTML report.
Plotly import is guarded so its absence does not crash the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    scenarios = {
        "reactivity_step_+50pcm": {"rho_ext_pcm": 50.0, "flow_fraction": 1.0,
                                    "dt": 0.5, "duration_s": 400.0},
        "rod_drop_-300pcm": {"rho_ext_pcm": -300.0, "flow_fraction": 1.0,
                              "dt": 0.5, "duration_s": 300.0},
        "flow_compare_stagnant": {"rho_ext_pcm": 100.0, "flow_fraction": 0.0,
                                   "dt": 0.2, "duration_s": 50.0},
        "flow_compare_rated": {"rho_ext_pcm": 100.0, "flow_fraction": 1.0,
                                "dt": 0.2, "duration_s": 50.0},
    }
    results = {name: cm.predict(cfg) for name, cfg in scenarios.items()}
    return cm, results


def main():
    cm, results = run_scenarios()
    info = cm.get_info()
    b = results["reactivity_step_+50pcm"]
    print(f"{info['component_name']} ({info['fidelity']})")
    print(f"beta_static={b['beta_static']:.5f}, beta_eff(flow=1)={b['beta_eff']:.5f}")
    for name, r in results.items():
        print(f"  {name:30s} P/P0_final={r['power_fraction'][-1]:.3f}  "
              f"T_core_final={r['T_core_K'][-1]:.1f} K")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # noqa: BLE001
        print(f"[plotly unavailable: {e}] skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Power fraction", "Core temperature [K]"))
    for name, r in results.items():
        fig.add_trace(go.Scatter(x=r["t"], y=r["power_fraction"], name=name),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=r["t"], y=r["T_core_K"], name=name,
                                 showlegend=False), row=2, col=1)
    fig.update_layout(title="EC119 MSR F2a -- Flowing-Fuel Point Kinetics",
                      height=700)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
