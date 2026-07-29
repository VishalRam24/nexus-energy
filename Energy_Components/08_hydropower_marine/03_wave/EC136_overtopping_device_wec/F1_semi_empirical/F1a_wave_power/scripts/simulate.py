"""EC136 — Overtopping WEC — F1a — Simulation Scenarios + HTML Report"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulation():
    model = ComponentModel()
    H_s_vals = np.linspace(0.5, 6.0, 50)
    T_e_vals  = np.array([7.0, 10.0, 13.0, 16.0])

    curves = {}
    for T_e in T_e_vals:
        r = model.predict({"H_s": H_s_vals, "T_e": T_e})
        curves[T_e] = r["power_kw"]

    H_grid, T_grid = np.meshgrid(np.linspace(0.5, 6.0, 30), np.linspace(5.0, 20.0, 30))
    P_grid = model.predict({"H_s": H_grid, "T_e": T_grid})["power_kw"]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Power vs H_s (various T_e)",
                                            "Power Heatmap (H_s × T_e)"])

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, T_e in enumerate(T_e_vals):
            fig.add_trace(go.Scatter(x=H_s_vals, y=curves[T_e],
                                     name=f"T_e={T_e}s", line=dict(color=colors[i])), row=1, col=1)

        fig.add_trace(go.Heatmap(z=P_grid,
                                  x=np.linspace(0.5, 6.0, 30),
                                  y=np.linspace(5.0, 20.0, 30),
                                  colorscale="Blues", showscale=True), row=1, col=2)

        fig.update_xaxes(title_text="H_s [m]", row=1, col=1)
        fig.update_yaxes(title_text="Power [kW]", row=1, col=1)
        fig.update_xaxes(title_text="H_s [m]", row=1, col=2)
        fig.update_yaxes(title_text="T_e [s]", row=1, col=2)

        fig.update_layout(title="EC136 Overtopping WEC F1a — Simulation Report",
                          height=500, width=1000)

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not available — skipping HTML report.")

    print("\n=== EC136 Overtopping WEC — Summary ===")
    eta = float(model.predict({"H_s": 2.0, "T_e": 10.0})["overall_efficiency"])
    print(f"  Overall wave-to-wire efficiency: {eta:.4f} ({eta*100:.1f}%)")


if __name__ == "__main__":
    run_simulation()
