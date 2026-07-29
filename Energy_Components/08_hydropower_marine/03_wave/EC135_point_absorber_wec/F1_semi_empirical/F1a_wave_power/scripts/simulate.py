"""EC135 — Point Absorber WEC — F1a — Simulation Scenarios + HTML Report"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulation():
    model = ComponentModel()
    T_e_vals  = np.linspace(4.0, 20.0, 60)
    H_s_vals  = np.array([1.0, 2.0, 3.0, 4.0])

    # Scenario 1: P vs T_e at various H_s (shows resonance peak)
    curves = {}
    for H_s in H_s_vals:
        r = model.predict({"H_s": H_s, "T_e": T_e_vals})
        curves[H_s] = r["power_kw"]

    # Scenario 2: CWR vs T_e
    r_cwr = model.predict({"H_s": 2.0, "T_e": T_e_vals})

    # Scenario 3: H_s sweep at T_n
    H_s_sweep = np.linspace(0.0, 6.0, 40)
    r_hs = model.predict({"H_s": H_s_sweep, "T_e": 10.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Power vs T_e (resonance peak)",
                                            "CWR vs T_e",
                                            "Power vs H_s at T_n=10s"])

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, H_s in enumerate(H_s_vals):
            fig.add_trace(go.Scatter(x=T_e_vals, y=curves[H_s],
                                     name=f"H_s={H_s}m", line=dict(color=colors[i])), row=1, col=1)

        fig.add_trace(go.Scatter(x=T_e_vals, y=r_cwr["capture_width_ratio"],
                                  name="CWR", line=dict(color="#9467bd")), row=1, col=2)

        fig.add_trace(go.Scatter(x=H_s_sweep, y=r_hs["power_kw"],
                                  name="P vs H_s", line=dict(color="#8c564b")), row=1, col=3)

        fig.update_xaxes(title_text="T_e [s]", row=1, col=1)
        fig.update_yaxes(title_text="Power [kW]", row=1, col=1)
        fig.update_xaxes(title_text="T_e [s]", row=1, col=2)
        fig.update_yaxes(title_text="CWR [-]", row=1, col=2)
        fig.update_xaxes(title_text="H_s [m]", row=1, col=3)
        fig.update_yaxes(title_text="Power [kW]", row=1, col=3)

        fig.update_layout(title="EC135 Point Absorber WEC F1a — Simulation Report",
                          height=500, width=1400)

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not available — skipping HTML report.")

    print("\n=== EC135 Point Absorber — Summary (T_n=10s) ===")
    print(f"  CWR at resonance: {float(r_cwr['capture_width_ratio'][30]):.3f}")
    print(f"  P at H_s=2m, T_e=10s: {float(curves[2.0][30]):.2f} kW")


if __name__ == "__main__":
    run_simulation()
