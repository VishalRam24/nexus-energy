"""EC138 — OTEC — F1a — Simulation Scenarios + HTML Report"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulation():
    model = ComponentModel()

    # Scenario 1: efficiency vs ΔT (T_warm fixed, T_cold varies)
    T_warm_fixed = 26.0
    T_cold_vals  = np.linspace(2.0, 20.0, 50)
    dT_vals = T_warm_fixed - T_cold_vals
    r_dT = model.predict({"T_warm": T_warm_fixed, "T_cold": T_cold_vals})

    # Scenario 2: T_warm sweep (T_cold fixed)
    T_warm_vals = np.linspace(20.0, 32.0, 30)
    r_Tw = model.predict({"T_warm": T_warm_vals, "T_cold": 5.0})

    # Scenario 3: Power map
    T_warm_grid, T_cold_grid = np.meshgrid(np.linspace(20, 32, 20), np.linspace(2, 12, 20))
    r_map = model.predict({"T_warm": T_warm_grid, "T_cold": T_cold_grid})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Efficiency vs ΔT (T_warm=26°C)",
                                            "Efficiency vs T_warm (T_cold=5°C)",
                                            "Net Power Heatmap (kW)"])

        fig.add_trace(go.Scatter(x=dT_vals, y=np.asarray(r_dT["eta_carnot"])*100,
                                  name="Carnot", line=dict(dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dT_vals, y=np.asarray(r_dT["eta_gross"])*100,
                                  name="Gross"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dT_vals, y=np.asarray(r_dT["eta_net"])*100,
                                  name="Net"), row=1, col=1)

        fig.add_trace(go.Scatter(x=T_warm_vals, y=np.asarray(r_Tw["eta_net"])*100,
                                  name="eta_net", line=dict(color="#2ca02c")), row=1, col=2)

        fig.add_trace(go.Heatmap(z=r_map["P_net_kw"],
                                  x=np.linspace(20, 32, 20),
                                  y=np.linspace(2, 12, 20),
                                  colorscale="Hot_r", showscale=True), row=1, col=3)

        fig.update_xaxes(title_text="ΔT [°C]", row=1, col=1)
        fig.update_yaxes(title_text="Efficiency [%]", row=1, col=1)
        fig.update_xaxes(title_text="T_warm [°C]", row=1, col=2)
        fig.update_yaxes(title_text="eta_net [%]", row=1, col=2)
        fig.update_xaxes(title_text="T_warm [°C]", row=1, col=3)
        fig.update_yaxes(title_text="T_cold [°C]", row=1, col=3)

        fig.update_layout(title="EC138 OTEC F1a — Simulation Report", height=500, width=1400)

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not available — skipping HTML report.")

    r_ref = model.predict({"T_warm": 26.0, "T_cold": 5.0})
    print("\n=== EC138 OTEC — Summary (T_warm=26°C, T_cold=5°C) ===")
    print(f"  Carnot: {float(r_ref['eta_carnot'])*100:.2f}%")
    print(f"  Gross:  {float(r_ref['eta_gross'])*100:.2f}%")
    print(f"  Net:    {float(r_ref['eta_net'])*100:.2f}%")
    print(f"  P_net:  {float(r_ref['P_net_kw']):.1f} kW")


if __name__ == "__main__":
    run_simulation()
