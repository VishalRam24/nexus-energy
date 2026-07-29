"""EC095 — Peltier TEC — F1b Multi-Stage — Simulation Scenarios"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

BASE = Path(__file__).parent.parent


def run():
    model = ComponentModel()
    n = model._model.n_stages

    # Scenario 1: Current sweep
    I_arr = np.linspace(0.5, 6.0, 50)
    cop_cascade = []
    cop_single  = []
    for I in I_arr:
        r = model.predict({"current_stages": [I]*n, "T_cold": 0.0, "T_hot": 50.0})
        cop_cascade.append(float(r["COP"]))
        cop_single.append(float(r["COP_single_stage_ref"]))

    # Scenario 2: dT sweep at optimum current
    Tc_arr = np.linspace(-30.0, 15.0, 50)
    cop_c2  = []
    cop_s2  = []
    for Tc in Tc_arr:
        I_opt = model._model.optimum_currents(Tc, 50.0)
        r = model.predict({"current_stages": I_opt, "T_cold": Tc, "T_hot": 50.0})
        cop_c2.append(float(r["COP"]))
        cop_s2.append(float(r["COP_single_stage_ref"]))

    print("=== EC095 Peltier TEC F1b — Multi-Stage Cascade Simulation ===")
    print(f"\n[Current sweep] T_cold=0C, T_hot=50C, {n} stages:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  I={I_arr[i]:.1f}A: COP_cascade={cop_cascade[i]:.3f}, "
              f"COP_single={cop_single[i]:.3f}")

    print(f"\n[dT sweep] T_hot=50C, optimal current:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  T_cold={Tc_arr[i]:.0f}C (dT={50-Tc_arr[i]:.0f}K): "
              f"COP_cascade={cop_c2[i]:.3f}, COP_single={cop_s2[i]:.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["COP vs Current", "COP vs Temperature Span"])
        fig.add_trace(go.Scatter(x=I_arr, y=cop_cascade, mode="lines",
                                 name=f"{n}-stage cascade"), row=1, col=1)
        fig.add_trace(go.Scatter(x=I_arr, y=cop_single, mode="lines",
                                 name="Single-stage ref", line=dict(dash="dash")),
                      row=1, col=1)
        dT_arr = 50.0 - Tc_arr
        fig.add_trace(go.Scatter(x=dT_arr, y=cop_c2, mode="lines",
                                 name=f"{n}-stage cascade"), row=1, col=2)
        fig.add_trace(go.Scatter(x=dT_arr, y=cop_s2, mode="lines",
                                 name="Single-stage ref", line=dict(dash="dash")),
                      row=1, col=2)
        fig.update_layout(title="EC095 Peltier TEC — F1b Multi-Stage Cascade", height=450)
        fig.update_xaxes(title_text="Current [A]", row=1, col=1)
        fig.update_xaxes(title_text="Temperature span [K]", row=1, col=2)
        fig.update_yaxes(title_text="COP [-]", row=1, col=1)
        html_path = BASE / "simulation_report.html"
        fig.write_html(str(html_path))
        print(f"\nReport written to {html_path}")
    except ImportError:
        print("plotly not available — skipping HTML report")


if __name__ == "__main__":
    run()
