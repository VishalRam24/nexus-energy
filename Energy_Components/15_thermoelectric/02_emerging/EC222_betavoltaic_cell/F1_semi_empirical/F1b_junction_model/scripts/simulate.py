"""EC222 — Betavoltaic Cell — F1b — Simulation Scenarios + HTML Report"""
import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    p = model.params["unit"]
    t_half = p["t_half_years"]["value"]
    design_life = p["design_life_years"]["value"]

    # --- Scenario 1: Power, Isc, Voc, FF over full design life ---
    t_arr = np.linspace(0.0, design_life * 2, 300)
    r_time = model.predict({"t_years": t_arr, "T_cell_K": 300.0})

    # --- Scenario 2: Temperature sweep at t=0 ---
    T_arr = np.linspace(200.0, 450.0, 200)
    r_T = model.predict({"t_years": 0.0, "T_cell_K": T_arr})

    # --- Scenario 3: Comparison with F1a approach (simple activity model) ---
    # F1a: P = A * E_beta * eta_cap * eta_conv
    eta_conv_f1a = model._model.eta_cap  # capture only; conv was separate in F1a
    A = np.asarray(r_time["activity_Bq"])
    P_f1a = A * p["E_beta_MeV"]["value"] * 1.602176634e-13 * model._model.eta_cap * 0.06  # 6% conv

    # --- Scenario 4: Voc decay over time ---
    t_long = np.linspace(0.0, 500.0, 400)
    r_Voc = model.predict({"t_years": t_long})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "P_out and Isc vs Time (0→2×Design Life)",
                "Voc and FF vs Time",
                "P_out vs Temperature at t=0",
                "F1b vs F1a Power Estimate",
            ],
        )

        fig.add_trace(go.Scatter(x=t_arr, y=np.asarray(r_time["P_out_uW"]),
                                  name="P_out (uW)", line=dict(color="blue")), row=1, col=1)
        fig_ax2 = go.Scatter(x=t_arr, y=np.asarray(r_time["Isc_uA"]),
                              name="Isc (uA)", line=dict(color="orange", dash="dash"),
                              yaxis="y2")
        fig.add_trace(fig_ax2, row=1, col=1)

        fig.add_trace(go.Scatter(x=t_arr, y=np.asarray(r_time["Voc_V"]),
                                  name="Voc (V)", line=dict(color="green")), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_arr, y=np.asarray(r_time["FF"]),
                                  name="Fill Factor", line=dict(color="red", dash="dot")), row=1, col=2)

        fig.add_trace(go.Scatter(x=T_arr, y=np.asarray(r_T["P_out_uW"]),
                                  name="P_out vs T (uW)", line=dict(color="purple")), row=2, col=1)

        fig.add_trace(go.Scatter(x=t_arr, y=np.asarray(r_time["P_out_uW"]),
                                  name="F1b (junction)", line=dict(color="blue")), row=2, col=2)
        fig.add_trace(go.Scatter(x=t_arr, y=P_f1a * 1e6,
                                  name="F1a (activity only)", line=dict(color="gray", dash="dash")),
                      row=2, col=2)

        fig.update_xaxes(title_text="Time [years]", row=1, col=1)
        fig.update_yaxes(title_text="P_out [uW]", row=1, col=1)
        fig.update_xaxes(title_text="Time [years]", row=1, col=2)
        fig.update_yaxes(title_text="Voc [V] / FF [-]", row=1, col=2)
        fig.update_xaxes(title_text="Cell temperature [K]", row=2, col=1)
        fig.update_yaxes(title_text="P_out [uW]", row=2, col=1)
        fig.update_xaxes(title_text="Time [years]", row=2, col=2)
        fig.update_yaxes(title_text="Power [uW]", row=2, col=2)

        fig.update_layout(
            title="EC222 Betavoltaic Cell F1b — Junction Electrical Model",
            height=800,
        )

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not installed — skipping HTML report.")

    r0 = model.predict({"t_years": 0.0, "T_cell_K": 300.0})
    print("\nEC222 Betavoltaic F1b — Design Point (t=0, T=300K):")
    for k, v in r0.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")


if __name__ == "__main__":
    run_simulations()
