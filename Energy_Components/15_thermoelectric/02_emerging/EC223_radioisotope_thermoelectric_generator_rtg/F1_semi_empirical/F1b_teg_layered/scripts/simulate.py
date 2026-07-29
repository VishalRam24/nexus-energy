"""EC223 — RTG — F1b — Simulation Scenarios + HTML Report"""
import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    p = model.params["unit"]

    # --- Scenario 1: Power and temperatures over mission lifetime ---
    t_arr = np.linspace(0.0, 100.0, 200)
    results = [model.predict({"t_years": float(ti)}) for ti in t_arr]
    P_thermal = np.array([float(r["P_thermal_W"]) for r in results])
    P_electric = np.array([float(r["P_electric_W"]) for r in results])
    T_hj = np.array([float(r["T_hj_K"]) for r in results])
    T_cj = np.array([float(r["T_cj_K"]) for r in results])
    eta = np.array([float(r["eta_teg"]) for r in results])
    ZT = np.array([float(r["ZT_avg"]) for r in results])
    V_oc = np.array([float(r["V_oc_V"]) for r in results])

    # --- Scenario 2: ZT vs temperature (material property) ---
    T_range = np.linspace(500.0, 1300.0, 200)
    ZT_T = model._model.zt_local(T_range)
    alpha_T = model._model.alpha(T_range) * 1e6  # uV/K
    k_T = model._model.k_thermal(T_range)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "P_thermal and P_electric vs Mission Time",
                "Junction Temperatures vs Time",
                "Efficiency and ZT_avg vs Time",
                "SiGe Material Properties vs Temperature",
            ],
        )

        fig.add_trace(go.Scatter(x=t_arr, y=P_thermal / 1000, name="P_thermal (kW)",
                                  line=dict(color="red")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_arr, y=P_electric, name="P_electric (W)",
                                  line=dict(color="blue")), row=1, col=1)

        fig.add_trace(go.Scatter(x=t_arr, y=T_hj - 273.15, name="T_hot_junction (°C)",
                                  line=dict(color="red")), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_arr, y=T_cj - 273.15, name="T_cold_junction (°C)",
                                  line=dict(color="blue", dash="dash")), row=1, col=2)

        fig.add_trace(go.Scatter(x=t_arr, y=eta * 100, name="eta_teg (%)",
                                  line=dict(color="green")), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_arr, y=ZT, name="ZT_avg",
                                  line=dict(color="purple", dash="dot")), row=2, col=1)

        fig.add_trace(go.Scatter(x=T_range - 273.15, y=ZT_T, name="ZT(T)",
                                  line=dict(color="orange")), row=2, col=2)
        fig.add_trace(go.Scatter(x=T_range - 273.15, y=k_T, name="k (W/m/K)",
                                  line=dict(color="teal", dash="dash")), row=2, col=2)

        fig.update_xaxes(title_text="Time [years]", row=1, col=1)
        fig.update_yaxes(title_text="Power [kW] / [W]", row=1, col=1)
        fig.update_xaxes(title_text="Time [years]", row=1, col=2)
        fig.update_yaxes(title_text="Temperature [°C]", row=1, col=2)
        fig.update_xaxes(title_text="Time [years]", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency [%] / ZT [-]", row=2, col=1)
        fig.update_xaxes(title_text="Temperature [°C]", row=2, col=2)
        fig.update_yaxes(title_text="ZT [-] / k [W/m/K]", row=2, col=2)

        fig.update_layout(
            title="EC223 RTG F1b — Multi-Layer SiGe TEG Model",
            height=800,
        )

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not installed — skipping HTML report.")

    r0 = model.predict({"t_years": 0.0})
    print("\nEC223 RTG F1b — Design Point (t=0, BOL):")
    for k, v in r0.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")


if __name__ == "__main__":
    run_simulations()
