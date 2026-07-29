"""EC097 — Rankine Steam Turbine F1b — Simulation Scenarios + HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def run_simulation():
    model = ComponentModel()
    plr = np.linspace(0.2, 1.0, 80)
    T_conds = [20, 33, 42, 50]
    results = {"plr": plr}
    for T in T_conds:
        r = model.predict({"PLR": plr, "T_condenser": T})
        results[f"eta_net_T{T}"] = np.asarray(r["efficiency_net"])

    # Condenser temperature sweep at full load
    T_cond_arr = np.linspace(15, 55, 80)
    r2 = model.predict({"PLR": 1.0, "T_condenser": T_cond_arr})
    results["T_cond_arr"] = T_cond_arr
    results["eta_vs_T"] = np.asarray(r2["efficiency_net"])
    results["P_cond_vs_T"] = np.asarray(r2["condenser_pressure_kpa"])

    # Heat rejection
    r3 = model.predict({"PLR": plr, "T_condenser": 33.0})
    results["heat_rejection"] = np.asarray(r3["heat_rejection_mw"])
    results["heat_input"]     = np.asarray(r3["heat_input_mw"])
    return results


def generate_html_report(results, output_path):
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    plr = results["plr"]
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency vs PLR (condenser sensitivity)",
                        "Efficiency vs Condenser Temperature (PLR=1)",
                        "Heat Balance vs PLR",
                        "Condenser Pressure vs Temperature"])

    colors = ["steelblue", "green", "orange", "red"]
    T_conds = [20, 33, 42, 50]
    for T, color in zip(T_conds, colors):
        fig.add_trace(go.Scatter(x=plr, y=results[f"eta_net_T{T}"],
                                 name=f"T_cond={T}C", line=dict(color=color)), row=1, col=1)

    T_arr = results["T_cond_arr"]
    fig.add_trace(go.Scatter(x=T_arr, y=results["eta_vs_T"], name="eta_net (PLR=1)", line=dict(color="steelblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=results["heat_input"],     name="Q_in [MW]",  line=dict(color="firebrick")), row=2, col=1)
    fig.add_trace(go.Scatter(x=plr, y=results["heat_rejection"], name="Q_rej [MW]", line=dict(color="steelblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_arr, y=results["P_cond_vs_T"], name="P_sat [kPa]", line=dict(color="darkorange")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="T_condenser [degC]", row=1, col=2)
    fig.update_xaxes(title_text="PLR [-]", row=2, col=1)
    fig.update_xaxes(title_text="T_condenser [degC]", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency_net [-]", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency_net [-]", row=1, col=2)
    fig.update_yaxes(title_text="Heat [MW]", row=2, col=1)
    fig.update_yaxes(title_text="P_sat [kPa]", row=2, col=2)
    fig.update_layout(title_text="EC097 Rankine Steam Turbine F1b — Ambient + Part-Load", height=700)
    fig.write_html(str(output_path))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    results = run_simulation()
    out = Path(__file__).parent.parent / "simulation_report.html"
    generate_html_report(results, out)
    print(f"PLR=1.0, T_cond=33C: eta_net={results['eta_net_T33'][-1]:.3f}")
    print(f"PLR=1.0, T_cond=50C: eta_net={results['eta_net_T50'][-1]:.3f}")
