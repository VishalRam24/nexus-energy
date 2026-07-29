"""EC099 — Stirling Engine F1b — Simulation Scenarios + HTML Report"""
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
    T_ambs = [-10, 10, 25, 40]
    results = {"plr": plr, "T_ambs": T_ambs}
    for T in T_ambs:
        r = model.predict({"PLR": plr, "T_ambient": T})
        results[f"eta_net_T{T}"] = np.asarray(r["efficiency_net"])
        results[f"T_c_T{T}"]     = np.asarray(r["T_cold_side_c"])

    # T_h sensitivity
    T_hs = [450, 550, 650, 750]
    for T in T_hs:
        r = model.predict({"PLR": 1.0, "T_hot": float(T),
                           "T_ambient": np.linspace(-10, 40, 80)})
        results[f"eta_vs_Tamb_Th{T}"] = np.asarray(r["efficiency_net"])
    results["T_amb_arr"] = np.linspace(-10, 40, 80)
    return results


def generate_html_report(results, output_path):
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    plr = results["plr"]
    colors = ["steelblue", "green", "orange", "red"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency vs PLR (ambient sensitivity)",
                        "Cold-side Temperature vs PLR",
                        "Efficiency vs Ambient (PLR=1, T_h sensitivity)",
                        "Part-load Factor vs PLR"])

    T_ambs = results["T_ambs"]
    for T, color in zip(T_ambs, colors):
        fig.add_trace(go.Scatter(x=plr, y=results[f"eta_net_T{T}"],
                                 name=f"T_amb={T}C", line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=plr, y=results[f"T_c_T{T}"],
                                 name=f"T_c (T_amb={T}C)", line=dict(color=color, dash="dot"),
                                 showlegend=False), row=1, col=2)

    T_amb_arr = results["T_amb_arr"]
    T_hs = [450, 550, 650, 750]
    for T_h, color in zip(T_hs, colors):
        fig.add_trace(go.Scatter(x=T_amb_arr, y=results[f"eta_vs_Tamb_Th{T_h}"],
                                 name=f"T_h={T_h}C", line=dict(color=color)), row=2, col=1)

    model = ComponentModel()
    r_pl = model.predict({"PLR": plr})
    fig.add_trace(go.Scatter(x=plr, y=np.asarray(r_pl["f_partload"]),
                             name="f_PLR", line=dict(color="purple")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=1, col=2)
    fig.update_xaxes(title_text="T_ambient [degC]", row=2, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=1)
    fig.update_yaxes(title_text="T_cold [degC]", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=2, col=1)
    fig.update_yaxes(title_text="f_PLR [-]", row=2, col=2)
    fig.update_layout(title_text="EC099 Stirling Engine F1b — Ambient T_c + Part-Load", height=700)
    fig.write_html(str(output_path))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    results = run_simulation()
    out = Path(__file__).parent.parent / "simulation_report.html"
    generate_html_report(results, out)
    print(f"PLR=1.0, T_amb=25C: eta={results['eta_net_T25'][-1]:.3f}")
    print(f"PLR=1.0, T_amb=40C: eta={results['eta_net_T40'][-1]:.3f}")
