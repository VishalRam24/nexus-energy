"""EC103 — sCO2 Brayton Cycle F1b — Simulation Scenarios + HTML Report"""
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

    # PLR sweep
    plr = np.linspace(0.25, 1.0, 80)
    r_plr = model.predict({"PLR": plr})
    results = {"plr": plr}
    for k, v in r_plr.items():
        results[f"plr_{k}"] = np.asarray(v)

    # T_reject sweep at PLR=1 — critical region
    T_rejs = np.linspace(25, 58, 200)
    r_Tr = model.predict({"PLR": 1.0, "T_reject": T_rejs})
    results["T_reject_arr"] = T_rejs
    results["eta_vs_Treject"] = np.asarray(r_Tr["efficiency_net"])
    results["fT_vs_Treject"]  = np.asarray(r_Tr["f_T_reject"])

    # PLR sweep for recuperator effectiveness
    results["recup_eps"] = np.asarray(r_plr["recuperator_effectiveness"])
    return results


def generate_html_report(results, output_path):
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    plr = results["plr"]
    T_rej = results["T_reject_arr"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency vs PLR",
                        "T_reject Sensitivity (PLR=1) — Critical at 31.1°C",
                        "Recuperator Effectiveness vs PLR",
                        "Correction Factors vs PLR"])

    fig.add_trace(go.Scatter(x=plr, y=results["plr_efficiency_net"],   name="eta_net",   line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=results["plr_efficiency_gross"],  name="eta_gross", line=dict(color="steelblue", dash="dash")), row=1, col=1)
    fig.add_vline(x=31.1, line_dash="dot", line_color="red", row=2, col=1,
                  annotation_text="T_crit=31.1°C")
    fig.add_trace(go.Scatter(x=T_rej, y=results["eta_vs_Treject"], name="eta_net (PLR=1)", line=dict(color="firebrick")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=results["recup_eps"],         name="eps_recuperator", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=plr, y=results["plr_f_partload"],    name="f_PLR",     line=dict(color="green")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="T_reject [degC]", row=1, col=2)
    fig.update_xaxes(title_text="PLR [-]", row=2, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=2)
    fig.update_yaxes(title_text="epsilon_recup [-]", row=2, col=1)
    fig.update_yaxes(title_text="Correction factor [-]", row=2, col=2)
    fig.update_layout(title_text="EC103 sCO2 Brayton F1b — T_reject + Part-Load + Recuperator", height=700)
    fig.write_html(str(output_path))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    results = run_simulation()
    out = Path(__file__).parent.parent / "simulation_report.html"
    generate_html_report(results, out)
    print(f"PLR=1.0, T_rej=32C: eta_net={results['plr_efficiency_net'][-1]:.3f}")
    idx_Tr_max = np.argmax(results["T_reject_arr"] >= 50.0)
    print(f"PLR=1.0, T_rej=50C: eta_net={results['eta_vs_Treject'][idx_Tr_max]:.3f}")
