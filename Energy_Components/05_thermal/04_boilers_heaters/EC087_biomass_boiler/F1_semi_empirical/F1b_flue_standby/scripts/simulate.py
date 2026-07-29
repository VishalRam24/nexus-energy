"""EC087 — Biomass Boiler F1b — Simulation Scenarios + HTML Report"""
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
    plr = np.linspace(0.15, 1.0, 80)
    r = model.predict({"PLR": plr})
    results = {"plr": plr}
    for k, v in r.items():
        results[k] = np.asarray(v)

    # Moisture sensitivity
    moistures = [0.08, 0.15, 0.25, 0.35]
    moisture_sweep = {}
    for w in moistures:
        m = ComponentModel({"moisture_content": w})
        rv = m.predict({"PLR": plr})
        moisture_sweep[w] = np.asarray(rv["efficiency"])
    results["moisture_sweep"] = moisture_sweep
    results["moisture_plr"] = plr
    return results


def generate_html_report(results, output_path):
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency vs PLR", "Loss Breakdown vs PLR",
                        "Flue Gas Temperature vs PLR", "Moisture Sensitivity"])

    plr = results["plr"]
    fig.add_trace(go.Scatter(x=plr, y=results["efficiency"], name="eta", line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=results["flue_loss_kw"], name="Q_flue [kW]", line=dict(color="firebrick")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=results["standby_loss_kw"], name="Q_standby [kW]", line=dict(color="orange", dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=results["cycling_loss_kw"], name="Q_cycling [kW]", line=dict(color="purple", dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=results["flue_gas_temp_c"], name="T_flue [degC]", line=dict(color="darkorange")), row=2, col=1)
    colors = ["green", "blue", "orange", "red"]
    for (w, eta), color in zip(results["moisture_sweep"].items(), colors):
        fig.add_trace(go.Scatter(x=results["moisture_plr"], y=eta,
                                 name=f"w={w*100:.0f}%", line=dict(color=color)), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]")
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=1)
    fig.update_yaxes(title_text="Loss [kW]", row=1, col=2)
    fig.update_yaxes(title_text="T_flue [degC]", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency [-]", row=2, col=2)
    fig.update_layout(title_text="EC087 Biomass Boiler F1b — Flue + Moisture + Cycling Standby", height=700)
    fig.write_html(str(output_path))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    results = run_simulation()
    out = Path(__file__).parent.parent / "simulation_report.html"
    generate_html_report(results, out)
    print(f"PLR=1.0: eta={results['efficiency'][-1]:.3f}, Q_flue={results['flue_loss_kw'][-1]:.2f} kW")
    print(f"PLR=0.2: eta={results['efficiency'][0]:.3f}, Q_cycle={results['cycling_loss_kw'][0]:.2f} kW")
