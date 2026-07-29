"""EC089 — Hydrogen Boiler F1b — Simulation Scenarios + HTML Report"""
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
    model_cond  = ComponentModel({"condensing": True})
    model_ncond = ComponentModel({"condensing": False})
    plr = np.linspace(0.1, 1.0, 80)

    rc = model_cond.predict({"PLR": plr})
    rn = model_ncond.predict({"PLR": plr})

    return {
        "plr": plr,
        "cond":   {k: np.asarray(v) for k, v in rc.items()},
        "ncond":  {k: np.asarray(v) for k, v in rn.items()},
    }


def generate_html_report(results, output_path):
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    plr = results["plr"]
    c = results["cond"]
    n = results["ncond"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Efficiency: Condensing vs Non-condensing",
                        "Flue Loss + Latent Recovery",
                        "H2 Mass Flow vs PLR",
                        "Flue Gas Temperature vs PLR"])

    fig.add_trace(go.Scatter(x=plr, y=c["efficiency"],  name="eta condensing",     line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=n["efficiency"],  name="eta non-condensing", line=dict(color="red", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=c["flue_loss_kw"],       name="Q_flue (cond) [kW]",    line=dict(color="firebrick")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=n["flue_loss_kw"],       name="Q_flue (noncond) [kW]", line=dict(color="orange",    dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=c["latent_recovery_kw"], name="Q_latent (cond) [kW]",  line=dict(color="teal",      dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=c["h2_flow_kg_s"] * 1000, name="H2 flow [g/s]", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=plr, y=c["flue_gas_temp_c"],  name="T_flue cond [degC]",     line=dict(color="darkorange")), row=2, col=2)
    fig.add_trace(go.Scatter(x=plr, y=n["flue_gas_temp_c"],  name="T_flue noncond [degC]",  line=dict(color="red", dash="dash")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]")
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=1, col=2)
    fig.update_yaxes(title_text="H2 flow [g/s]", row=2, col=1)
    fig.update_yaxes(title_text="Temperature [degC]", row=2, col=2)
    fig.update_layout(title_text="EC089 Hydrogen Boiler F1b — H2O-rich Flue + Condensing", height=700)
    fig.write_html(str(output_path))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    results = run_simulation()
    out = Path(__file__).parent.parent / "simulation_report.html"
    generate_html_report(results, out)
    c = results["cond"]
    n = results["ncond"]
    print(f"PLR=1.0 condensing:     eta={c['efficiency'][-1]:.3f}, Q_flue={c['flue_loss_kw'][-1]:.2f} kW, Q_lat={c['latent_recovery_kw'][-1]:.2f} kW")
    print(f"PLR=1.0 non-condensing: eta={n['efficiency'][-1]:.3f}, Q_flue={n['flue_loss_kw'][-1]:.2f} kW")
