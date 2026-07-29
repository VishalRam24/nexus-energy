"""EC052 — Bifacial PV Module — F1b — Simulation Scenarios + HTML Report"""

import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def run_simulations(model: ComponentModel):
    results = {}

    irr = np.linspace(100, 1200, 80)
    results["irr_sweep"] = {}
    for alb in [0.1, 0.2, 0.4, 0.6]:
        r = model.predict({"irradiance_front_w_m2": irr, "T_ambient_degC": 25.0, "albedo": alb})
        results["irr_sweep"][alb] = {"irr": irr, "p_mp": r["p_mp"], "G_rear": r["G_rear_w_m2"]}

    # Bifacial gain vs albedo at STC
    albedos = np.linspace(0.0, 0.9, 50)
    p_vals = [float(model.predict({"irradiance_front_w_m2": 1000.0,
                                    "T_ambient_degC": 25.0,
                                    "albedo": float(a)})["p_mp"]) for a in albedos]
    p_front = float(model.predict({"irradiance_front_w_m2": 1000.0,
                                    "T_ambient_degC": 25.0,
                                    "albedo": 0.0})["p_mp"])
    gains = [(p - p_front) / p_front * 100 for p in p_vals]
    results["bifacial_gain"] = {"albedo": albedos, "gain_pct": np.array(gains)}

    # Temperature comparison: front vs rear vs effective
    G_fronts = np.linspace(100, 1200, 60)
    r = model.predict({"irradiance_front_w_m2": G_fronts, "T_ambient_degC": 20.0, "albedo": 0.2})
    results["cell_temps"] = {
        "G_front": G_fronts,
        "T_front": r["T_cell_front_c"],
        "T_rear": r["T_cell_rear_c"],
        "T_eff": r["T_cell_eff_c"],
    }

    # Power vs T_amb
    T_ambs = np.linspace(-5, 50, 50)
    r = model.predict({"irradiance_front_w_m2": 1000.0, "T_ambient_degC": T_ambs, "albedo": 0.2})
    results["power_vs_T"] = {"T_amb": T_ambs, "p_mp": r["p_mp"]}

    return results


def generate_report(model: ComponentModel, output_path: Path):
    data = run_simulations(model)
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Power vs Irradiance (various albedo)",
                                        "Bifacial Gain vs Albedo",
                                        "Cell Temperatures vs Front Irradiance",
                                        "Power vs Ambient Temperature"])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (alb, d) in enumerate(data["irr_sweep"].items()):
        fig.add_trace(go.Scatter(x=d["irr"], y=d["p_mp"],
                                 name=f"albedo={alb}", line=dict(color=colors[i])), row=1, col=1)

    d = data["bifacial_gain"]
    fig.add_trace(go.Scatter(x=d["albedo"], y=d["gain_pct"],
                             name="Bifacial Gain (%)", line=dict(color="teal")), row=1, col=2)

    d = data["cell_temps"]
    fig.add_trace(go.Scatter(x=d["G_front"], y=d["T_front"],
                             name="T_cell_front", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["G_front"], y=d["T_rear"],
                             name="T_cell_rear", line=dict(color="blue", dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=d["G_front"], y=d["T_eff"],
                             name="T_cell_eff", line=dict(color="purple", dash="dash")), row=2, col=1)

    d = data["power_vs_T"]
    fig.add_trace(go.Scatter(x=d["T_amb"], y=d["p_mp"],
                             name="Pmp (W)", line=dict(color="orange")), row=2, col=2)

    fig.update_xaxes(title_text="G_front (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Albedo", row=1, col=2)
    fig.update_xaxes(title_text="G_front (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="T_ambient (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Pmp (W)", row=1, col=1)
    fig.update_yaxes(title_text="Bifacial Gain (%)", row=1, col=2)
    fig.update_yaxes(title_text="T_cell (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Pmp (W)", row=2, col=2)

    fig.update_layout(title="EC052 Bifacial PV — F1b Bifacial + Thermal Simulation", height=700)
    fig.write_html(str(output_path))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    model = ComponentModel()
    generate_report(model, Path(__file__).parent.parent / "simulation_report.html")
