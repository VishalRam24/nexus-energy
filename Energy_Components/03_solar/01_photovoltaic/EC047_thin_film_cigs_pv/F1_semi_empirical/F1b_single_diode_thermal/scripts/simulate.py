"""EC047 — Thin-Film CIGS PV — F1b — Simulation Scenarios + HTML Report"""

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

    irr = np.linspace(50, 1200, 100)
    results["irr_sweep"] = {}
    for T_amb in [5.0, 20.0, 35.0, 50.0]:
        r = model.predict({"irradiance_w_m2": irr, "T_ambient_degC": T_amb})
        results["irr_sweep"][T_amb] = {"irr": irr, "p_mp": r["p_mp"]}

    T_cells = np.linspace(10, 75, 60)
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": T_cells})
    results["tempco"] = {"T_cell": T_cells, "p_mp": r["p_mp"], "v_oc": r["v_oc"]}

    T_ambs = np.linspace(-5, 50, 60)
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": T_ambs})
    results["eff_vs_T"] = {"T_amb": T_ambs, "efficiency": r["efficiency"],
                           "T_cell": r["T_cell_c"]}
    return results


def generate_report(model: ComponentModel, output_path: Path):
    data = run_simulations(model)
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Power vs Irradiance", "Pmp vs Cell Temp",
                                        "Efficiency vs Ambient Temp",
                                        "Technology Tempco Comparison"])

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (T_amb, d) in enumerate(data["irr_sweep"].items()):
        fig.add_trace(go.Scatter(x=d["irr"], y=d["p_mp"], name=f"T_amb={T_amb}C",
                                 line=dict(color=colors[i])), row=1, col=1)

    d = data["tempco"]
    fig.add_trace(go.Scatter(x=d["T_cell"], y=d["p_mp"], name="Pmp",
                             line=dict(color="teal")), row=1, col=2)

    d = data["eff_vs_T"]
    fig.add_trace(go.Scatter(x=d["T_amb"], y=d["efficiency"] * 100,
                             name="Efficiency %", line=dict(color="green")), row=2, col=1)

    # Comparison with CdTe and poly-Si
    T_c = np.linspace(10, 75, 60)
    ref_idx = np.argmin(np.abs(T_c - 25.0))
    pv_data = {"CIGS -0.31%/K": (-0.0031, "teal"),
               "CdTe -0.28%/K": (-0.0028, "green"),
               "Poly-Si -0.39%/K": (-0.0039, "orange")}
    for label, (gamma, color) in pv_data.items():
        p_norm = (1 + gamma * (T_c - 25)) * 100
        fig.add_trace(go.Scatter(x=T_c, y=p_norm, name=label,
                                 line=dict(color=color)), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="T_cell (degC)", row=1, col=2)
    fig.update_xaxes(title_text="T_ambient (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_cell (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Pmp (W)", row=1, col=1)
    fig.update_yaxes(title_text="Pmp (W)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Normalised Pmp (%)", row=2, col=2)

    fig.update_layout(title="EC047 CIGS PV — F1b Single-Diode + Thermal", height=700)
    fig.write_html(str(output_path))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    model = ComponentModel()
    generate_report(model, Path(__file__).parent.parent / "simulation_report.html")
