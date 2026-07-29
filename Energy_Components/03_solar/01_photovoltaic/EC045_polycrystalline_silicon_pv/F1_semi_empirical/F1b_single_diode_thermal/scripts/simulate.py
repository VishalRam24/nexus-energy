"""EC045 — Poly-Si PV — F1b — Simulation Scenarios + HTML Report"""

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

    # 1. Power vs irradiance at different ambient temperatures
    irr = np.linspace(50, 1200, 100)
    results["irr_sweep"] = {}
    for T_amb in [5.0, 20.0, 35.0, 50.0]:
        r = model.predict({"irradiance_w_m2": irr, "T_ambient_degC": T_amb})
        results["irr_sweep"][T_amb] = {"irr": irr, "p_mp": r["p_mp"], "T_cell": r["T_cell_c"]}

    # 2. Efficiency vs ambient temperature at STC irradiance
    T_ambs = np.linspace(-10, 50, 80)
    r = model.predict({"irradiance_w_m2": 1000.0, "T_ambient_degC": T_ambs})
    results["eff_vs_T"] = {"T_amb": T_ambs, "efficiency": r["efficiency"], "T_cell": r["T_cell_c"]}

    # 3. I-V style: sweep cell temperature for tempco check
    T_cells = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    r = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": T_cells})
    results["tempco"] = {"T_cell": T_cells, "p_mp": r["p_mp"], "v_oc": r["v_oc"]}

    return results


def generate_report(model: ComponentModel, output_path: Path):
    data = run_simulations(model)

    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power vs Irradiance (various T_amb)",
            "Cell Temperature vs Irradiance",
            "Efficiency vs Ambient Temperature (1000 W/m2)",
            "Tempco: Pmp & Voc vs Cell Temperature",
        ],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (T_amb, d) in enumerate(data["irr_sweep"].items()):
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=d["irr"], y=d["p_mp"], name=f"T_amb={T_amb}C",
                                 line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=d["irr"], y=d["T_cell"], name=f"T_cell T_amb={T_amb}C",
                                 line=dict(color=c, dash="dot"), showlegend=False), row=1, col=2)

    d = data["eff_vs_T"]
    fig.add_trace(go.Scatter(x=d["T_amb"], y=d["efficiency"] * 100,
                             name="Efficiency (%)", line=dict(color="blue")), row=2, col=1)

    d = data["tempco"]
    fig.add_trace(go.Scatter(x=d["T_cell"], y=d["p_mp"],
                             name="Pmp (W)", line=dict(color="red")), row=2, col=2)
    fig.add_trace(go.Scatter(x=d["T_cell"], y=d["v_oc"],
                             name="Voc (V)", line=dict(color="orange"),
                             yaxis="y5"), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=2)
    fig.update_xaxes(title_text="T_ambient (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_cell (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Pmp (W)", row=1, col=1)
    fig.update_yaxes(title_text="T_cell (degC)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Pmp (W)", row=2, col=2)

    fig.update_layout(
        title="EC045 Poly-Si PV — F1b Single-Diode + Thermal Simulation",
        height=700,
    )
    fig.write_html(str(output_path))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    model = ComponentModel()
    report_path = Path(__file__).parent.parent / "simulation_report.html"
    generate_report(model, report_path)
