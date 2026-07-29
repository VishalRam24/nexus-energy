"""EC059 — Evacuated Tube Solar Collector — F1b — Simulation Scenarios + HTML Report"""

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

    # 1. Efficiency curve: eta vs (T_in - T_amb)/G (normalized curve)
    G_ref = 800.0
    T_amb = 10.0
    dT_G = np.linspace(0, 0.15, 50)
    T_ins = T_amb + dT_G * G_ref
    r = model.predict({"irradiance_w_m2": G_ref, "T_inlet_degC": T_ins, "T_ambient_degC": T_amb})
    results["eff_curve"] = {"dT_G": dT_G, "efficiency": r["efficiency"],
                             "U_L_eff": r["U_L_eff_w_m2k"]}

    # 2. Efficiency at different irradiance levels (shows non-linearity)
    T_in_vals = np.linspace(20, 150, 60)
    results["irr_effect"] = {}
    for G in [400.0, 800.0, 1000.0]:
        r = model.predict({"irradiance_w_m2": G, "T_inlet_degC": T_in_vals,
                            "T_ambient_degC": 15.0})
        results["irr_effect"][G] = {"T_in": T_in_vals, "efficiency": r["efficiency"]}

    # 3. IAM curve
    thetas = np.linspace(0, 78, 60)
    r_iam = [float(model.predict({"irradiance_w_m2": 800.0, "T_inlet_degC": 50.0,
                                   "T_ambient_degC": 20.0,
                                   "incidence_angle_deg": float(th)})["iam"]) for th in thetas]
    results["iam_curve"] = {"theta": thetas, "iam": np.array(r_iam)}

    # 4. U_L(DeltaT) curve
    dTs = np.linspace(0, 120, 60)
    from model import EvacuatedTubeF1b
    import json as _json
    params_path = Path(__file__).parent.parent / "data" / "parameters.json"
    with open(params_path) as f:
        params = _json.load(f)
    m = EvacuatedTubeF1b(params)
    results["UL_curve"] = {"dT": dTs, "U_L": m.U_L(dTs)}

    return results


def generate_report(model: ComponentModel, output_path: Path):
    data = run_simulations(model)
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Efficiency Curve (eta vs dT/G)",
                                        "Efficiency vs T_inlet at different G",
                                        "Incidence Angle Modifier (IAM)",
                                        "U_L(DeltaT) — Loss Coefficient Curve"])

    d = data["eff_curve"]
    fig.add_trace(go.Scatter(x=d["dT_G"], y=d["efficiency"] * 100, name="ETC F1b",
                             line=dict(color="blue")), row=1, col=1)
    # Add flat-plate reference for comparison (typical a1=4.0, a2=0.06)
    dT_G = d["dT_G"]
    eta_fp = np.clip(0.80 - 4.0 * dT_G - 0.06 * (dT_G * 800) * dT_G, 0, 100)
    fig.add_trace(go.Scatter(x=dT_G, y=eta_fp * 100, name="Flat-plate ref",
                             line=dict(color="gray", dash="dash")), row=1, col=1)

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, (G, d) in enumerate(data["irr_effect"].items()):
        fig.add_trace(go.Scatter(x=d["T_in"], y=d["efficiency"] * 100,
                                 name=f"G={G} W/m2", line=dict(color=colors[i])), row=1, col=2)

    d = data["iam_curve"]
    fig.add_trace(go.Scatter(x=d["theta"], y=d["iam"], name="IAM",
                             line=dict(color="orange")), row=2, col=1)

    d = data["UL_curve"]
    fig.add_trace(go.Scatter(x=d["dT"], y=d["U_L"], name="U_L (W/m2K)",
                             line=dict(color="red")), row=2, col=2)

    fig.update_xaxes(title_text="(T_in - T_amb) / G  (m2K/W)", row=1, col=1)
    fig.update_xaxes(title_text="T_inlet (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Incidence Angle (deg)", row=2, col=1)
    fig.update_xaxes(title_text="DeltaT = T_m - T_amb (K)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="IAM", row=2, col=1)
    fig.update_yaxes(title_text="U_L (W/m2K)", row=2, col=2)

    fig.update_layout(title="EC059 ETC — F1b Thermal Loss Curve + IAM Simulation", height=700)
    fig.write_html(str(output_path))
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    model = ComponentModel()
    generate_report(model, Path(__file__).parent.parent / "simulation_report.html")
