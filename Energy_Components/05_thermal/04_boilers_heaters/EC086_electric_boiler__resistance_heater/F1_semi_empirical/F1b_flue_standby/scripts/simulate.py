"""EC086 — Electric Boiler F1b — Simulation Scenarios + HTML Report"""
import json, sys, numpy as np
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
    results = {}

    # 1. PLR sweep at design ambient
    plr = np.linspace(0.05, 1.0, 80)
    r = model.predict({"PLR": plr})
    results["plr_sweep"] = {"PLR": plr, **{k: np.asarray(v) for k, v in r.items()}}

    # 2. Ambient temperature sensitivity at PLR=0.3
    T_ambs = np.linspace(-20, 40, 60)
    r2 = model.predict({"PLR": 0.3, "T_ambient": T_ambs})
    results["ambient_sweep"] = {"T_ambient": T_ambs, **{k: np.asarray(v) for k, v in r2.items()}}

    # 3. Standby loss vs dT (T_fluid - T_ambient)
    dT = np.linspace(0, 80, 80)
    T_fl = 70.0 + dT * 0.0
    T_am = 70.0 - dT
    r3 = model.predict({"PLR": 0.5, "T_ambient": T_am, "T_fluid": np.full(80, 70.0)})
    results["standby_sweep"] = {"dT": dT, **{k: np.asarray(v) for k, v in r3.items()}}

    return results


def generate_html_report(results, output_path):
    if not HAS_PLOTLY:
        print("plotly not available — skipping HTML report")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs PLR (design ambient)",
            "Heat Output & Standby vs PLR",
            "Ambient Sensitivity (PLR=0.3)",
            "Standby Loss vs Temperature Differential",
        ]
    )

    ps = results["plr_sweep"]
    am = results["ambient_sweep"]
    sb = results["standby_sweep"]

    fig.add_trace(go.Scatter(x=ps["PLR"], y=ps["efficiency"], name="eta_eff", line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ps["PLR"], y=ps["heat_output_kw"], name="Q_out [kW]", line=dict(color="tomato")), row=1, col=2)
    fig.add_trace(go.Scatter(x=ps["PLR"], y=ps["standby_loss_kw"], name="Q_standby [kW]", line=dict(color="orange", dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=am["T_ambient"], y=am["efficiency"], name="eta (PLR=0.3)", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=sb["dT"], y=sb["standby_loss_kw"], name="Q_standby [kW]", line=dict(color="darkorange")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=1, col=2)
    fig.update_xaxes(title_text="T_ambient [degC]", row=2, col=1)
    fig.update_xaxes(title_text="T_fluid - T_ambient [K]", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=2, col=1)
    fig.update_yaxes(title_text="Standby Loss [kW]", row=2, col=2)

    fig.update_layout(title_text="EC086 Electric Boiler F1b — Standby Loss + Ambient", height=700)
    fig.write_html(str(output_path))
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    results = run_simulation()
    out = Path(__file__).parent.parent / "simulation_report.html"
    generate_html_report(results, out)
    ps = results["plr_sweep"]
    print(f"PLR=1.0: eta={ps['efficiency'][-1]:.4f}, Q_out={ps['heat_output_kw'][-1]:.2f} kW")
    print(f"PLR=0.1: eta={ps['efficiency'][0]:.4f}, Q_sb={ps['standby_loss_kw'][0]:.3f} kW")
