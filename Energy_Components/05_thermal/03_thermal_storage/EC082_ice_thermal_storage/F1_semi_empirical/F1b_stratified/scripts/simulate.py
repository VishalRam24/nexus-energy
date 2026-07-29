"""EC082 — Ice TES — F1b Stratified — Simulation Scenarios"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

BASE = Path(__file__).parent.parent


def run():
    model = ComponentModel()
    N = model._model.N

    # Scenario 1: Full charge cycle
    r_ch = model.predict({"q_charge_W": 80000.0, "q_discharge_W": 0.0,
                          "t_ambient": 20.0, "duration_s": 6 * 3600,
                          "f_initial": 0.0, "dt": 120.0})

    # Scenario 2: Discharge from charged state
    f_half = np.zeros(N)
    f_half[:N // 2] = 1.0   # bottom half fully frozen
    r_dis = model.predict({"q_charge_W": 0.0, "q_discharge_W": 120000.0,
                           "t_ambient": 20.0, "duration_s": 3 * 3600,
                           "f_initial": f_half, "dt": 120.0})

    print("=== EC082 Ice TES F1b — Simulation Summary ===")
    print("\n[Charge 6h at 80 kW]:")
    print(f"  Final SOC: {r_ch['soc']:.3f}")
    print(f"  Node fracs (bottom→top): {np.round(r_ch['f_nodes'], 3)}")
    print(f"  Strat index: {r_ch['stratification_index']:.3f}")
    print(f"  Heat loss: {r_ch['Q_loss_kw']:.3f} kW")

    print("\n[Discharge 3h at 120 kW from half-charged state]:")
    print(f"  Initial SOC: 0.50")
    print(f"  Final SOC: {r_dis['soc']:.3f}")
    print(f"  Node fracs (bottom→top): {np.round(r_dis['f_nodes'], 3)}")
    print(f"  Strat index: {r_dis['stratification_index']:.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Time series of SOC during charge
        f_hist = r_ch["f_history"]
        t_h = np.arange(len(f_hist)) * 120.0 / 3600.0
        soc_ts = np.mean(f_hist, axis=1)

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["SOC vs Time (Charge)",
                                            "Node Ice Fractions (end of charge)",
                                            "Node Ice Fractions (end of discharge)"])
        fig.add_trace(go.Scatter(x=t_h, y=soc_ts, mode="lines", name="SOC"),
                      row=1, col=1)

        node_labels = [f"N{i+1}" for i in range(N)]
        fig.add_trace(go.Bar(x=node_labels, y=r_ch["f_nodes"], name="Charge end"),
                      row=1, col=2)
        fig.add_trace(go.Bar(x=node_labels, y=r_dis["f_nodes"], name="Discharge end"),
                      row=1, col=3)

        fig.update_layout(title="EC082 Ice TES — F1b Stratified", height=450)
        fig.update_xaxes(title_text="Time [h]", row=1, col=1)
        fig.update_xaxes(title_text="Node (bottom→top)", row=1, col=2)
        fig.update_xaxes(title_text="Node (bottom→top)", row=1, col=3)
        fig.update_yaxes(title_text="SOC [-]", row=1, col=1)
        fig.update_yaxes(title_text="Ice fraction [-]", row=1, col=2)

        html_path = BASE / "simulation_report.html"
        fig.write_html(str(html_path))
        print(f"\nReport written to {html_path}")
    except ImportError:
        print("plotly not available — skipping HTML report")


if __name__ == "__main__":
    run()
