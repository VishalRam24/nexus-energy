"""EC184 — PFC Unit — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Achieved PF vs Initial PF (P=2000 kW, target=0.95)",
            "Q_compensated vs Load Power (pf_initial=0.75, target=0.95)",
            "Bank Utilization vs Initial PF",
            "Capacitor Losses vs Load",
        ],
        vertical_spacing=0.14,
    )

    pf_range = np.linspace(0.5, 0.94, 100)
    r1 = model.predict({"P_kW": 2000.0, "pf_initial": pf_range, "pf_target": 0.95})
    fig.add_trace(go.Scatter(x=pf_range, y=r1["pf_achieved"],
                             name="pf_achieved", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                  annotation_text="target 0.95", row=1, col=1)

    P_range = np.linspace(100, 5000, 100)
    r2 = model.predict({"P_kW": P_range, "pf_initial": 0.75, "pf_target": 0.95})
    fig.add_trace(go.Scatter(x=P_range, y=r2["Q_compensated_kVAR"],
                             name="Q_comp (kVAR)", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)
    fig.add_hline(y=model._model.Q_rated_kVAR, line_dash="dash", line_color="red",
                  annotation_text="Q_rated", row=1, col=2)

    fig.add_trace(go.Scatter(x=pf_range, y=r1["bank_utilization"] * 100,
                             name="Utilization (%)", line=dict(color="#00CC96", width=2)),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=P_range, y=r2["P_loss_kW"],
                             name="P_loss (kW)", line=dict(color="#AB63FA", width=2)),
                  row=2, col=2)

    fig.update_xaxes(title_text="Initial PF", row=1, col=1)
    fig.update_xaxes(title_text="Load P (kW)", row=1, col=2)
    fig.update_xaxes(title_text="Initial PF", row=2, col=1)
    fig.update_xaxes(title_text="Load P (kW)", row=2, col=2)
    fig.update_yaxes(title_text="Achieved PF", row=1, col=1)
    fig.update_yaxes(title_text="Q_comp (kVAR)", row=1, col=2)
    fig.update_yaxes(title_text="Bank Utilization (%)", row=2, col=1)
    fig.update_yaxes(title_text="P_loss (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- PFC Unit Summary (P=2000 kW, target pf=0.95) ---")
    print(f"{'pf_initial':>12} {'Q_load(kVAR)':>14} {'Q_comp(kVAR)':>14} {'pf_achieved':>12} {'P_loss(kW)':>11}")
    for pf0 in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        rv = model.predict({"P_kW": 2000.0, "pf_initial": pf0, "pf_target": 0.95})
        print(f"{pf0:>12.2f} {float(rv['Q_load_kVAR']):>14.1f} {float(rv['Q_compensated_kVAR']):>14.1f} "
              f"{float(rv['pf_achieved']):>12.4f} {float(rv['P_loss_kW']):>11.3f}")


if __name__ == "__main__":
    generate_report()
