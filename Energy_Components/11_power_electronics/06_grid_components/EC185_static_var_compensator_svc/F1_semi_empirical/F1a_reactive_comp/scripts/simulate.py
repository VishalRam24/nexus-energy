"""EC185 — SVC — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    Q_dem_range = np.linspace(-80, 130, 300)
    r = model.predict({"Q_demand_MVAR": Q_dem_range})

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Q_out vs Q_demand",
            "P_loss vs Q_demand",
            "Q_out (time-series example: voltage sag/swell)",
            "Utilization vs Q_demand",
        ],
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(x=Q_dem_range, y=r["Q_out_MVAR"],
                             name="Q_out", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)
    fig.add_hline(y=model._model.Q_max, line_dash="dash", line_color="red",
                  annotation_text=f"Q_max={model._model.Q_max}", row=1, col=1)
    fig.add_hline(y=model._model.Q_min, line_dash="dash", line_color="blue",
                  annotation_text=f"Q_min={model._model.Q_min}", row=1, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=Q_dem_range, y=r["P_loss_MW"],
                             name="P_loss (MW)", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)

    # Simulated voltage event: load steps requiring reactive support
    t = np.arange(0, 200, 1)
    Q_event = np.where(t < 50, 0, np.where(t < 100, 80, np.where(t < 150, -40, 20)))
    r_event = model.predict({"Q_demand_MVAR": Q_event.astype(float)})
    fig.add_trace(go.Scatter(x=t, y=Q_event, name="Q_demand", line=dict(dash="dash", color="gray")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r_event["Q_out_MVAR"],
                             name="Q_out (actual)", line=dict(color="#00CC96", width=2)),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=Q_dem_range, y=r["utilization"],
                             name="utilization", line=dict(color="#AB63FA", width=2)),
                  row=2, col=2)

    fig.update_xaxes(title_text="Q_demand (MVAR)", row=1, col=1)
    fig.update_xaxes(title_text="Q_demand (MVAR)", row=1, col=2)
    fig.update_xaxes(title_text="Time step", row=2, col=1)
    fig.update_xaxes(title_text="Q_demand (MVAR)", row=2, col=2)
    fig.update_yaxes(title_text="Q_out (MVAR)", row=1, col=1)
    fig.update_yaxes(title_text="P_loss (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Q (MVAR)", row=2, col=1)
    fig.update_yaxes(title_text="Utilization", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- SVC Summary ---")
    print(f"{'Q_demand':>12} {'Q_out':>9} {'P_loss_MW':>11} {'mode':>12} {'limited':>8}")
    for Qd in [-60, -50, -20, 0, 50, 80, 100, 120]:
        rv = model.predict({"Q_demand_MVAR": float(Qd)})
        print(f"{Qd:>12} {float(rv['Q_out_MVAR']):>9.1f} {float(rv['P_loss_MW']):>11.3f} "
              f"{str(rv['operating_mode']):>12} {str(bool(rv['Q_limited'])):>8}")


if __name__ == "__main__":
    generate_report()
