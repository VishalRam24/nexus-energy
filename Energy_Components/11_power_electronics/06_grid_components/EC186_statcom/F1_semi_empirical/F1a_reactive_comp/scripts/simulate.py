"""EC186 — STATCOM — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    Q_dem = np.linspace(-130, 130, 300)
    r = model.predict({"Q_demand_MVAR": Q_dem})

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Q_out vs Q_demand (symmetric range)",
            "Total Losses vs Q_demand",
            "Loss Breakdown (standby vs variable)",
            "Utilization vs Q_demand",
        ],
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(x=Q_dem, y=r["Q_out_MVAR"],
                             name="Q_out", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)
    fig.add_hline(y=model._model.Q_max, line_dash="dash", line_color="red",
                  annotation_text=f"+{model._model.Q_max} MVAR", row=1, col=1)
    fig.add_hline(y=model._model.Q_min, line_dash="dash", line_color="blue",
                  annotation_text=f"{model._model.Q_min} MVAR", row=1, col=1)

    fig.add_trace(go.Scatter(x=Q_dem, y=r["P_total_loss_MW"],
                             name="P_total_loss", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=Q_dem, y=r["P_standby_MW"],
                             name="P_standby", line=dict(color="#FFA15A")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=Q_dem, y=r["P_loss_MW"],
                             name="P_variable", line=dict(color="#EF553B")),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=Q_dem, y=r["utilization"] * 100,
                             name="utilization (%)", line=dict(color="#AB63FA", width=2)),
                  row=2, col=2)

    for ax_r, ax_c, xt, yt in [
        (1, 1, "Q_demand (MVAR)", "Q_out (MVAR)"),
        (1, 2, "Q_demand (MVAR)", "P_total_loss (MW)"),
        (2, 1, "Q_demand (MVAR)", "P_loss (MW)"),
        (2, 2, "Q_demand (MVAR)", "Utilization (%)"),
    ]:
        fig.update_xaxes(title_text=xt, row=ax_r, col=ax_c)
        fig.update_yaxes(title_text=yt, row=ax_r, col=ax_c)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- STATCOM Summary ---")
    print(f"{'Q_demand':>12} {'Q_out':>9} {'P_var(MW)':>11} {'P_total(MW)':>12}")
    for Qd in [-120, -80, -40, 0, 40, 80, 100, 120]:
        rv = model.predict({"Q_demand_MVAR": float(Qd)})
        print(f"{Qd:>12} {float(rv['Q_out_MVAR']):>9.1f} {float(rv['P_loss_MW']):>11.3f} {float(rv['P_total_loss_MW']):>12.3f}")


if __name__ == "__main__":
    generate_report()
