"""EC085 — Natural Gas Boiler — F1b Part-Load — Simulation & HTML Report"""
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
            "Efficiency vs PLR",
            "Fuel Input vs PLR",
            "Flue Gas Loss vs PLR",
            "Loss Breakdown at Various PLR",
        ],
        vertical_spacing=0.12,
    )

    plr = np.linspace(0.1, 1.0, 100)
    r = model.predict({"PLR": plr})

    # Row 1 Col 1 — Efficiency curve
    fig.add_trace(
        go.Scatter(x=plr, y=r["efficiency"], name="Efficiency",
                   line=dict(color="steelblue")),
        row=1, col=1,
    )

    # Row 1 Col 2 — Fuel input
    fig.add_trace(
        go.Scatter(x=plr, y=r["fuel_input_kw"], name="Fuel Input",
                   line=dict(color="darkorange"), showlegend=False),
        row=1, col=2,
    )

    # Row 2 Col 1 — Flue loss
    fig.add_trace(
        go.Scatter(x=plr, y=r["flue_loss_kw"], name="Flue Loss",
                   line=dict(color="crimson"), showlegend=False),
        row=2, col=1,
    )

    # Row 2 Col 2 — Stacked loss breakdown
    plr_bar = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    r_bar = model.predict({"PLR": plr_bar})
    fig.add_trace(
        go.Bar(x=plr_bar, y=r_bar["heat_output_kw"], name="Useful Heat"),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(x=plr_bar, y=r_bar["flue_loss_kw"], name="Flue Loss"),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(x=plr_bar, y=r_bar["standby_loss_kw"], name="Standby Loss"),
        row=2, col=2,
    )
    fig.update_layout(barmode="stack")

    fig.update_xaxes(title_text="PLR", row=1, col=1)
    fig.update_xaxes(title_text="PLR", row=1, col=2)
    fig.update_xaxes(title_text="PLR", row=2, col=1)
    fig.update_xaxes(title_text="PLR", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=2)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | Part-Load + Flue + Standby",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
