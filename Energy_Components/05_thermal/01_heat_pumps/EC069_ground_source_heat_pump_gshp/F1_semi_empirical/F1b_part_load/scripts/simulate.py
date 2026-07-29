"""EC069 — GSHP — F1b Part-Load — Simulation & HTML Report"""
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
            "Seasonal Ground Temperature",
            "Monthly COP at various PLR (T_sink=35C)",
            "COP vs PLR at various T_ground",
            "Annual Electrical Input Profile",
        ],
        vertical_spacing=0.12,
    )

    months = np.linspace(1, 12, 120)

    # Row 1 Col 1 — Ground temperature over year
    r = model.predict({"month": months, "T_sink": 35.0})
    fig.add_trace(
        go.Scatter(x=months, y=r["T_ground"], name="T_ground",
                   line=dict(color="sienna")),
        row=1, col=1,
    )

    # Row 1 Col 2 — COP by month at different PLR
    months_int = np.arange(1, 13)
    for plr in [1.0, 0.75, 0.5, 0.25]:
        r = model.predict({"month": months_int, "T_sink": 35.0, "part_load_ratio": plr})
        fig.add_trace(
            go.Scatter(x=months_int, y=r["cop"], name=f"PLR={plr:.2f}"),
            row=1, col=2,
        )

    # Row 2 Col 1 — COP vs PLR at various T_ground
    plr_range = np.linspace(0.1, 1.0, 100)
    for tg in [5, 8, 10, 13, 15]:
        r = model.predict({"T_ground": tg, "T_sink": 35.0, "part_load_ratio": plr_range})
        fig.add_trace(
            go.Scatter(x=plr_range, y=r["cop"], name=f"T_gnd={tg}C", showlegend=False),
            row=2, col=1,
        )

    # Row 2 Col 2 — Annual electrical input
    for plr in [1.0, 0.7, 0.5]:
        r = model.predict({"month": months_int, "T_sink": 35.0, "part_load_ratio": plr})
        fig.add_trace(
            go.Scatter(x=months_int, y=r["electrical_input_kw"],
                       name=f"W PLR={plr}", showlegend=False),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_xaxes(title_text="PLR", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="T_ground (C)", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="COP", row=2, col=1)
    fig.update_yaxes(title_text="kW_e", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | Seasonal + Part-Load",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
