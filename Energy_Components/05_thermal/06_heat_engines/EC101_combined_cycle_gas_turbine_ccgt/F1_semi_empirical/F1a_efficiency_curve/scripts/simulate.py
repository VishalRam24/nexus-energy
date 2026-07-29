"""EC101 — CCGT — F1a — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info  = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs Part-Load Ratio",
            "Efficiency vs Ambient Temperature",
            "Fuel Rate vs PLR",
            "Exhaust Temperature vs PLR",
        ],
        vertical_spacing=0.13,
    )

    plr = np.linspace(0.3, 1.0, 100)

    # Row 1 Col 1 — efficiency vs PLR at several T_amb
    for T_amb in [-10, 0, 15, 30, 45]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=plr, y=r["efficiency"], name=f"T_amb={T_amb} C"),
            row=1, col=1,
        )

    # Row 1 Col 2 — efficiency vs T_amb at several PLR values
    T_amb_range = np.linspace(-20, 50, 100)
    for plr_val in [0.3, 0.5, 0.7, 1.0]:
        r = model.predict({"part_load_ratio": float(plr_val), "ambient_temp": T_amb_range})
        fig.add_trace(
            go.Scatter(x=T_amb_range, y=r["efficiency"], name=f"PLR={plr_val}"),
            row=1, col=2,
        )

    # Row 2 Col 1 — fuel rate vs PLR at several T_amb
    for T_amb in [0, 15, 35]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=plr, y=r["fuel_rate_kgs"], name=f"fuel T_amb={T_amb}C", showlegend=False),
            row=2, col=1,
        )

    # Row 2 Col 2 — exhaust temperature vs PLR
    r = model.predict({"part_load_ratio": plr, "ambient_temp": 15.0})
    fig.add_trace(
        go.Scatter(x=plr, y=r["exhaust_temp_c"], name="Exhaust Temp", line=dict(color="firebrick"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Ambient Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)

    fig.update_yaxes(title_text="Net LHV Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Net LHV Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Fuel Rate (kg/s)", row=2, col=1)
    fig.update_yaxes(title_text="Exhaust Temperature (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 571 MW CCGT",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
