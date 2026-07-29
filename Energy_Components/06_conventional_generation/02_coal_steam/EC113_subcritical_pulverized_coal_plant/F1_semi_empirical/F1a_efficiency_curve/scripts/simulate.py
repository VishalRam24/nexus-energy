"""EC113 — Subcritical Pulverized Coal Plant — F1a — Simulation & HTML Report"""
import sys, numpy as np
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
            "CO2 Intensity vs Part-Load Ratio",
            "Coal Feed Rate vs PLR",
            "Efficiency vs Ambient Temperature",
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.12,
    )

    plr = np.linspace(0.30, 1.0, 100)

    # Row 1 Col 1 — efficiency vs PLR at several T_amb
    for T_amb in [0, 15, 30, 45]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=plr, y=r["efficiency"] * 100, name=f"T_amb={T_amb} C"),
            row=1, col=1,
        )

    # Row 1 Col 2 — CO2 intensity vs PLR
    for T_amb in [0, 15, 45]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=plr, y=r["co2_intensity"], name=f"CO2 T_amb={T_amb}C", showlegend=False),
            row=1, col=2,
        )
    # Reference band for subcritical coal 900-1000 g/kWh
    fig.add_hrect(y0=900, y1=1000, row=1, col=2,
                  fillcolor="rgba(255,165,0,0.15)", line_width=0,
                  annotation_text="Typical 900-1000 g/kWh", annotation_position="top right")

    # Row 2 Col 1 — coal feed rate vs PLR
    for T_amb in [0, 15, 45]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=plr, y=r["coal_rate_kgs"], name=f"coal T={T_amb}C", showlegend=False),
            row=2, col=1,
        )

    # Row 2 Col 2 — efficiency vs T_amb at several PLR values
    T_amb_range = np.linspace(-10, 45, 100)
    for plr_val in [0.30, 0.50, 0.75, 1.0]:
        r = model.predict({"part_load_ratio": float(plr_val), "ambient_temp": T_amb_range})
        fig.add_trace(
            go.Scatter(x=T_amb_range, y=r["efficiency"] * 100, name=f"PLR={plr_val}"),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Ambient Temperature (degC)", row=2, col=2)

    fig.update_yaxes(title_text="Net LHV Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="CO2 Intensity (g/kWh)", row=1, col=2)
    fig.update_yaxes(title_text="Coal Feed Rate (kg/s)", row=2, col=1)
    fig.update_yaxes(title_text="Net LHV Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 500 MW Subcritical PC",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    r_iso = model.predict({"part_load_ratio": 1.0, "ambient_temp": 15.0})
    print(f"\nISO rated: eta={float(r_iso['efficiency'])*100:.1f}%  "
          f"coal={float(r_iso['coal_rate_kgs']):.1f} kg/s  "
          f"CO2={float(r_iso['co2_intensity']):.0f} g/kWh")


if __name__ == "__main__":
    generate_report()
