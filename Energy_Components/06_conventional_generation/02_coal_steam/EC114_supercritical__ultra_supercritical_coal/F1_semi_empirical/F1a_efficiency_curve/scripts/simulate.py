"""EC114 — Supercritical / Ultra-Supercritical Coal Plant — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info  = model.get_info()
    m     = model._model

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
    # Add Carnot efficiency as upper bound
    eta_carnot = m.carnot_efficiency(T_cond_c=40.0)
    fig.add_hline(y=eta_carnot * 100, row=1, col=1,
                  line_dash="dot", line_color="gray",
                  annotation_text=f"Carnot {eta_carnot*100:.1f}%")

    # Row 1 Col 2 — CO2 intensity vs PLR (with reference bands)
    for T_amb in [0, 15, 45]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=plr, y=r["co2_intensity"], name=f"CO2 T={T_amb}C", showlegend=False),
            row=1, col=2,
        )
    # SC/USC typical band
    fig.add_hrect(y0=750, y1=850, row=1, col=2,
                  fillcolor="rgba(0,200,0,0.12)", line_width=0,
                  annotation_text="SC/USC typical 750-850 g/kWh", annotation_position="top right")

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
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 800 MW USC",
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
    print(f"Carnot limit (700C steam, 40C cond): {eta_carnot*100:.1f}%")


if __name__ == "__main__":
    generate_report()
