"""EC109 -- Simple Cycle Gas Turbine -- F1b Part-Load + Ambient -- Simulation & HTML Report"""
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
            "Efficiency vs PLR (at various ambient temperatures)",
            "Efficiency vs Ambient Temperature (at full load)",
            "Power Output vs PLR (at various ambient conditions)",
            "Heat Rate vs PLR",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    PLR = np.linspace(0.3, 1.0, 200)

    # Plot 1: Efficiency vs PLR at different ambient temps
    for T_c in [-20, 0, 15, 30, 45]:
        T_k = T_c + 273.15
        r = model.predict({"PLR": PLR, "T_ambient": T_k})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["efficiency"],
                       name=f"T={T_c} degC", line=dict(width=2)),
            row=1, col=1,
        )

    # Plot 2: Efficiency vs ambient temperature at full load
    T_range_c = np.linspace(-20, 50, 200)
    T_range_k = T_range_c + 273.15
    for plr_val in [0.5, 0.75, 1.0]:
        r = model.predict({"PLR": plr_val, "T_ambient": T_range_k})
        fig.add_trace(
            go.Scatter(x=T_range_c, y=r["efficiency"],
                       name=f"PLR={plr_val}", line=dict(width=2)),
            row=1, col=2,
        )
    fig.add_vline(x=15, row=1, col=2, line_dash="dot", line_color="gray",
                  annotation_text="ISO 15 degC")

    # Plot 3: Power output vs PLR at different ambient conditions
    for T_c, P_kpa, label in [(-10, 101.3, "-10C/101kPa"),
                                (15, 101.3, "15C/101kPa (ISO)"),
                                (40, 101.3, "40C/101kPa"),
                                (15, 90.0, "15C/90kPa (altitude)")]:
        T_k = T_c + 273.15
        r = model.predict({"PLR": PLR, "T_ambient": T_k, "P_ambient": P_kpa})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["power_output_kw"] / 1e3,
                       name=label, line=dict(width=2)),
            row=2, col=1,
        )

    # Plot 4: Heat rate vs PLR
    for T_c in [-10, 15, 40]:
        T_k = T_c + 273.15
        r = model.predict({"PLR": PLR, "T_ambient": T_k})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["heat_rate_kj_kwh"],
                       name=f"HR T={T_c} degC", line=dict(width=2)),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Ambient Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Power Output (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Heat Rate (kJ/kWh)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Part-Load + Ambient",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
