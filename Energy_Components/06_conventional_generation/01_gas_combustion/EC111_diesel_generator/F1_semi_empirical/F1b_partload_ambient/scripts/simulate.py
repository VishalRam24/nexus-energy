"""EC111 -- Diesel Generator -- F1b Part-Load + Ambient -- Simulation & HTML Report"""
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
            "Efficiency vs PLR (various altitudes)",
            "Efficiency vs Ambient Temperature (full load)",
            "Power Output vs PLR (various conditions)",
            "SFC vs PLR (various altitudes)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    PLR = np.linspace(0.25, 1.0, 200)

    # Plot 1: Efficiency vs PLR at different altitudes
    for alt in [0, 1000, 2000, 3000, 4000]:
        r = model.predict({"PLR": PLR, "altitude_m": float(alt)})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["efficiency"],
                       name=f"Alt={alt}m", line=dict(width=2)),
            row=1, col=1,
        )

    # Plot 2: Efficiency vs temperature at full load
    T_range = np.linspace(-20, 55, 200)
    for alt in [0, 2000, 4000]:
        r = model.predict({"PLR": 1.0, "T_ambient": T_range, "altitude_m": float(alt)})
        fig.add_trace(
            go.Scatter(x=T_range, y=r["efficiency"],
                       name=f"FL Alt={alt}m", line=dict(width=2)),
            row=1, col=2,
        )
    fig.add_vline(x=40, row=1, col=2, line_dash="dot", line_color="red",
                  annotation_text="Derating starts 40C")

    # Plot 3: Power output vs PLR
    for T, alt, label in [(25, 0, "25C/0m"), (25, 3000, "25C/3000m"),
                           (45, 0, "45C/0m"), (45, 3000, "45C/3000m")]:
        r = model.predict({"PLR": PLR, "T_ambient": float(T), "altitude_m": float(alt)})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["power_output_kw"],
                       name=label, line=dict(width=2)),
            row=2, col=1,
        )

    # Plot 4: SFC vs PLR
    for alt in [0, 2000, 4000]:
        r = model.predict({"PLR": PLR, "altitude_m": float(alt)})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["sfc_g_kwh"],
                       name=f"SFC Alt={alt}m", line=dict(width=2)),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Ambient Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Power Output (kW)", row=2, col=1)
    fig.update_yaxes(title_text="SFC (g/kWh)", row=2, col=2)

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
