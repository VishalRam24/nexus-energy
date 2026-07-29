"""EC065 — Offshore Fixed-Bottom Wind Turbine — F1a Power Curve — Simulation & HTML Report"""
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
            "Power Curve (Standard & Off-Design Density)",
            "Capacity Factor vs Wind Speed",
            "Power Coefficient (Cp) vs Wind Speed",
            "Annual Energy Production Estimate (Rayleigh Distribution)",
        ],
        vertical_spacing=0.14,
    )

    v = np.linspace(0, 30, 500)

    # Plot 1: Power curve at multiple densities
    for rho, label, color in [
        (1.05, "rho=1.05 (warm)", "red"),
        (1.225, "rho=1.225 (STC)", "blue"),
        (1.35, "rho=1.35 (cold)", "green"),
    ]:
        r = model.predict({"wind_speed": v, "air_density": rho})
        fig.add_trace(
            go.Scatter(x=v, y=r["power_kw"], name=label, line=dict(color=color)),
            row=1, col=1,
        )

    # Add cut-in and cut-out lines
    for vx, label in [(3.5, "cut-in 3.5m/s"), (25.0, "cut-out 25m/s")]:
        fig.add_vline(x=vx, line_dash="dot", line_color="gray", row=1, col=1)

    # Plot 2: Capacity factor
    r_ref = model.predict({"wind_speed": v})
    fig.add_trace(
        go.Scatter(x=v, y=r_ref["capacity_factor"] * 100, name="CF (%)", showlegend=False),
        row=1, col=2,
    )

    # Plot 3: Cp vs wind speed
    v_op = np.linspace(0.5, 30, 300)
    r_op = model.predict({"wind_speed": v_op})
    fig.add_trace(
        go.Scatter(x=v_op, y=r_op["power_coefficient"], name="Cp", showlegend=False),
        row=2, col=1,
    )
    # Betz limit reference
    betz = 16.0 / 27.0
    fig.add_hline(y=betz, line_dash="dash", line_color="red",
                  annotation_text=f"Betz limit={betz:.3f}", row=2, col=1)

    # Plot 4: AEP estimation with Rayleigh wind distribution
    v_range = np.linspace(0, 30, 1000)
    for v_mean, label in [(7.0, "v_mean=7m/s"), (8.5, "v_mean=8.5m/s"), (10.0, "v_mean=10m/s")]:
        # Rayleigh PDF: f(v) = (pi/2) * (v/v_mean^2) * exp(-pi/4 * (v/v_mean)^2)
        pdf = (np.pi / 2) * (v_range / v_mean ** 2) * np.exp(-np.pi / 4 * (v_range / v_mean) ** 2)
        r_v = model.predict({"wind_speed": v_range})
        aep_kw = np.trapezoid(r_v["power_kw"] * pdf, v_range) * 8760  # kWh/yr
        fig.add_trace(
            go.Scatter(x=v_range, y=r_v["power_kw"] * pdf * 8760,
                       name=f"{label}: AEP={aep_kw/1e6:.1f} GWh/yr"),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Capacity Factor (%)", row=1, col=2)
    fig.update_yaxes(title_text="Power Coefficient Cp (-)", row=2, col=1)
    fig.update_yaxes(title_text="Weighted Power (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Power Curve",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
