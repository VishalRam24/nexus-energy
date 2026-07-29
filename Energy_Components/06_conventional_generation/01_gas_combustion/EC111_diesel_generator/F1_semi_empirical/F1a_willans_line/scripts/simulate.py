"""EC111 — Diesel Generator — F1a Willans Line — Simulation & HTML Report"""
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
            "Willans Line: Fuel Rate vs Power Output",
            "SFC vs Power Output (Load Curve)",
            "Generator Efficiency vs Load",
            "CO2 Emissions vs Power Output",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    P = np.linspace(0, 500, 200)

    # Plot 1: Willans line at different ambient temperatures
    for T_amb in [15, 25, 35, 45]:
        r = model.predict({"power_output_kw": P, "ambient_temp_c": T_amb})
        fig.add_trace(
            go.Scatter(x=P, y=r["fuel_rate_lph"], name=f"T_amb={T_amb}°C",
                       line=dict(width=2)),
            row=1, col=1,
        )

    # Plot 2: SFC curve (load characteristic)
    P_load = np.linspace(130, 500, 150)
    for T_amb in [25, 40]:
        r = model.predict({"power_output_kw": P_load, "ambient_temp_c": T_amb})
        sfc = r["sfc_gkwh"]
        valid = ~np.isnan(sfc)
        fig.add_trace(
            go.Scatter(x=P_load[valid], y=sfc[valid], name=f"SFC T={T_amb}°C",
                       line=dict(width=2)),
            row=1, col=2,
        )

    # Add reference SFC lines
    fig.add_hline(y=210, row=1, col=2, line_dash="dash", line_color="red",
                  annotation_text="Rated SFC=210 g/kWh")

    # Plot 3: Efficiency vs load
    P_eff = np.linspace(130, 500, 150)
    for T_amb in [15, 25, 40]:
        r = model.predict({"power_output_kw": P_eff, "ambient_temp_c": T_amb})
        fig.add_trace(
            go.Scatter(x=P_eff / 500 * 100, y=r["efficiency"] * 100,
                       name=f"η T={T_amb}°C", line=dict(width=2)),
            row=2, col=1,
        )
    fig.add_hline(y=45, row=2, col=1, line_dash="dash", line_color="red",
                  annotation_text="Max η = 45%")

    # Plot 4: CO2 vs power output at rated conditions
    r_co2 = model.predict({"power_output_kw": P, "ambient_temp_c": 25.0})
    fig.add_trace(
        go.Scatter(x=P, y=r_co2["co2_emissions_kgh"], name="CO2 (25°C)",
                   fill="tozeroy", line=dict(color="crimson", width=2)),
        row=2, col=2,
    )
    # Add specific CO2 intensity on second y-axis via annotation
    r_sco2 = model.predict({"power_output_kw": 500.0})
    co2_intensity = float(r_sco2["co2_emissions_kgh"]) / 500.0 * 1000  # g/kWh
    fig.add_annotation(
        x=400, y=float(r_co2["co2_emissions_kgh"][-1]) * 0.9,
        text=f"CO2 intensity ≈ {co2_intensity:.0f} g/kWh @ rated",
        showarrow=False, row=2, col=2,
    )

    # Axes labels
    fig.update_xaxes(title_text="Power Output (kW)", row=1, col=1)
    fig.update_xaxes(title_text="Power Output (kW)", row=1, col=2)
    fig.update_xaxes(title_text="Load (%)", row=2, col=1)
    fig.update_xaxes(title_text="Power Output (kW)", row=2, col=2)
    fig.update_yaxes(title_text="Fuel Rate (L/h)", row=1, col=1)
    fig.update_yaxes(title_text="SFC (g/kWh)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="CO2 Emissions (kg/h)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Willans Line",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
