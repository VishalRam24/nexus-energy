"""EC140 — Anaerobic Digester (Mesophilic) — F1a Yield Model — Simulation & HTML Report"""
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
            "Methane Yield vs HRT (at various VS loadings)",
            "Temperature Effect on Methane Yield (HRT=20 days)",
            "Biogas & Methane Rate vs VS Loading",
            "Energy Output Heatmap (HRT × VS Loading)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Plot 1: Methane yield vs HRT at different VS loadings (yield is independent of VS)
    hrt_range = np.linspace(5, 40, 200)
    for T in [25, 30, 37, 42, 50]:
        r = model.predict({"vs_loading": 3.0, "hrt": hrt_range, "temperature": float(T)})
        fig.add_trace(
            go.Scatter(x=hrt_range, y=r["methane_yield_m3kgvs"],
                       name=f"T={T}°C", line=dict(width=2)),
            row=1, col=1,
        )
    fig.add_hline(y=0.35, row=1, col=1, line_dash="dash", line_color="gray",
                  annotation_text="Y_max = 0.35 m³/kgVS")

    # Plot 2: Temperature effect on yield at HRT=20 days
    T_range = np.linspace(25, 55, 200)
    for hrt_val in [10, 20, 30]:
        r = model.predict({"vs_loading": 3.0, "hrt": float(hrt_val), "temperature": T_range})
        fig.add_trace(
            go.Scatter(x=T_range, y=r["methane_yield_m3kgvs"],
                       name=f"HRT={hrt_val}d", line=dict(width=2)),
            row=1, col=2,
        )
    fig.add_vline(x=37, row=1, col=2, line_dash="dot", line_color="green",
                  annotation_text="37°C optimum")
    fig.add_vline(x=42, row=1, col=2, line_dash="dot", line_color="orange",
                  annotation_text="42°C inhibition")

    # Plot 3: Biogas and methane rate vs VS loading at design HRT and temperature
    vs_range = np.linspace(1, 8, 200)
    r_bg = model.predict({"vs_loading": vs_range, "hrt": 20.0, "temperature": 37.0})
    fig.add_trace(
        go.Scatter(x=vs_range, y=r_bg["biogas_rate_m3day"], name="Biogas (m³/day)",
                   line=dict(color="royalblue", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=vs_range, y=r_bg["methane_rate_m3day"], name="Methane (m³/day)",
                   line=dict(color="green", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=vs_range, y=r_bg["energy_output_kwh_day"] / 24,
                   name="Power (kW_th)", line=dict(color="orange", width=2, dash="dash")),
        row=2, col=1,
    )

    # Plot 4: Energy output heatmap — HRT × VS Loading
    HRT_grid = np.linspace(5, 40, 50)
    VS_grid = np.linspace(1, 8, 50)
    E_map = np.zeros((50, 50))
    for i, vs in enumerate(VS_grid):
        r_map = model.predict({"vs_loading": float(vs), "hrt": HRT_grid, "temperature": 37.0})
        E_map[i, :] = r_map["energy_output_kwh_day"]

    fig.add_trace(
        go.Heatmap(
            x=HRT_grid, y=VS_grid, z=E_map,
            colorscale="Greens", name="Energy (kWh/day)",
            colorbar=dict(title="kWh/day", x=1.02),
        ),
        row=2, col=2,
    )
    # Mark design point
    fig.add_trace(
        go.Scatter(x=[20], y=[3], mode="markers",
                   marker=dict(symbol="star", size=12, color="red"),
                   name="Design point", showlegend=True),
        row=2, col=2,
    )

    # Axes labels
    fig.update_xaxes(title_text="HRT (days)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="VS Loading (kgVS/m³/day)", row=2, col=1)
    fig.update_xaxes(title_text="HRT (days)", row=2, col=2)
    fig.update_yaxes(title_text="Methane Yield (m³_CH4/kgVS)", row=1, col=1)
    fig.update_yaxes(title_text="Methane Yield (m³_CH4/kgVS)", row=1, col=2)
    fig.update_yaxes(title_text="Rate (m³/day) / Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="VS Loading (kgVS/m³/day)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Biogas Yield Model",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
