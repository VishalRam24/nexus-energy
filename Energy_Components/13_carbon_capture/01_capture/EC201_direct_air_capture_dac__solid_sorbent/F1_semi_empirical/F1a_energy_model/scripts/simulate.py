"""EC201 — DAC Solid Sorbent — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "CO2 Captured vs Air Flow",
            "Specific Thermal Energy vs Humidity",
            "Annual Energy vs Air Flow (RH=0.5)",
            "Thermal Energy Map (Flow vs RH)",
        ],
        vertical_spacing=0.14)

    # Plot 1: CO2 captured vs air flow for several RH values
    flows = np.logspace(5, 7, 100)
    for rh in [0.2, 0.4, 0.6, 0.8]:
        r = model.predict({"air_flow_m3h": flows, "relative_humidity": rh})
        fig.add_trace(go.Scatter(x=flows/1e6, y=r["co2_captured_tpa"],
            name=f"RH={rh:.1f}"), row=1, col=1)

    # Plot 2: Specific thermal vs humidity
    rh_arr = np.linspace(0.1, 0.9, 100)
    r2 = model.predict({"air_flow_m3h": 1e6, "relative_humidity": rh_arr})
    fig.add_trace(go.Scatter(x=rh_arr, y=r2["specific_thermal_kwht"],
        name="E_th specific", line=dict(color="firebrick")), row=1, col=2)
    fig.add_trace(go.Scatter(x=rh_arr,
        y=np.full_like(rh_arr, float(np.atleast_1d(r2["specific_electric_kwhe"])[0])),
        name="E_el specific", line=dict(color="steelblue", dash="dash")), row=1, col=2)

    # Plot 3: Annual thermal + electrical vs air flow
    r3 = model.predict({"air_flow_m3h": flows, "relative_humidity": 0.5})
    fig.add_trace(go.Scatter(x=flows/1e6, y=r3["thermal_energy_mwh_pa"]/1e3,
        name="Thermal GWh/yr", line=dict(color="firebrick")), row=2, col=1)
    fig.add_trace(go.Scatter(x=flows/1e6, y=r3["electrical_energy_mwh_pa"]/1e3,
        name="Electrical GWh/yr", line=dict(color="steelblue")), row=2, col=1)

    # Plot 4: Heatmap — thermal MWh/yr vs flow and RH
    rh_grid = np.linspace(0.1, 0.9, 40)
    flow_grid = np.logspace(5, 7, 40)
    th_map = np.zeros((40, 40))
    for i, fl in enumerate(flow_grid):
        r4 = model.predict({"air_flow_m3h": fl, "relative_humidity": rh_grid})
        th_map[i, :] = r4["thermal_energy_mwh_pa"] / 1e3  # GWh/yr
    fig.add_trace(go.Heatmap(
        x=rh_grid, y=flow_grid/1e6, z=th_map,
        colorscale="Hot", colorbar=dict(title="GWh_th/yr"),
        name="Thermal GWh"), row=2, col=2)

    fig.update_xaxes(title_text="Air Flow (million m3/hr)", row=1, col=1)
    fig.update_xaxes(title_text="Relative Humidity (-)", row=1, col=2)
    fig.update_xaxes(title_text="Air Flow (million m3/hr)", row=2, col=1)
    fig.update_xaxes(title_text="Relative Humidity (-)", row=2, col=2)
    fig.update_yaxes(title_text="CO2 Captured (tCO2/yr)", row=1, col=1)
    fig.update_yaxes(title_text="kWh/tCO2", row=1, col=2)
    fig.update_yaxes(title_text="Annual Energy (GWh/yr)", row=2, col=1)
    fig.update_yaxes(title_text="Air Flow (million m3/hr)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
