"""EC201 — DAC Solid Sorbent — F1b Part-Load Degradation — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    cycles = np.linspace(0, 15000, 50)
    PLRs = np.linspace(0.3, 1.0, 50)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Sorbent Capacity vs Cycles",
            "CO2 Captured vs Cycles",
            "Thermal Energy vs Ambient Temperature",
            "Electrical Energy vs PLR",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Capacity degradation
    for rh in [0.3, 0.5, 0.7]:
        cap = []
        for n in cycles:
            r = model.predict({"n_cycles": float(n), "relative_humidity": rh})
            cap.append(float(np.atleast_1d(r["sorbent_capacity_pct"])[0]))
        fig.add_trace(
            go.Scatter(x=cycles, y=cap, name=f"RH={rh}", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: CO2 captured vs cycles (degradation effect)
    for rh in [0.3, 0.5, 0.7]:
        co2 = []
        for n in cycles:
            r = model.predict({"air_flow_m3_s": 10.0, "n_cycles": float(n),
                               "relative_humidity": rh})
            co2.append(float(np.atleast_1d(r["co2_captured_kg_h"])[0]))
        fig.add_trace(
            go.Scatter(x=cycles, y=co2, name=f"CO2 RH={rh}", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Thermal energy vs T_ambient
    Ts = np.linspace(-10, 45, 50)
    for n_cyc in [0, 5000, 10000]:
        E_th = []
        for T in Ts:
            r = model.predict({"T_ambient_degC": float(T), "n_cycles": n_cyc})
            E_th.append(float(np.atleast_1d(r["thermal_energy_kwh_ton"])[0]))
        fig.add_trace(
            go.Scatter(x=Ts, y=E_th, name=f"Eth n={n_cyc}", line=dict(width=2)),
            row=2, col=1
        )

    # Plot 4: Electrical vs PLR
    for n_cyc in [0, 5000, 10000]:
        E_el = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "n_cycles": n_cyc})
            E_el.append(float(np.atleast_1d(r["electrical_energy_kwh_ton"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=E_el, name=f"Eel n={n_cyc}", line=dict(width=2)),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Cycles", row=1, col=1)
    fig.update_xaxes(title_text="Cycles", row=1, col=2)
    fig.update_xaxes(title_text="Ambient Temperature (degC)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="Sorbent Capacity (%)", row=1, col=1)
    fig.update_yaxes(title_text="CO2 Captured (kg/h)", row=1, col=2)
    fig.update_yaxes(title_text="Thermal Energy (kWh_th/tCO2)", row=2, col=1)
    fig.update_yaxes(title_text="Electrical Energy (kWh_e/tCO2)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Solid sorbent DAC | Degradation + humidity + part-load</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
