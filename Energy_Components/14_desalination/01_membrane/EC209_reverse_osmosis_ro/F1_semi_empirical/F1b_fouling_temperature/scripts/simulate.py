"""EC209 — Reverse Osmosis — F1b Fouling Temperature — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    hours = np.linspace(0, 87600, 50)  # 10 years
    temps = np.linspace(10, 40, 50)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Flux Decline Factor vs Operating Hours",
            "Permeate Flow vs Temperature",
            "SEC vs Operating Hours",
            "Salt Rejection vs Operating Hours",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Flux decline vs time at different temperatures
    for T in [15, 25, 35]:
        ff = []
        for h in hours:
            r = model.predict({"operating_hours": float(h), "feed_temperature_degC": T})
            ff.append(float(np.atleast_1d(r["flux_decline_factor"])[0]))
        fig.add_trace(
            go.Scatter(x=hours / 8760, y=ff, name=f"T={T}C", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: Permeate flow vs temperature at various ages
    for h in [0, 26280, 52560]:
        Q = []
        for T in temps:
            r = model.predict({"feed_temperature_degC": float(T), "operating_hours": h})
            Q.append(float(np.atleast_1d(r["permeate_flow_m3_h"])[0]))
        fig.add_trace(
            go.Scatter(x=temps, y=Q, name=f"{h/8760:.0f}yr", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: SEC vs time
    for S in [20000, 35000, 42000]:
        sec = []
        for h in hours:
            r = model.predict({"operating_hours": float(h), "feed_salinity_ppm": S})
            sec.append(float(np.atleast_1d(r["sec_kwh_m3"])[0]))
        fig.add_trace(
            go.Scatter(x=hours / 8760, y=sec, name=f"S={S}ppm", line=dict(width=2)),
            row=2, col=1
        )

    # Plot 4: Rejection vs time
    rej = []
    for h in hours:
        r = model.predict({"operating_hours": float(h)})
        rej.append(float(np.atleast_1d(r["rejection_pct"])[0]))
    fig.add_trace(
        go.Scatter(x=hours / 8760, y=rej, name="Rejection", line=dict(width=2)),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Years", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Years", row=2, col=1)
    fig.update_xaxes(title_text="Years", row=2, col=2)
    fig.update_yaxes(title_text="Flux Factor (-)", row=1, col=1)
    fig.update_yaxes(title_text="Permeate Flow (m3/h)", row=1, col=2)
    fig.update_yaxes(title_text="SEC (kWh/m3)", row=2, col=1)
    fig.update_yaxes(title_text="Rejection (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>SWRO membrane | Fouling + temperature effects</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
