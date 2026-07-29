"""EC198 — Post-Combustion Capture — F1b Part-Load Degradation — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    PLRs = np.linspace(0.3, 1.0, 50)
    hours_range = np.linspace(0, 50000, 50)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Reboiler Duty vs PLR",
            "Solvent Degradation vs Operating Hours",
            "Electrical Consumption vs PLR",
            "Total Energy Penalty vs Operating Hours",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Reboiler duty vs PLR at various degradation levels
    for hours in [0, 10000, 30000, 50000]:
        q = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "operating_hours": hours})
            q.append(float(np.atleast_1d(r["reboiler_duty_gj_ton"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=q, name=f"{hours/1000:.0f}kh", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: Degradation vs hours
    for plr_val in [0.3, 0.5, 0.7, 1.0]:
        deg = []
        for h in hours_range:
            r = model.predict({"PLR": plr_val, "operating_hours": float(h)})
            deg.append(float(np.atleast_1d(r["solvent_degradation_pct"])[0]))
        fig.add_trace(
            go.Scatter(x=hours_range, y=deg, name=f"PLR={plr_val}", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Electrical vs PLR
    for cr in [0.7, 0.85, 0.95]:
        e = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "capture_rate": cr, "operating_hours": 0})
            e.append(float(np.atleast_1d(r["electrical_kwh_ton"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=e, name=f"CR={cr}", line=dict(width=2)),
            row=2, col=1
        )

    # Plot 4: Energy penalty vs hours at different PLRs
    for plr_val in [0.3, 0.5, 0.7, 1.0]:
        pen = []
        for h in hours_range:
            r = model.predict({"PLR": plr_val, "operating_hours": float(h)})
            pen.append(float(np.atleast_1d(r["total_energy_penalty_pct"])[0]))
        fig.add_trace(
            go.Scatter(x=hours_range, y=pen, name=f"pen PLR={plr_val}", line=dict(width=2)),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Operating Hours", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Operating Hours", row=2, col=2)
    fig.update_yaxes(title_text="Reboiler Duty (GJ/tCO2)", row=1, col=1)
    fig.update_yaxes(title_text="Degradation (%)", row=1, col=2)
    fig.update_yaxes(title_text="Electrical (kWh/tCO2)", row=2, col=1)
    fig.update_yaxes(title_text="Energy Penalty (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>MEA amine scrubbing | Part-load + solvent degradation</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
