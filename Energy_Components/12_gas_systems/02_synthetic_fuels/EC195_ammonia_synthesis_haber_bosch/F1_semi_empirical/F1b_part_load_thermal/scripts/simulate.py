"""EC195 — Ammonia Synthesis (Haber-Bosch) — F1b Part-Load Thermal — Simulation & HTML Report"""
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

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Single-Pass Conversion vs PLR",
            "Recycle Ratio vs PLR",
            "Energy Consumption vs PLR",
            "NH3 Production vs PLR",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Conversion vs PLR at various pressures
    for P in [150, 200, 250]:
        conv = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "pressure_bar": P})
            conv.append(float(np.atleast_1d(r["single_pass_conversion"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=conv, name=f"P={P}bar", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: Recycle ratio vs PLR
    for P in [150, 200, 250]:
        rr = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "pressure_bar": P})
            rr.append(float(np.atleast_1d(r["recycle_ratio"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=rr, name=f"RR P={P}bar", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Energy vs PLR
    for T in [400, 450, 500]:
        E = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "temperature_c": T})
            E.append(float(np.atleast_1d(r["energy_kwh_per_ton"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=E, name=f"E T={T}C", line=dict(width=2)),
            row=2, col=1
        )

    # Plot 4: NH3 production vs PLR
    for n2 in [0.5, 1.0, 2.0]:
        nh3 = []
        for plr in PLRs:
            r = model.predict({"PLR": float(plr), "n2_flow_mol_s": n2})
            nh3.append(float(np.atleast_1d(r["nh3_production_mol_s"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=nh3, name=f"nN2={n2}", line=dict(width=2)),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="Single-Pass Conversion (-)", row=1, col=1)
    fig.update_yaxes(title_text="Recycle Ratio (-)", row=1, col=2)
    fig.update_yaxes(title_text="Energy (kWh/ton NH3)", row=2, col=1)
    fig.update_yaxes(title_text="NH3 Production (mol/s)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>N2 + 3H2 -> 2NH3 | Part-load + recycle loop</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    r_dp = model.predict({"PLR": 1.0})
    print("\n--- Design Point Summary (PLR=1.0, T=450C, P=200bar) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")


if __name__ == "__main__":
    generate_report()
