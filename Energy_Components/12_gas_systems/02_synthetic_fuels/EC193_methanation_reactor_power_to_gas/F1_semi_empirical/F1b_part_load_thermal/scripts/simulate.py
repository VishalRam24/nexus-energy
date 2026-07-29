"""EC193 — Methanation Reactor — F1b Part-Load Thermal — Simulation & HTML Report"""
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
            "Conversion vs PLR",
            "CH4 Production vs PLR",
            "Heat Recovery vs PLR",
            "Overall Efficiency vs PLR",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Conversion vs PLR at various H2/CO2 ratios
    for ratio in [3.5, 4.0, 4.5, 5.0]:
        conv = []
        for plr in PLRs:
            r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": ratio, "PLR": float(plr)})
            conv.append(float(np.atleast_1d(r["conversion"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=conv, name=f"H2/CO2={ratio}", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: CH4 production vs PLR
    for T in [250, 300, 350]:
        ch4 = []
        for plr in PLRs:
            r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0,
                               "PLR": float(plr), "T_reactor_degC": T})
            ch4.append(float(np.atleast_1d(r["ch4_production_mol_s"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=ch4, name=f"T={T}C", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Heat recovery vs PLR
    for n_co2 in [0.5, 1.0, 2.0]:
        Q = []
        for plr in PLRs:
            r = model.predict({"co2_flow_mol_s": n_co2, "h2_co2_ratio": 4.0, "PLR": float(plr)})
            Q.append(float(np.atleast_1d(r["heat_recovery_kw"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=Q, name=f"nCO2={n_co2}", line=dict(width=2)),
            row=2, col=1
        )

    # Plot 4: Efficiency vs PLR
    for T in [250, 300, 350]:
        eta = []
        for plr in PLRs:
            r = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0,
                               "PLR": float(plr), "T_reactor_degC": T})
            eta.append(float(np.atleast_1d(r["overall_efficiency"])[0]))
        fig.add_trace(
            go.Scatter(x=PLRs, y=eta, name=f"eta T={T}C", line=dict(width=2)),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="Conversion (-)", row=1, col=1)
    fig.update_yaxes(title_text="CH4 Production (mol/s)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Recovery (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Overall Efficiency (-)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>CO2 + 4H2 -> CH4 + 2H2O | Part-load + heat recovery</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    r_dp = model.predict({"co2_flow_mol_s": 1.0, "h2_co2_ratio": 4.0, "PLR": 1.0})
    print("\n--- Design Point Summary (PLR=1.0, T=300C, P=10bar) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(np.atleast_1d(v)[0]):.4f}")


if __name__ == "__main__":
    generate_report()
