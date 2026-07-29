"""EC218 — Thermionic Converter — F1b Space-Charge — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_emitters = np.linspace(1300, 2100, 60)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Current Density vs Emitter Temperature",
            "Power Density vs Emitter Temperature",
            "Efficiency vs Emitter Temperature",
            "Work Function vs Emitter Temperature",
        ],
        vertical_spacing=0.15,
    )

    for T_c in [800.0, 1000.0, 1200.0]:
        J_nets, P_dens, etas, phi_es = [], [], [], []
        for T_e in T_emitters:
            r = model.predict({"T_emitter_K": float(T_e), "T_collector_K": T_c})
            J_nets.append(float(np.atleast_1d(r["J_net_Am2"])[0]))
            P_dens.append(float(np.atleast_1d(r["power_density_w_cm2"])[0]))
            etas.append(float(np.atleast_1d(r["efficiency"])[0]) * 100)
            phi_es.append(float(np.atleast_1d(r["phi_e_eV"])[0]))

        fig.add_trace(go.Scatter(x=T_emitters, y=J_nets, name=f"J_net Tc={T_c:.0f}K", line=dict(width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=T_emitters, y=P_dens, name=f"P_dens Tc={T_c:.0f}K", line=dict(width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=T_emitters, y=etas, name=f"eta Tc={T_c:.0f}K", line=dict(width=2)), row=2, col=1)

    fig.add_trace(go.Scatter(x=T_emitters, y=phi_es, name="phi_emitter(T)", line=dict(width=2, color="red")), row=2, col=2)
    phi_c0 = model._model.phi_c0
    fig.add_trace(go.Scatter(x=T_emitters, y=[phi_c0]*len(T_emitters), name="phi_collector (T=900K)", line=dict(width=2, dash="dash", color="blue")), row=2, col=2)

    fig.update_xaxes(title_text="T_emitter (K)", row=1, col=1)
    fig.update_xaxes(title_text="T_emitter (K)", row=1, col=2)
    fig.update_xaxes(title_text="T_emitter (K)", row=2, col=1)
    fig.update_xaxes(title_text="T_emitter (K)", row=2, col=2)
    fig.update_yaxes(title_text="J_net (A/m²)", row=1, col=1)
    fig.update_yaxes(title_text="Power Density (W/cm²)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Work Function (eV)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Space-charge correction + T-dependent work function</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
