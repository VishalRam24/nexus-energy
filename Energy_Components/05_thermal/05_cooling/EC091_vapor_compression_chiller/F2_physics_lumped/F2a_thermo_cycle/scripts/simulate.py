"""EC091 — Vapor Compression Chiller — F2a — Simulation & HTML Report"""
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
        subplot_titles=["COP_cooling vs T_evap", "Compressor Power vs T_evap",
                        "COP_cooling vs T_cond", "COP Cooling Map"],
        vertical_spacing=0.14)

    T_evaps = np.linspace(-3, 12, 25)
    for Tc in [30, 35, 40, 45]:
        cops, powers = [], []
        for Te in T_evaps:
            r = model.predict({"T_evap_degC": float(Te), "T_cond_degC": Tc})
            cops.append(r["cop_cooling"])
            powers.append(r["compressor_kw"])
        fig.add_trace(go.Scatter(x=T_evaps, y=cops, name=f"T_c={Tc}C"), row=1, col=1)
        fig.add_trace(go.Scatter(x=T_evaps, y=powers, name=f"W T_c={Tc}C", showlegend=False), row=1, col=2)

    T_conds = np.linspace(28, 50, 25)
    for Te in [2, 5, 7, 10]:
        cops = []
        for Tc in T_conds:
            r = model.predict({"T_evap_degC": Te, "T_cond_degC": float(Tc)})
            cops.append(r["cop_cooling"])
        fig.add_trace(go.Scatter(x=T_conds, y=cops, name=f"T_e={Te}C"), row=2, col=1)

    Te_g = np.linspace(-3, 12, 20)
    Tc_g = np.linspace(28, 50, 20)
    z = np.zeros((len(Te_g), len(Tc_g)))
    for i, Te in enumerate(Te_g):
        for j, Tc in enumerate(Tc_g):
            r = model.predict({"T_evap_degC": float(Te), "T_cond_degC": float(Tc)})
            z[i,j] = r["cop_cooling"]
    fig.add_trace(go.Heatmap(x=Tc_g, y=Te_g, z=z, colorscale="Viridis"), row=2, col=2)

    fig.update_xaxes(title_text="T_evap (C)", row=1, col=1)
    fig.update_xaxes(title_text="T_evap (C)", row=1, col=2)
    fig.update_xaxes(title_text="T_cond (C)", row=2, col=1)
    fig.update_xaxes(title_text="T_cond (C)", row=2, col=2)
    fig.update_yaxes(title_text="COP_cool", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=2)
    fig.update_yaxes(title_text="COP_cool", row=2, col=1)
    fig.update_yaxes(title_text="T_evap (C)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}", height=850, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
