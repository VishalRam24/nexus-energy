"""EC092 — Absorption Chiller — F2a — Simulation & HTML Report"""
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
        subplot_titles=["COP vs Generator Temp", "Heat Input vs Generator Temp",
                        "COP vs Condenser Temp", "COP vs Evaporator Temp"],
        vertical_spacing=0.14)

    T_gens = np.linspace(78, 115, 30)
    for Tc in [30, 35, 40]:
        cops, qgens = [], []
        for Tg in T_gens:
            r = model.predict({"T_gen_degC": float(Tg), "T_cond_degC": Tc, "T_evap_degC": 7.0, "T_abs_degC": Tc})
            cops.append(r["cop"])
            qgens.append(r["heat_input_kw"])
        fig.add_trace(go.Scatter(x=T_gens, y=cops, name=f"T_c={Tc}C"), row=1, col=1)
        fig.add_trace(go.Scatter(x=T_gens, y=qgens, name=f"Qg T_c={Tc}C", showlegend=False), row=1, col=2)

    T_conds = np.linspace(26, 42, 20)
    cops = []
    for Tc in T_conds:
        r = model.predict({"T_gen_degC": 90.0, "T_cond_degC": float(Tc), "T_evap_degC": 7.0, "T_abs_degC": float(Tc)})
        cops.append(r["cop"])
    fig.add_trace(go.Scatter(x=T_conds, y=cops, name="COP vs T_cond"), row=2, col=1)

    T_evaps = np.linspace(4, 14, 20)
    cops = []
    for Te in T_evaps:
        r = model.predict({"T_gen_degC": 90.0, "T_cond_degC": 35.0, "T_evap_degC": float(Te), "T_abs_degC": 35.0})
        cops.append(r["cop"])
    fig.add_trace(go.Scatter(x=T_evaps, y=cops, name="COP vs T_evap"), row=2, col=2)

    fig.update_xaxes(title_text="T_gen (C)", row=1, col=1)
    fig.update_xaxes(title_text="T_gen (C)", row=1, col=2)
    fig.update_xaxes(title_text="T_cond (C)", row=2, col=1)
    fig.update_xaxes(title_text="T_evap (C)", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=2)
    fig.update_yaxes(title_text="COP", row=2, col=1)
    fig.update_yaxes(title_text="COP", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}", height=850, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
