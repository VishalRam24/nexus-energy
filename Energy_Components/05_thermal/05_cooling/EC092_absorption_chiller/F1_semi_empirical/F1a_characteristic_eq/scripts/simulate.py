"""EC092 — Absorption Chiller — F1a — Simulation & HTML Report"""
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
            "COP vs Generator Temperature",
            "COP vs Condenser Temperature",
            "Heat Flows at Rated Conditions vs T_generator",
            "COP Map (T_gen vs T_cond Heatmap)",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: COP vs T_gen at fixed T_cond
    T_gens = np.linspace(70, 120, 100)
    for T_cond in [28.0, 32.0, 35.0, 38.0, 42.0]:
        r = model.predict({"T_generator": T_gens, "T_condenser": T_cond, "T_evaporator": 7.0})
        fig.add_trace(go.Scatter(x=T_gens, y=r["cop"], name=f"T_cond={T_cond}C"), row=1, col=1)

    # Panel 2: COP vs T_cond at fixed T_gen
    T_conds = np.linspace(25, 45, 100)
    for T_gen in [80.0, 90.0, 100.0, 110.0]:
        r = model.predict({"T_generator": T_gen, "T_condenser": T_conds, "T_evaporator": 7.0})
        fig.add_trace(go.Scatter(x=T_conds, y=r["cop"], name=f"T_gen={T_gen}C"), row=1, col=2)

    # Panel 3: Heat flows vs T_gen
    T_gens = np.linspace(70, 120, 100)
    r = model.predict({"T_generator": T_gens, "T_condenser": 35.0, "T_evaporator": 7.0})
    fig.add_trace(go.Scatter(x=T_gens, y=r["heat_input_kw"], name="Q_generator (kW)", line=dict(dash="solid")), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_gens, y=r["cooling_kw"], name="Q_cool (kW)", line=dict(dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_gens, y=r["heat_rejection_kw"], name="Q_reject (kW)", line=dict(dash="dash")), row=2, col=1)

    # Panel 4: COP heatmap
    T_gen_g = np.linspace(70, 120, 50)
    T_cond_g = np.linspace(25, 45, 50)
    cop_map = np.zeros((50, 50))
    for i, tg in enumerate(T_gen_g):
        r = model.predict({"T_generator": tg, "T_condenser": T_cond_g, "T_evaporator": 7.0})
        cop_map[i, :] = r["cop"]
    fig.add_trace(go.Heatmap(
        x=T_cond_g, y=T_gen_g, z=cop_map,
        colorscale="RdYlGn", colorbar=dict(title="COP", x=1.02),
        name="COP"), row=2, col=2)

    fig.update_xaxes(title_text="T_generator (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_condenser (degC)", row=1, col=2)
    fig.update_xaxes(title_text="T_generator (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_condenser (degC)", row=2, col=2)
    fig.update_yaxes(title_text="COP (-)", row=1, col=1)
    fig.update_yaxes(title_text="COP (-)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Flow (kW)", row=2, col=1)
    fig.update_yaxes(title_text="T_generator (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Characteristic Equation",
        height=850,
        template="plotly_white",
        legend=dict(orientation="v", x=1.08, y=0.95),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
