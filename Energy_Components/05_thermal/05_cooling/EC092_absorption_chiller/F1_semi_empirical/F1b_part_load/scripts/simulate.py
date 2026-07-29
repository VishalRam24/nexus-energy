"""EC092 — Absorption Chiller — F1b Part-Load — Simulation & HTML Report"""
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
            "COP vs PLR at various T_hot",
            "COP vs T_hot at various PLR",
            "Heat Input vs PLR",
            "COP Heatmap: PLR vs T_hot",
        ],
        vertical_spacing=0.12,
    )

    plr = np.linspace(0.15, 1.0, 100)

    # Row 1 Col 1 — COP vs PLR
    for T_hot in [75, 85, 90, 100, 110]:
        r = model.predict({"T_hot": T_hot, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr})
        fig.add_trace(
            go.Scatter(x=plr, y=r["cop"], name=f"T_hot={T_hot}C"),
            row=1, col=1,
        )

    # Row 1 Col 2 — COP vs T_hot
    T_hot_range = np.linspace(70, 120, 100)
    for p in [1.0, 0.75, 0.5, 0.3]:
        r = model.predict({"T_hot": T_hot_range, "T_cw": 30.0, "T_chw": 7.0, "PLR": p})
        fig.add_trace(
            go.Scatter(x=T_hot_range, y=r["cop"], name=f"PLR={p:.2f}", showlegend=False),
            row=1, col=2,
        )

    # Row 2 Col 1 — Heat input vs PLR
    r = model.predict({"T_hot": 90.0, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr})
    fig.add_trace(
        go.Scatter(x=plr, y=r["heat_input_kw"], name="Q_gen",
                   line=dict(color="crimson"), showlegend=False),
        row=2, col=1,
    )

    # Row 2 Col 2 — Heatmap
    plr_g = np.linspace(0.15, 1.0, 50)
    Thot_g = np.linspace(70, 120, 50)
    cop_map = np.zeros((50, 50))
    for i, th in enumerate(Thot_g):
        r = model.predict({"T_hot": th, "T_cw": 30.0, "T_chw": 7.0, "PLR": plr_g})
        cop_map[i, :] = r["cop"]
    fig.add_trace(
        go.Heatmap(x=plr_g, y=Thot_g, z=cop_map, colorscale="Viridis"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="PLR", row=1, col=1)
    fig.update_xaxes(title_text="T_hot (C)", row=1, col=2)
    fig.update_xaxes(title_text="PLR", row=2, col=1)
    fig.update_xaxes(title_text="PLR", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="kW_th", row=2, col=1)
    fig.update_yaxes(title_text="T_hot (C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
