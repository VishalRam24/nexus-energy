"""EC091 — Vapor Compression Chiller — F1b Part-Load — Simulation & HTML Report"""
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
            "COP vs PLR at various T_cw",
            "COP vs T_cw at various PLR",
            "Electrical Input vs PLR",
            "COP Heatmap: PLR vs T_cw",
        ],
        vertical_spacing=0.12,
    )

    plr = np.linspace(0.1, 1.0, 100)

    # Row 1 Col 1 — COP vs PLR
    for T_cw in [18, 24, 29, 35, 40]:
        r = model.predict({"T_chw": 6.7, "T_cw": T_cw, "PLR": plr})
        fig.add_trace(
            go.Scatter(x=plr, y=r["cop"], name=f"T_cw={T_cw}C"),
            row=1, col=1,
        )

    # Row 1 Col 2 — COP vs T_cw
    T_cw_range = np.linspace(15, 45, 100)
    for p in [1.0, 0.75, 0.5, 0.25]:
        r = model.predict({"T_chw": 6.7, "T_cw": T_cw_range, "PLR": p})
        fig.add_trace(
            go.Scatter(x=T_cw_range, y=r["cop"], name=f"PLR={p:.2f}", showlegend=False),
            row=1, col=2,
        )

    # Row 2 Col 1 — W vs PLR
    for T_cw in [20, 29, 35]:
        r = model.predict({"T_chw": 6.7, "T_cw": T_cw, "PLR": plr})
        fig.add_trace(
            go.Scatter(x=plr, y=r["electrical_input_kw"],
                       name=f"W T_cw={T_cw}C", showlegend=False),
            row=2, col=1,
        )

    # Row 2 Col 2 — Heatmap
    plr_g = np.linspace(0.1, 1.0, 50)
    Tcw_g = np.linspace(15, 45, 50)
    cop_map = np.zeros((50, 50))
    for i, tcw in enumerate(Tcw_g):
        r = model.predict({"T_chw": 6.7, "T_cw": tcw, "PLR": plr_g})
        cop_map[i, :] = r["cop"]
    fig.add_trace(
        go.Heatmap(x=plr_g, y=Tcw_g, z=cop_map, colorscale="Viridis"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="PLR", row=1, col=1)
    fig.update_xaxes(title_text="T_cw (C)", row=1, col=2)
    fig.update_xaxes(title_text="PLR", row=2, col=1)
    fig.update_xaxes(title_text="PLR", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="kW_e", row=2, col=1)
    fig.update_yaxes(title_text="T_cw (C)", row=2, col=2)

    iplv = model._model.iplv()
    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | IPLV={iplv:.2f}",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")
    print(f"IPLV = {iplv:.2f}")


if __name__ == "__main__":
    generate_report()
