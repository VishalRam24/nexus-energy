"""EC068 — ASHP — F1b Part-Load — Simulation & HTML Report"""
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
            "COP vs PLR at various T_source (T_sink=35C)",
            "COP Degradation Factor vs PLR",
            "Electrical Input vs PLR",
            "COP Map: PLR vs T_source (T_sink=35C)",
        ],
        vertical_spacing=0.12,
    )

    plr = np.linspace(0.05, 1.0, 100)

    # Row 1 Col 1 — COP vs PLR at various T_source
    for Ts in [-10, 0, 7, 15, 25]:
        r = model.predict({"T_source": Ts, "T_sink": 35.0, "part_load_ratio": plr})
        fig.add_trace(
            go.Scatter(x=plr, y=r["cop"], name=f"T_src={Ts}C"),
            row=1, col=1,
        )

    # Row 1 Col 2 — Degradation factor vs PLR
    r = model.predict({"T_source": 7.0, "T_sink": 35.0, "part_load_ratio": plr})
    fig.add_trace(
        go.Scatter(x=plr, y=r["cop_degradation_factor"],
                   name="PLF", line=dict(color="crimson"), showlegend=False),
        row=1, col=2,
    )

    # Row 2 Col 1 — Electrical input vs PLR
    for Ts in [-10, 0, 7, 15]:
        r = model.predict({"T_source": Ts, "T_sink": 35.0, "part_load_ratio": plr})
        fig.add_trace(
            go.Scatter(x=plr, y=r["electrical_input_kw"],
                       name=f"W T_src={Ts}C", showlegend=False),
            row=2, col=1,
        )

    # Row 2 Col 2 — Heatmap COP(PLR, T_source)
    plr_grid = np.linspace(0.1, 1.0, 50)
    Ts_grid = np.linspace(-15, 30, 50)
    cop_map = np.zeros((50, 50))
    for i, ts in enumerate(Ts_grid):
        r = model.predict({"T_source": ts, "T_sink": 35.0, "part_load_ratio": plr_grid})
        cop_map[i, :] = r["cop"]
    fig.add_trace(
        go.Heatmap(x=plr_grid, y=Ts_grid, z=cop_map, colorscale="Viridis", name="COP"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="PLR", row=1, col=1)
    fig.update_xaxes(title_text="PLR", row=1, col=2)
    fig.update_xaxes(title_text="PLR", row=2, col=1)
    fig.update_xaxes(title_text="PLR", row=2, col=2)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="Degradation Factor", row=1, col=2)
    fig.update_yaxes(title_text="kW_e", row=2, col=1)
    fig.update_yaxes(title_text="T_source (C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | Part-Load with EN 14825 PLF",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
