"""EC091 — Vapor Compression Chiller — F1a COP Map — Simulation & HTML Report"""
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
            "COP vs Condenser Temperature",
            "COP vs Evaporator Temperature",
            "Part-Load Performance (PLR Effect)",
            "COP Map — T_evap vs T_cond (Heatmap)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: COP vs T_cond for several T_evap values
    T_cond_arr = np.linspace(25, 45, 100)
    for te in [4, 6, 8, 10, 12]:
        r = model.predict({"T_chw_supply": te, "T_cond": T_cond_arr})
        fig.add_trace(
            go.Scatter(x=T_cond_arr, y=r["cop"], name=f"T_evap={te}C"),
            row=1, col=1,
        )

    # Plot 2: COP vs T_evap for several T_cond values
    T_evap_arr = np.linspace(4, 12, 100)
    for tc in [25, 30, 35, 40, 45]:
        r = model.predict({"T_chw_supply": T_evap_arr, "T_cond": tc})
        fig.add_trace(
            go.Scatter(x=T_evap_arr, y=r["cop"], name=f"T_cond={tc}C"),
            row=1, col=2,
        )

    # Plot 3: Part-load curves — COP and W_comp vs PLR
    plr_arr = np.linspace(0.1, 1.0, 100)
    for tc in [25, 35, 45]:
        r = model.predict({"T_chw_supply": 5.0, "T_cond": tc, "part_load_ratio": plr_arr})
        fig.add_trace(
            go.Scatter(x=plr_arr, y=r["cop"], name=f"COP T_cond={tc}C", showlegend=True),
            row=2, col=1,
        )

    # Plot 4: COP heatmap T_evap vs T_cond
    T_evap_grid = np.linspace(4, 12, 50)
    T_cond_grid = np.linspace(25, 45, 50)
    cop_map = np.zeros((50, 50))
    for i, te in enumerate(T_evap_grid):
        r = model.predict({"T_chw_supply": te, "T_cond": T_cond_grid, "part_load_ratio": 1.0})
        cop_map[i, :] = r["cop"]
    fig.add_trace(
        go.Heatmap(
            x=T_cond_grid, y=T_evap_grid, z=cop_map,
            colorscale="RdYlGn", colorbar=dict(title="COP"),
            name="COP Map",
        ),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="T_cond (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_evap (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="T_cond (degC)", row=2, col=2)
    fig.update_yaxes(title_text="COP (-)", row=1, col=1)
    fig.update_yaxes(title_text="COP (-)", row=1, col=2)
    fig.update_yaxes(title_text="COP (-)", row=2, col=1)
    fig.update_yaxes(title_text="T_evap (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} COP Map",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
