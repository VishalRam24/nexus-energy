"""EC069 — GSHP — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import GSHPF1a
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    gshp = GSHPF1a(params)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "COP vs T_source (Ground Temperature)",
            "COP vs T_sink (Heating Distribution)",
            "Electrical Input vs T_source",
            "COP Map (T_source vs T_sink)",
        ],
        vertical_spacing=0.15,
    )

    Ts = np.linspace(0, 20, 100)

    # 1) COP vs T_source at multiple sink temperatures
    for Tk in [30, 35, 45, 55, 65]:
        r = model.predict({"T_source": Ts, "T_sink": Tk})
        fig.add_trace(
            go.Scatter(x=Ts, y=r["cop"], name=f"T_sink={Tk}°C"),
            row=1, col=1,
        )
    # Mark rating point G10/W35
    r_rated = model.predict({"T_source": 10.0, "T_sink": 35.0})
    fig.add_trace(
        go.Scatter(x=[10.0], y=[float(r_rated["cop"])],
                   mode="markers", marker=dict(size=12, color="red", symbol="star"),
                   name="G10/W35 rated", showlegend=True),
        row=1, col=1,
    )

    # 2) COP vs T_sink at typical ground temps
    Tk = np.linspace(25, 65, 100)
    for Tg in [0, 5, 10, 15, 20]:
        r = model.predict({"T_source": Tg, "T_sink": Tk})
        fig.add_trace(
            go.Scatter(x=Tk, y=r["cop"], name=f"T_ground={Tg}°C"),
            row=1, col=2,
        )

    # 3) Electrical input vs T_source
    for Tk_val in [35, 45, 55]:
        r = model.predict({"T_source": Ts, "T_sink": Tk_val})
        fig.add_trace(
            go.Scatter(x=Ts, y=r["electrical_input_kw"],
                       name=f"W @ T_sink={Tk_val}°C", showlegend=False),
            row=2, col=1,
        )

    # 4) COP heatmap
    Ts_grid = np.linspace(0, 20, 40)
    Tk_grid = np.linspace(25, 65, 40)
    cop_map = np.zeros((40, 40))
    for i, ts in enumerate(Ts_grid):
        r = model.predict({"T_source": ts, "T_sink": Tk_grid})
        cop_map[i, :] = r["cop"]
    fig.add_trace(
        go.Heatmap(x=Tk_grid, y=Ts_grid, z=cop_map, colorscale="Viridis",
                   colorbar=dict(title="COP"), name="COP"),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="T_source / Ground (°C)", row=1, col=1)
    fig.update_xaxes(title_text="T_sink / Heating (°C)", row=1, col=2)
    fig.update_xaxes(title_text="T_source / Ground (°C)", row=2, col=1)
    fig.update_xaxes(title_text="T_sink / Heating (°C)", row=2, col=2)
    fig.update_yaxes(title_text="COP",    row=1, col=1)
    fig.update_yaxes(title_text="COP",    row=1, col=2)
    fig.update_yaxes(title_text="kW_e",   row=2, col=1)
    fig.update_yaxes(title_text="T_source (°C)", row=2, col=2)

    fig.update_layout(
        title=(
            f"{info['ec_id']} — {info['name']} — {info['fidelity']} COP Map<br>"
            "<sup>Carnot fraction = 0.50 | Rated G10/W35 | 15 kW_th | Staffell et al. (2012)</sup>"
        ),
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")

    # Summary at rated conditions
    print("\n=== Rated Condition Summary (G10/W35) ===")
    r_stc = model.predict({"T_source": 10.0, "T_sink": 35.0})
    for k, v in r_stc.items():
        print(f"  {k:22s}: {float(v):.3f}")

    # ASHP comparison
    print("\n=== GSHP vs ASHP COP comparison at W35 ===")
    compare_sources = [0.0, 5.0, 10.0, 15.0]
    for ts in compare_sources:
        r_g = model.predict({"T_source": ts, "T_sink": 35.0})
        cop_gshp = float(r_g["cop"])
        # ASHP COP at same temp
        T_s_K = ts + 273.15; T_k_K = 35.0 + 273.15
        cop_ashp = min(0.45 * T_k_K / (T_k_K - T_s_K), 15.0)
        advantage = gshp.cop_advantage_over_ashp(ts, 35.0)
        print(f"  T_source={ts:5.1f}°C: GSHP={cop_gshp:.2f}, ASHP~{cop_ashp:.2f}, Ratio={float(advantage):.2f}x")


if __name__ == "__main__":
    generate_report()
