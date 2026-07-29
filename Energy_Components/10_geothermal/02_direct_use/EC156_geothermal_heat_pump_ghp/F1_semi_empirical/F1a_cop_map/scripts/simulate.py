"""EC156 — Geothermal Heat Pump (GHP) — F1a COP Map — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import GHPF1a
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_srcs = np.linspace(0, 25, 100)
    T_snks = np.linspace(25, 65, 100)

    # Load GHP model for COP advantage calculation
    params_path = Path(__file__).parent.parent / "data" / "parameters.json"
    with open(params_path) as f:
        params = json.load(f)
    ghp_m = GHPF1a(params)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Heating COP vs T_source (ground loop)",
            "COP (Heat/Cool) vs T_sink (load T)",
            "GHP vs ASHP COP Advantage (winter)",
            "COP Heating Map: T_source × T_sink",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: COP_heating vs T_source for several T_sink
    for T_snk in [30, 35, 45, 55]:
        r = model.predict({"T_source": T_srcs, "T_sink": float(T_snk)})
        fig.add_trace(go.Scatter(x=T_srcs, y=r["cop_heating"], name=f"T_sink={T_snk}°C",
                                 line=dict(width=2)), row=1, col=1)

    # Plot 2: COP heat & cool vs T_sink at T_src=10°C (typical ground)
    r = model.predict({"T_source": 10.0, "T_sink": T_snks})
    fig.add_trace(go.Scatter(x=T_snks, y=r["cop_heating"], name="COP_heating (T_src=10°C)",
                             line=dict(width=2, color="red")), row=1, col=2)
    fig.add_trace(go.Scatter(x=T_snks, y=r["cop_cooling"], name="COP_cooling (T_src=10°C)",
                             line=dict(width=2, color="blue")), row=1, col=2)

    # Plot 3: GHP vs ASHP advantage
    T_air = np.linspace(-15, 20, 100)  # ambient air T (winter range)
    T_gnd = 10.0  # stable ground T
    T_load = 35.0  # radiator supply T
    delta_cop = ghp_m.cop_advantage_over_ashp(
        T_source_ghp_c=T_gnd, T_source_ashp_c=T_air, T_sink_c=T_load
    )
    cop_ghp  = model.predict({"T_source": T_gnd, "T_sink": T_load})["cop_heating"]
    ashp_cop = 0.45 * (T_load + 273.15) / np.clip((T_load + 273.15) - (T_air + 273.15), 1e-3, None)
    ashp_cop = np.clip(ashp_cop, 1.0, 20.0)
    fig.add_trace(go.Scatter(x=T_air, y=float(cop_ghp) * np.ones_like(T_air),
                             name=f"GHP COP (T_gnd=10°C)", line=dict(width=2, color="green")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=T_air, y=ashp_cop, name="ASHP COP (variable air)",
                             line=dict(width=2, color="orange")), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_air, y=delta_cop, name="Delta COP",
                             line=dict(width=2, color="purple", dash="dash")), row=2, col=1)

    # Plot 4: COP heating heatmap
    T_src_g = np.linspace(0, 25, 40)
    T_snk_g = np.linspace(25, 65, 40)
    cop_map = np.zeros((len(T_src_g), len(T_snk_g)))
    for i, ts in enumerate(T_src_g):
        r = model.predict({"T_source": float(ts), "T_sink": T_snk_g})
        cop_map[i, :] = r["cop_heating"]

    fig.add_trace(go.Heatmap(x=T_snk_g, y=T_src_g, z=cop_map,
                             colorscale="RdYlGn", colorbar=dict(title="COP_h"),
                             name="COP Heating Map"), row=2, col=2)

    fig.update_xaxes(title_text="T_source / Ground T (°C)", row=1, col=1)
    fig.update_xaxes(title_text="T_sink / Load T (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Ambient Air T (°C)", row=2, col=1)
    fig.update_xaxes(title_text="T_sink (°C)", row=2, col=2)
    fig.update_yaxes(title_text="COP_heating", row=1, col=1)
    fig.update_yaxes(title_text="COP", row=1, col=2)
    fig.update_yaxes(title_text="COP", row=2, col=1)
    fig.update_yaxes(title_text="T_source (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} COP Map<br>"
              f"<sup>Source: Staffell et al. (2012); ASHRAE (2011)</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    r_dp = model.predict({"T_source": 10.0, "T_sink": 35.0})
    print("\n--- Design Point Summary (T_src=10°C, T_sink=35°C) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.3f}")


if __name__ == "__main__":
    generate_report()
