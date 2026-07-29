"""EC098 — ORC — F1a — Simulation & HTML Report"""
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
            "Efficiency vs T_hot (several T_cold)",
            "Efficiency vs T_cold (several T_hot)",
            "Part-Load Efficiency Curve",
            "Efficiency Map (T_hot vs T_cold Heatmap)",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: eta vs T_hot
    T_hots = np.linspace(80, 300, 150)
    for T_cold in [15.0, 25.0, 35.0, 45.0]:
        r = model.predict({"T_hot": T_hots, "T_cold": T_cold, "part_load_ratio": 1.0})
        fig.add_trace(go.Scatter(x=T_hots, y=np.asarray(r["efficiency"]) * 100,
                                 name=f"T_cold={T_cold}C"), row=1, col=1)
    # Carnot reference
    T_hots_K = T_hots + 273.15
    eta_carnot = (1.0 - (30 + 273.15) / T_hots_K) * 100
    fig.add_trace(go.Scatter(x=T_hots, y=eta_carnot, name="Carnot (T_cold=30C)",
                             line=dict(dash="dash", color="black")), row=1, col=1)

    # Panel 2: eta vs T_cold
    T_colds = np.linspace(10, 50, 100)
    for T_hot in [100.0, 150.0, 200.0, 250.0]:
        r = model.predict({"T_hot": T_hot, "T_cold": T_colds})
        fig.add_trace(go.Scatter(x=T_colds, y=np.asarray(r["efficiency"]) * 100,
                                 name=f"T_hot={T_hot}C"), row=1, col=2)

    # Panel 3: Part-load curve
    plrs = np.linspace(0.3, 1.0, 100)
    for T_hot in [100.0, 150.0, 200.0]:
        r = model.predict({"T_hot": T_hot, "T_cold": 30.0, "part_load_ratio": plrs})
        fig.add_trace(go.Scatter(x=plrs, y=np.asarray(r["efficiency"]) * 100,
                                 name=f"T_hot={T_hot}C PLR", showlegend=True), row=2, col=1)

    # Panel 4: Heatmap
    T_hot_g = np.linspace(80, 280, 50)
    T_cold_g = np.linspace(15, 50, 50)
    eta_map = np.zeros((50, 50))
    for i, th in enumerate(T_hot_g):
        r = model.predict({"T_hot": th, "T_cold": T_cold_g, "part_load_ratio": 1.0})
        eta_map[i, :] = np.asarray(r["efficiency"]) * 100
    fig.add_trace(go.Heatmap(
        x=T_cold_g, y=T_hot_g, z=eta_map,
        colorscale="Viridis", colorbar=dict(title="eta (%)", x=1.02),
        name="Efficiency"), row=2, col=2)

    fig.update_xaxes(title_text="T_hot (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_cold (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="T_cold (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="T_hot (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Efficiency Curve",
        height=850,
        template="plotly_white",
        legend=dict(orientation="v", x=1.08, y=0.95),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
