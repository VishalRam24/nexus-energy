"""EC128 — Conventional Hydro Dam — F1b Head-Flow — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import HydroelectricDamF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = HydroelectricDamF1b(params)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Francis: Efficiency Hill Chart (Q vs H)",
            "Efficiency vs Flow Ratio for Three Turbine Types",
            "Power vs Flow Rate at Different Heads (Francis)",
            "Power vs Head at Design Flow (All Turbines)",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: Francis efficiency heatmap (Q vs H)
    Q_arr = np.linspace(5, 60, 80)
    H_arr = np.linspace(50, 150, 80)
    QQ, HH = np.meshgrid(Q_arr, H_arr)
    eta_map = m.overall_efficiency(QQ.ravel(), HH.ravel(), "francis").reshape(80, 80)
    fig.add_trace(
        go.Heatmap(
            x=Q_arr, y=H_arr, z=eta_map * 100,
            colorscale="RdYlGn", colorbar=dict(title="Efficiency (%)"),
            zmin=50, zmax=95,
        ),
        row=1, col=1,
    )

    # Panel 2: Efficiency vs flow ratio for each turbine type
    colors = {"francis": "#636EFA", "kaplan": "#EF553B", "pelton": "#00CC96"}
    for ttype in ["francis", "kaplan", "pelton"]:
        t = m.turbines[ttype]
        q_range = np.linspace(0.05, 1.3, 200)
        Q_range = q_range * t["Q_rated"]
        H_design = t["H_rated"]
        eta = m.overall_efficiency(Q_range, H_design, ttype)
        fig.add_trace(go.Scatter(
            x=q_range, y=eta * 100,
            name=f"{ttype.capitalize()}", line=dict(color=colors[ttype], width=2.5),
        ), row=1, col=2)

    # Panel 3: Power vs flow at different heads (Francis)
    Q_francis = np.linspace(5, 60, 200)
    head_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, H in enumerate([60, 80, 100, 125]):
        P = m.power_kw(Q_francis, H, "francis")
        fig.add_trace(go.Scatter(
            x=Q_francis, y=P / 1000.0,
            name=f"H={H}m", line=dict(color=head_colors[i], width=2),
        ), row=2, col=1)

    # Panel 4: Power vs head at design flow for each turbine
    for ttype in ["francis", "kaplan", "pelton"]:
        t = m.turbines[ttype]
        H_range = np.linspace(t["H_rated"] * t["h_min"],
                              t["H_rated"] * t["h_max"], 200)
        P = m.power_kw(t["Q_rated"], H_range, ttype)
        fig.add_trace(go.Scatter(
            x=H_range, y=P / 1000.0,
            name=f"{ttype.capitalize()} (Q_rated)", line=dict(color=colors[ttype], width=2),
        ), row=2, col=2)

    fig.update_xaxes(title_text="Flow Rate (m3/s)", row=1, col=1)
    fig.update_xaxes(title_text="Flow Ratio (Q/Q_rated)", row=1, col=2)
    fig.update_xaxes(title_text="Flow Rate (m3/s)", row=2, col=1)
    fig.update_xaxes(title_text="Head (m)", row=2, col=2)
    fig.update_yaxes(title_text="Head (m)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
