"""EC013 — Liquid H2 Storage — F1a — Simulation & HTML Report"""
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
            "Stored LH2 Mass vs Fill Fraction",
            "Boil-off Rate (%/day) vs Fill Fraction",
            "Heat Leak vs Ambient Temperature",
            "Time to Empty vs Fill Fraction",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    f = np.linspace(0.05, 0.95, 200)
    T_vals = [253.15, 298.15, 318.15]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    labels = ["-20 C", "25 C", "45 C"]

    for T, c, lbl in zip(T_vals, colors, labels):
        r = model.predict({"fill_fraction": f, "T_ambient": T})
        fig.add_trace(go.Scatter(x=f, y=r["stored_mass_kg"], name=lbl,
                                  line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=r["boiloff_pct_per_day"], name=lbl,
                                  line=dict(color=c), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=f, y=r["time_to_empty_days"], name=lbl,
                                  line=dict(color=c), showlegend=False), row=2, col=2)

    T_arr = np.linspace(233, 333, 100)
    r_q = model.predict({"fill_fraction": 0.8, "T_ambient": T_arr})
    fig.add_trace(go.Scatter(x=T_arr - 273.15, y=r_q["heat_leak_W"],
                              name="Heat leak", line=dict(color="#2ca02c")), row=2, col=1)

    fig.update_xaxes(title_text="Fill fraction (-)", row=1, col=1)
    fig.update_xaxes(title_text="Fill fraction (-)", row=1, col=2)
    fig.update_xaxes(title_text="Ambient T (C)", row=2, col=1)
    fig.update_xaxes(title_text="Fill fraction (-)", row=2, col=2)
    fig.update_yaxes(title_text="LH2 mass (kg)", row=1, col=1)
    fig.update_yaxes(title_text="BOR (%/day)", row=1, col=2)
    fig.update_yaxes(title_text="Q (W)", row=2, col=1)
    fig.update_yaxes(title_text="Days", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Boil-off Model<br>"
              f"<sup>1 m3 MLI/vacuum dewar | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
