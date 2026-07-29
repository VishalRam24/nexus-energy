"""EC074 — Plate HX — F1b Fouling — Simulation & HTML Report"""
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
            "Q vs Fouling Resistance (both sides equal)",
            "Effectiveness vs Rf",
            "U_fouled vs Rf",
            "Effectiveness Reduction vs Rf at various flow rates",
        ],
        vertical_spacing=0.12,
    )

    Rf_range = np.linspace(0.0, 0.005, 100)

    # Row 1 Col 1 — Q vs Rf
    base = {"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0, "m_dot_cold": 1.0}
    Q_vals, eps_vals, U_vals = [], [], []
    for rf in Rf_range:
        r = model.predict({**base, "fouling_resistance_hot": rf, "fouling_resistance_cold": rf})
        Q_vals.append(float(r["Q_kw"]))
        eps_vals.append(float(r["effectiveness"]))
        U_vals.append(float(r["U_fouled"]))

    fig.add_trace(
        go.Scatter(x=Rf_range * 1000, y=Q_vals, name="Q", line=dict(color="steelblue")),
        row=1, col=1,
    )

    # Row 1 Col 2 — Effectiveness vs Rf
    fig.add_trace(
        go.Scatter(x=Rf_range * 1000, y=eps_vals, name="eff", line=dict(color="darkorange"),
                   showlegend=False),
        row=1, col=2,
    )

    # Row 2 Col 1 — U_fouled vs Rf
    fig.add_trace(
        go.Scatter(x=Rf_range * 1000, y=U_vals, name="U_f", line=dict(color="crimson"),
                   showlegend=False),
        row=2, col=1,
    )

    # Row 2 Col 2 — Effectiveness reduction at various flow rates
    for mdot in [0.5, 1.0, 2.0, 3.0]:
        eps_red = []
        for rf in Rf_range:
            r = model.predict({
                "T_h_in": 80.0, "T_c_in": 20.0,
                "m_dot_hot": mdot, "m_dot_cold": mdot,
                "fouling_resistance_hot": rf, "fouling_resistance_cold": rf,
            })
            eps_red.append(float(r["effectiveness_reduction"]))
        fig.add_trace(
            go.Scatter(x=Rf_range * 1000, y=np.array(eps_red) * 100,
                       name=f"mdot={mdot}"),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Rf (x10^-3 m2K/W)", row=1, col=1)
    fig.update_xaxes(title_text="Rf (x10^-3 m2K/W)", row=1, col=2)
    fig.update_xaxes(title_text="Rf (x10^-3 m2K/W)", row=2, col=1)
    fig.update_xaxes(title_text="Rf (x10^-3 m2K/W)", row=2, col=2)
    fig.update_yaxes(title_text="Q (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Effectiveness", row=1, col=2)
    fig.update_yaxes(title_text="U_fouled (W/m2K)", row=2, col=1)
    fig.update_yaxes(title_text="Effectiveness Reduction (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | Fouling Analysis",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
