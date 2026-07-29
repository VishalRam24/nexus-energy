"""EC222 — Betavoltaic Cell — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    params = model.params

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Power Output vs Time (Ni-63)",
            "Activity Decay vs Time",
            "Isotope Comparison: Power vs Time",
            "Fraction Remaining vs Time (log scale)",
        ],
        vertical_spacing=0.14)

    t = np.linspace(0.0, 300.0, 500)

    # Plot 1: Power vs time (main isotope)
    r = model.predict({"t_years": t})
    fig.add_trace(go.Scatter(x=t, y=r["P_out_uW"],
        name="Ni-63 P_out", line=dict(color="firebrick")), row=1, col=1)

    # Plot 2: Activity decay
    fig.add_trace(go.Scatter(x=t, y=r["activity_Bq"] / 1e9,
        name="Activity (GBq)", line=dict(color="steelblue")), row=1, col=2)

    # Plot 3: Compare isotopes
    isotopes_data = {
        "Tritium":  {"t_half": 12.32, "E": 0.005685},
        "Ni-63":    {"t_half": 100.2, "E": 0.017},
        "Pm-147":   {"t_half": 2.623, "E": 0.062},
    }
    A0 = params["unit"]["A0_Bq"]["value"]
    eta_cap = params["unit"]["eta_capture"]["value"]
    eta_conv = params["unit"]["eta_conv"]["value"]
    MeV_to_J = 1.602176634e-13
    t_iso = np.linspace(0.0, 150.0, 300)
    for name, iso in isotopes_data.items():
        A = A0 * np.exp(-np.log(2) * t_iso / iso["t_half"])
        P = A * iso["E"] * MeV_to_J * eta_cap * eta_conv * 1e6
        fig.add_trace(go.Scatter(x=t_iso, y=P, name=name), row=2, col=1)

    # Plot 4: Fraction remaining (log scale)
    fig.add_trace(go.Scatter(x=t, y=r["fraction_remaining"],
        name="Ni-63 fraction", line=dict(color="purple")), row=2, col=2)

    fig.update_xaxes(title_text="Time (years)", row=1, col=1)
    fig.update_xaxes(title_text="Time (years)", row=1, col=2)
    fig.update_xaxes(title_text="Time (years)", row=2, col=1)
    fig.update_xaxes(title_text="Time (years)", row=2, col=2)
    fig.update_yaxes(title_text="Power (uW)", row=1, col=1)
    fig.update_yaxes(title_text="Activity (GBq)", row=1, col=2)
    fig.update_yaxes(title_text="Power (uW)", row=2, col=1)
    fig.update_yaxes(title_text="Fraction Remaining", row=2, col=2)
    fig.update_yaxes(type="log", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
