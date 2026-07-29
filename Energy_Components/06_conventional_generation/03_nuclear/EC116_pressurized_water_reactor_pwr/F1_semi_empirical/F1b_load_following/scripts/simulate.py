"""EC116 -- PWR -- F1b Load Following -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import PWRF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    pwr = PWRF1b(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Xe Transient After 100%->50% Step",
            "Xe Transient After 100%->30% Step",
            "Available Reactivity vs Time",
            "Power Output vs Power Fraction",
            "Equilibrium Xenon vs Power",
            "Ramp Rate Constraint Example",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # --- 1) Xe transient: 100% -> 50% ---
    t_hours = np.linspace(0, 72, 500)
    Xe_50 = [pwr.xenon_transient(1.0, 0.5, t) for t in t_hours]
    fig.add_trace(
        go.Scatter(x=t_hours, y=Xe_50, name="100%->50%",
                   line=dict(color="firebrick", width=2)),
        row=1, col=1,
    )
    fig.add_hline(y=float(pwr.equilibrium_xenon(0.5)), line_dash="dash",
                  line_color="gray", annotation_text="Eq at 50%", row=1, col=1)

    # --- 2) Xe transient: 100% -> 30% ---
    Xe_30 = [pwr.xenon_transient(1.0, 0.3, t) for t in t_hours]
    fig.add_trace(
        go.Scatter(x=t_hours, y=Xe_30, name="100%->30%",
                   line=dict(color="steelblue", width=2)),
        row=1, col=2,
    )
    fig.add_hline(y=float(pwr.equilibrium_xenon(0.3)), line_dash="dash",
                  line_color="gray", annotation_text="Eq at 30%", row=1, col=2)

    # --- 3) Available reactivity after various step-downs ---
    for P_new, color in [(0.5, "firebrick"), (0.3, "steelblue"), (0.7, "green")]:
        avail = [pwr.available_reactivity_pcm(pwr.xenon_transient(1.0, P_new, t))
                 for t in t_hours]
        fig.add_trace(
            go.Scatter(x=t_hours, y=avail, name=f"100%->{int(P_new*100)}%",
                       line=dict(width=2)),
            row=1, col=3,
        )
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="Restart limit", row=1, col=3)

    # --- 4) Power output vs PLR ---
    PLR_range = np.linspace(0.3, 1.0, 50)
    P_out = [pwr.P_thermal * p * pwr.eta for p in PLR_range]
    fig.add_trace(
        go.Scatter(x=PLR_range, y=P_out, name="P_electric (MW)",
                   line=dict(color="darkorange", width=2)),
        row=2, col=1,
    )

    # --- 5) Equilibrium Xe vs power ---
    PLR_range2 = np.linspace(0.0, 1.0, 100)
    Xe_eq = [float(pwr.equilibrium_xenon(p)) for p in PLR_range2]
    fig.add_trace(
        go.Scatter(x=PLR_range2, y=Xe_eq, name="Equilibrium Xe",
                   line=dict(color="purple", width=2)),
        row=2, col=2,
    )

    # --- 6) Ramp rate constraint ---
    t_ramp = np.linspace(0, 30, 100)  # minutes
    power_ramp = []
    P_current = 0.5
    for t in t_ramp:
        P_achievable, _ = pwr.ramp_rate_limit(0.5, 1.0, t)
        power_ramp.append(P_achievable)
    fig.add_trace(
        go.Scatter(x=t_ramp, y=power_ramp, name="Achievable power",
                   line=dict(color="green", width=2)),
        row=2, col=3,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Target: 100%", row=2, col=3)

    # Axes
    fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=3)
    fig.update_xaxes(title_text="Power Fraction", row=2, col=1)
    fig.update_xaxes(title_text="Power Fraction", row=2, col=2)
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=3)
    fig.update_yaxes(title_text="Xe (relative)", row=1, col=1)
    fig.update_yaxes(title_text="Xe (relative)", row=1, col=2)
    fig.update_yaxes(title_text="Reactivity (pcm)", row=1, col=3)
    fig.update_yaxes(title_text="Power (MW_e)", row=2, col=1)
    fig.update_yaxes(title_text="Xe (relative)", row=2, col=2)
    fig.update_yaxes(title_text="Power Fraction", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Load Following<br>"
            f"<sup>P_th=3400 MW | eta=0.33 | Ramp limit 5%/min | "
            f"Xe-135/I-135 dynamics</sup>"
        ),
        height=900,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    generate_report()
