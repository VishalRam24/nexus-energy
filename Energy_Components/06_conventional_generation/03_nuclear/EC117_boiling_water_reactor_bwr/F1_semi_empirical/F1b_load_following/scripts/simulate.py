"""EC117 -- BWR -- F1b -- Simulation Scenarios + HTML Report"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def run_simulations():
    model = ComponentModel()
    m = model._model

    # Xenon transient: step from 100% to various levels
    t_arr = np.linspace(0, 72, 200)
    power_steps = {
        "100% -> 75%": (1.0, 0.75),
        "100% -> 80%": (1.0, 0.80),
        "100% -> 60%": (1.0, 0.60),
    }
    xe_transients = {}
    for label, (P_prev, P_new) in power_steps.items():
        xe_transients[label] = [m.xenon_transient(P_prev, P_new, t) for t in t_arr]

    # Equilibrium Xe vs power
    PLR_arr = np.linspace(0.6, 1.0, 50)
    xe_eq = [float(m.equilibrium_xenon(p)) for p in PLR_arr]

    # Available reactivity after step from 100% to 75%
    avail_pcm = []
    for t in t_arr:
        Xe = m.xenon_transient(1.0, 0.75, t)
        avail = m.available_reactivity_pcm(Xe, 0.75)
        avail_pcm.append(avail)

    # Ramp rate scenario: BWR (1%/min) vs PWR (5%/min) comparison
    t_ramp = np.linspace(0, 40, 200)
    P_bwr = [min(0.6 + m.ramp_limit / 100.0 * t, 1.0) for t in t_ramp]
    P_pwr = [min(0.6 + 5.0 / 100.0 * t, 1.0) for t in t_ramp]

    return t_arr, xe_transients, PLR_arr, xe_eq, avail_pcm, t_ramp, P_bwr, P_pwr


def generate_html_report():
    t_arr, xe_transients, PLR_arr, xe_eq, avail_pcm, t_ramp, P_bwr, P_pwr = run_simulations()

    if not HAS_PLOTLY:
        print("plotly not installed -- skipping HTML report generation")
        return

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Xenon Transient After Power Reduction",
            "Equilibrium Xenon vs Power Fraction",
            "Available Reactivity After 100%->75% Step",
            "Ramp Rate: BWR (1%/min) vs PWR (5%/min)",
        ])

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for (label, xe_t), col in zip(xe_transients.items(), colors):
        fig.add_trace(go.Scatter(x=t_arr, y=xe_t, name=label, line=dict(color=col)),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=PLR_arr, y=xe_eq, name="Xe_eq(PLR)",
                              line=dict(color="#9467bd")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_arr, y=avail_pcm, name="Available reactivity",
                              line=dict(color="#ff7f0e")), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    fig.add_trace(go.Scatter(x=t_ramp, y=P_bwr, name="BWR (1%/min)",
                              line=dict(color="#1f77b4")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_ramp, y=P_pwr, name="PWR (5%/min)",
                              line=dict(color="#2ca02c", dash="dash")), row=2, col=2)

    fig.update_xaxes(title_text="Time [hours]", row=1, col=1)
    fig.update_xaxes(title_text="Power Fraction [-]", row=1, col=2)
    fig.update_xaxes(title_text="Time [hours]", row=2, col=1)
    fig.update_xaxes(title_text="Time [minutes]", row=2, col=2)
    fig.update_yaxes(title_text="Xe (relative to eq at 100%)", row=1, col=1)
    fig.update_yaxes(title_text="Xe_eq (normalized)", row=1, col=2)
    fig.update_yaxes(title_text="Available Reactivity [pcm]", row=2, col=1)
    fig.update_yaxes(title_text="Power Fraction [-]", row=2, col=2)

    fig.update_layout(
        title="EC117 BWR -- F1b Load-Following: Xenon Dynamics + Void Feedback",
        height=700, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report written to {out}")


if __name__ == "__main__":
    model = ComponentModel()
    print("EC117 F1b -- Full power equilibrium:")
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    for k, v in r.items():
        print(f"  {k}: {v}")
    generate_html_report()
