"""EC118 -- SMR -- F1b -- Simulation Scenarios + HTML Report"""
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

    # Xenon transient
    t_arr = np.linspace(0, 72, 200)
    power_steps = {
        "100% -> 20% (deep)": (1.0, 0.2),
        "100% -> 50%":         (1.0, 0.5),
        "100% -> 80%":         (1.0, 0.8),
    }
    xe_transients = {label: [m.xenon_transient(P1, P2, t) for t in t_arr]
                     for label, (P1, P2) in power_steps.items()}

    # Thermal inertia during ramp-down 100% -> 30%
    t_ramp_min = np.linspace(0, 5 * m.tau_min, 100)
    T_outlet_lag = [m.coolant_outlet_temp_transient(1.0, 0.3, t) for t in t_ramp_min]
    T_outlet_target = m.coolant_outlet_temp_steady(0.3)
    T_outlet_initial = m.coolant_outlet_temp_steady(1.0)

    # Ramp rate comparison: SMR (5%/min, range 0.2-1.0) vs BWR (1%/min, range 0.6-1.0)
    t_ramp2 = np.linspace(0, 40, 200)
    P_smr = [min(0.2 + 5.0 / 100.0 * t, 1.0) for t in t_ramp2]
    P_bwr = [min(0.6 + 1.0 / 100.0 * t, 1.0) for t in t_ramp2]
    P_pwr = [min(0.3 + 5.0 / 100.0 * t, 1.0) for t in t_ramp2]

    # Available reactivity after 100% -> 30% step
    avail_pcm = [m.available_reactivity_pcm(m.xenon_transient(1.0, 0.3, t)) for t in t_arr]

    return (t_arr, xe_transients, t_ramp_min, T_outlet_lag, T_outlet_target, T_outlet_initial,
            t_ramp2, P_smr, P_bwr, P_pwr, avail_pcm)


def generate_html_report():
    (t_arr, xe_transients, t_ramp_min, T_outlet_lag, T_outlet_target, T_outlet_initial,
     t_ramp2, P_smr, P_bwr, P_pwr, avail_pcm) = run_simulations()

    if not HAS_PLOTLY:
        print("plotly not installed -- skipping HTML report generation")
        return

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Xenon Transient (deep load-following)",
            "Thermal Inertia: Coolant Outlet T (100% -> 30%)",
            "Available Reactivity After 100%->30%",
            "Ramp Rate Comparison: SMR vs BWR vs Large PWR",
        ])

    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    for (label, xe_t), col in zip(xe_transients.items(), colors):
        fig.add_trace(go.Scatter(x=t_arr, y=xe_t, name=label, line=dict(color=col)),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=t_ramp_min, y=T_outlet_lag, name="T_outlet (actual)",
                              line=dict(color="#1f77b4")), row=1, col=2)
    fig.add_hline(y=T_outlet_target, line_dash="dash", line_color="red",
                  annotation_text=f"Target: {T_outlet_target:.1f}C", row=1, col=2)
    fig.add_hline(y=T_outlet_initial, line_dash="dash", line_color="green",
                  annotation_text=f"Initial: {T_outlet_initial:.1f}C", row=1, col=2)

    fig.add_trace(go.Scatter(x=t_arr, y=avail_pcm, name="Available reactivity",
                              line=dict(color="#9467bd")), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    fig.add_trace(go.Scatter(x=t_ramp2, y=P_smr, name="SMR (5%/min, 20-100%)",
                              line=dict(color="#d62728")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_ramp2, y=P_pwr, name="Large PWR (5%/min, 30-100%)",
                              line=dict(color="#2ca02c", dash="dash")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_ramp2, y=P_bwr, name="BWR (1%/min, 60-100%)",
                              line=dict(color="#1f77b4", dash="dot")), row=2, col=2)

    fig.update_xaxes(title_text="Time [hours]", row=1, col=1)
    fig.update_xaxes(title_text="Time [minutes]", row=1, col=2)
    fig.update_xaxes(title_text="Time [hours]", row=2, col=1)
    fig.update_xaxes(title_text="Time [minutes]", row=2, col=2)
    fig.update_yaxes(title_text="Xe (rel. to full-power eq.)", row=1, col=1)
    fig.update_yaxes(title_text="T_outlet [degC]", row=1, col=2)
    fig.update_yaxes(title_text="Available Reactivity [pcm]", row=2, col=1)
    fig.update_yaxes(title_text="Power Fraction [-]", row=2, col=2)

    fig.update_layout(
        title="EC118 SMR -- F1b Deep Load-Following + Xenon Dynamics + Thermal Inertia",
        height=700, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report written to {out}")


if __name__ == "__main__":
    model = ComponentModel()
    print("EC118 F1b -- Full power equilibrium:")
    r = model.predict({"power_fraction": 1.0, "time_at_power_hours": 48.0,
                       "previous_power_fraction": 1.0})
    for k, v in r.items():
        print(f"  {k}: {v}")
    generate_html_report()
