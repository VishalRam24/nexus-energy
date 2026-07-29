"""EC112 -- Micro Gas Turbine -- F1b -- Simulation Scenarios + HTML Report"""
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

ISO_T = 288.15


def run_simulations():
    model = ComponentModel()
    PLR = np.linspace(0.3, 1.0, 60)

    T_scenarios = {"-10C (263K)": 263.15, "ISO 15C (288K)": 288.15,
                   "30C (303K)": 303.15, "45C (318K)": 318.15}
    results_T = {label: model.predict({"PLR": PLR, "T_ambient": T})
                 for label, T in T_scenarios.items()}

    T_range_k = np.linspace(248.15, 323.15, 60)
    r_T_sens = model.predict({"PLR": 1.0, "T_ambient": T_range_k})

    alt_range = np.linspace(0, 3000, 60)
    r_alt_sens = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "altitude_m": alt_range})

    return PLR, results_T, T_range_k, r_T_sens, alt_range, r_alt_sens


def generate_html_report():
    PLR, results_T, T_range_k, r_T_sens, alt_range, r_alt_sens = run_simulations()

    if not HAS_PLOTLY:
        print("plotly not installed -- skipping HTML report generation")
        return

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs PLR (varying T_amb)",
            "Power vs PLR (varying T_amb)",
            "Efficiency vs Ambient Temperature (PLR=1, ISO pressure)",
            "Power vs Altitude (PLR=1, ISO T & P)",
        ])

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for (label, r), col in zip(results_T.items(), colors):
        fig.add_trace(go.Scatter(x=PLR, y=r["efficiency_electrical"],
                                 name=label, line=dict(color=col), legendgroup=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=PLR, y=r["power_electrical_kw"],
                                 name=label, line=dict(color=col), legendgroup=label,
                                 showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=T_range_k - 273.15, y=r_T_sens["efficiency_electrical"],
                              name="eta vs T", line=dict(color="#9467bd")), row=2, col=1)
    fig.add_trace(go.Scatter(x=alt_range, y=r_alt_sens["power_electrical_kw"],
                              name="P vs altitude", line=dict(color="#8c564b")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=1, col=2)
    fig.update_xaxes(title_text="Ambient Temperature [degC]", row=2, col=1)
    fig.update_xaxes(title_text="Altitude [m]", row=2, col=2)
    fig.update_yaxes(title_text="eta_el [-]", row=1, col=1)
    fig.update_yaxes(title_text="P_el [kW]", row=1, col=2)
    fig.update_yaxes(title_text="eta_el [-]", row=2, col=1)
    fig.update_yaxes(title_text="P_el [kW]", row=2, col=2)

    fig.update_layout(
        title="EC112 Micro Gas Turbine -- F1b Part-Load + Ambient (strong ~0.01/K) + Altitude",
        height=700, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report written to {out}")


if __name__ == "__main__":
    model = ComponentModel()
    print("EC112 F1b -- ISO conditions:")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    generate_html_report()
