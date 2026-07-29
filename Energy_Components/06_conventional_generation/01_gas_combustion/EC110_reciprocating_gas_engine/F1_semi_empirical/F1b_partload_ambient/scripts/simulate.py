"""EC110 -- Reciprocating Gas Engine -- F1b -- Simulation Scenarios + HTML Report"""
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
    PLR = np.linspace(0.5, 1.0, 60)

    T_scenarios = {"-10C": -10.0, "25C (ref)": 25.0, "40C": 40.0}
    alt_scenarios = {"Sea level (0m)": 0.0, "1000m": 1000.0, "2000m": 2000.0}

    results_T = {}
    for label, T in T_scenarios.items():
        results_T[label] = model.predict({"PLR": PLR, "T_ambient": T, "altitude_m": 0.0})

    results_alt = {}
    for label, alt in alt_scenarios.items():
        results_alt[label] = model.predict({"PLR": PLR, "T_ambient": 25.0, "altitude_m": alt})

    T_range = np.linspace(-20, 50, 60)
    r_T_sens = model.predict({"PLR": 1.0, "T_ambient": T_range, "altitude_m": 0.0})

    alt_range = np.linspace(0, 3000, 60)
    r_alt_sens = model.predict({"PLR": 1.0, "T_ambient": 25.0, "altitude_m": alt_range})

    return PLR, results_T, results_alt, T_range, r_T_sens, alt_range, r_alt_sens


def generate_html_report():
    PLR, results_T, results_alt, T_range, r_T_sens, alt_range, r_alt_sens = run_simulations()

    if not HAS_PLOTLY:
        print("plotly not installed -- skipping HTML report generation")
        return

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Electrical Efficiency vs PLR (varying T_amb)",
            "Power Output vs PLR (varying Altitude)",
            "Power vs Ambient Temperature (PLR=1, sea level)",
            "Power vs Altitude (PLR=1, T=25C)",
        ])

    colors_T = ["#1f77b4", "#2ca02c", "#d62728"]
    for (label, r), col in zip(results_T.items(), colors_T):
        fig.add_trace(go.Scatter(x=PLR, y=r["efficiency_electrical"],
                                 name=label, line=dict(color=col), legendgroup="T"),
                      row=1, col=1)

    colors_alt = ["#1f77b4", "#ff7f0e", "#9467bd"]
    for (label, r), col in zip(results_alt.items(), colors_alt):
        fig.add_trace(go.Scatter(x=PLR, y=r["power_electrical_kw"],
                                 name=label, line=dict(color=col), legendgroup="alt"),
                      row=1, col=2)

    fig.add_trace(go.Scatter(x=T_range, y=r_T_sens["power_electrical_kw"],
                              name="P vs T", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=alt_range, y=r_alt_sens["power_electrical_kw"],
                              name="P vs altitude", line=dict(color="#8c564b")), row=2, col=2)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=1, col=2)
    fig.update_xaxes(title_text="Ambient Temperature [degC]", row=2, col=1)
    fig.update_xaxes(title_text="Altitude [m]", row=2, col=2)
    fig.update_yaxes(title_text="eta_el [-]", row=1, col=1)
    fig.update_yaxes(title_text="P_el [kW]", row=1, col=2)
    fig.update_yaxes(title_text="P_el [kW]", row=2, col=1)
    fig.update_yaxes(title_text="P_el [kW]", row=2, col=2)

    fig.update_layout(
        title="EC110 Reciprocating Gas Engine -- F1b Part-Load + Altitude + Ambient",
        height=700, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report written to {out}")


if __name__ == "__main__":
    model = ComponentModel()
    print("EC110 F1b -- Standard conditions (PLR=1.0, 25C, sea level):")
    r = model.predict({"PLR": 1.0})
    for k, v in r.items():
        print(f"  {k}: {float(v):.4f}")
    generate_html_report()
