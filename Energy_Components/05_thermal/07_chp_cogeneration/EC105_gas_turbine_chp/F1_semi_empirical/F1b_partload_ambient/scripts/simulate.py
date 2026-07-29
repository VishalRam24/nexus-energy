"""EC105 -- Gas Turbine CHP -- F1b -- Simulation Scenarios + HTML Report"""
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

ISO_T = 288.15   # 15 degC ISO reference


def run_simulations():
    model = ComponentModel()
    PLR = np.linspace(0.4, 1.0, 60)
    T_scenarios = {"-10C (263K)": 263.15, "ISO 15C (288K)": 288.15,
                   "30C (303K)": 303.15, "45C (318K)": 318.15}

    results = {}
    for label, T in T_scenarios.items():
        r = model.predict({"PLR": PLR, "T_ambient": T})
        results[label] = r

    # ISO full-load point
    r_iso = model.predict({"PLR": 1.0, "T_ambient": ISO_T, "P_ambient": 101.325})

    # T sensitivity at full load
    T_range = np.linspace(253.15, 323.15, 60)
    r_T = model.predict({"PLR": 1.0, "T_ambient": T_range})

    # Exhaust T vs PLR
    r_exh = model.predict({"PLR": PLR, "T_ambient": ISO_T})

    return PLR, results, T_range, r_T, r_exh, r_iso


def generate_html_report():
    PLR, results, T_range, r_T, r_exh, r_iso = run_simulations()

    if not HAS_PLOTLY:
        print("plotly not installed -- skipping HTML report generation")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Electrical Efficiency vs PLR",
            "Total CHP Efficiency vs PLR",
            "Power Output vs Ambient Temperature (PLR=1)",
            "Heat Recovery vs PLR",
            "Exhaust Temperature vs PLR",
            "Heat-to-Power Ratio vs PLR",
        ],
    )

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for (label, r), col in zip(results.items(), colors):
        fig.add_trace(go.Scatter(x=PLR, y=r["efficiency_electrical"],
                                 name=label, line=dict(color=col), legendgroup=label,
                                 showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=PLR, y=r["efficiency_total"],
                                 name=label, line=dict(color=col, dash="dot"), legendgroup=label,
                                 showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=PLR, y=r["heat_recovery_kw"],
                                 name=label, line=dict(color=col), legendgroup=label,
                                 showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=PLR, y=r["heat_to_power_ratio"],
                                 name=label, line=dict(color=col), legendgroup=label,
                                 showlegend=False), row=3, col=2)

    # Power vs ambient T
    fig.add_trace(go.Scatter(x=T_range - 273.15, y=r_T["power_electrical_kw"],
                              name="P_el vs T_amb", line=dict(color="#9467bd")),
                  row=2, col=1)

    # Exhaust temp vs PLR
    fig.add_trace(go.Scatter(x=PLR, y=r_exh["exhaust_temp_K"] - 273.15,
                              name="T_exhaust (ISO)", line=dict(color="#8c564b")),
                  row=3, col=1)

    fig.update_xaxes(title_text="PLR [-]", row=1, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=1, col=2)
    fig.update_xaxes(title_text="Ambient Temperature [degC]", row=2, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=2, col=2)
    fig.update_xaxes(title_text="PLR [-]", row=3, col=1)
    fig.update_xaxes(title_text="PLR [-]", row=3, col=2)
    fig.update_yaxes(title_text="eta_el [-]", row=1, col=1)
    fig.update_yaxes(title_text="eta_total [-]", row=1, col=2)
    fig.update_yaxes(title_text="P_el [kW]", row=2, col=1)
    fig.update_yaxes(title_text="Q_th [kW]", row=2, col=2)
    fig.update_yaxes(title_text="T_exhaust [degC]", row=3, col=1)
    fig.update_yaxes(title_text="HPR [-]", row=3, col=2)

    fig.update_layout(
        title="EC105 Gas Turbine CHP -- F1b Part-Load + Ambient + HRSG Model",
        height=900, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report written to {out}")


if __name__ == "__main__":
    PLR, results, T_range, r_T, r_exh, r_iso = run_simulations()
    print("EC105 F1b -- ISO full load summary:")
    for k, v in r_iso.items():
        print(f"  {k}: {float(v):.4f}")
    generate_html_report()
