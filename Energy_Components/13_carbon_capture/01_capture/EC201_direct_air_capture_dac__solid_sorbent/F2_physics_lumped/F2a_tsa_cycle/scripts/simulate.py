"""
EC201 -- Direct Air Capture (DAC) Solid Sorbent -- F2a TSA Cycle -- Simulation scenarios.

Generates an interactive HTML report with Plotly showing:
  1. Langmuir isotherm: q vs P_CO2 at different temperatures
  2. Working capacity vs desorption temperature
  3. Specific energy consumption vs desorption temperature
  4. Productivity vs cycle parameters
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TSACycleModel

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


def run_simulations():
    params = load_params()
    m = TSACycleModel(params)

    # ------------------------------------------------------------------
    # Scenario 1: Langmuir isotherm at different temperatures
    # ------------------------------------------------------------------
    P_range_kPa = np.linspace(0.001, 0.1, 200)
    temps_degC = [15, 25, 50, 80, 100, 120]
    isotherms = {}
    for T in temps_degC:
        isotherms[T] = m.loading(T + 273.15, P_range_kPa)

    # ------------------------------------------------------------------
    # Scenario 2: Working capacity vs T_des
    # ------------------------------------------------------------------
    T_des_range = np.linspace(60, 150, 50)
    dq_vs_Tdes = np.array([m.working_capacity(T_des_degC=Td) for Td in T_des_range])

    # ------------------------------------------------------------------
    # Scenario 3: Specific energy vs T_des
    # ------------------------------------------------------------------
    sec_th = np.array([m.specific_thermal_energy_GJ_tCO2(T_des_degC=Td) for Td in T_des_range])
    sec_el = np.array([m.specific_electrical_energy_GJ_tCO2(T_des_degC=Td) for Td in T_des_range])
    sec_total = sec_th + sec_el

    # ------------------------------------------------------------------
    # Scenario 4: Productivity vs T_des
    # ------------------------------------------------------------------
    prod = np.array([m.productivity_kg_h(T_des_degC=Td) for Td in T_des_range])

    # ------------------------------------------------------------------
    # Scenario 5: Effect of vacuum pressure on working capacity
    # ------------------------------------------------------------------
    P_vac_range = np.linspace(0.05, 1.0, 50)
    dq_vs_Pvac = np.array([m.working_capacity(P_vac_atm=pv) for pv in P_vac_range])

    # ------------------------------------------------------------------
    # Scenario 6: Default compute summary
    # ------------------------------------------------------------------
    default_results = m.compute()

    return {
        "P_range_kPa": P_range_kPa,
        "isotherms": isotherms,
        "temps_degC": temps_degC,
        "T_des_range": T_des_range,
        "dq_vs_Tdes": dq_vs_Tdes,
        "sec_th": sec_th,
        "sec_el": sec_el,
        "sec_total": sec_total,
        "prod": prod,
        "P_vac_range": P_vac_range,
        "dq_vs_Pvac": dq_vs_Pvac,
        "default_results": default_results,
    }


def generate_html(data):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed -- skipping HTML report generation.")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Langmuir Isotherm: q vs P_CO2",
            "Working Capacity vs T_des",
            "Specific Energy vs T_des",
            "Productivity vs T_des",
            "Working Capacity vs Vacuum Pressure",
            "Energy Breakdown (Default Conditions)",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    # Plot 1: Isotherms
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, T in enumerate(data["temps_degC"]):
        fig.add_trace(go.Scatter(
            x=data["P_range_kPa"], y=data["isotherms"][T],
            name=f"{T} C", line=dict(color=colors[i % len(colors)]),
            legendgroup="iso", showlegend=True,
        ), row=1, col=1)
    # Mark ambient CO2
    fig.add_vline(x=0.042, line_dash="dash", line_color="gray",
                  annotation_text="420 ppm", row=1, col=1)
    fig.update_xaxes(title_text="P_CO2 [kPa]", row=1, col=1)
    fig.update_yaxes(title_text="q [mmol/g]", row=1, col=1)

    # Plot 2: Working capacity vs T_des
    fig.add_trace(go.Scatter(
        x=data["T_des_range"], y=data["dq_vs_Tdes"],
        name="Working capacity", line=dict(color="#2ca02c"),
        showlegend=False,
    ), row=1, col=2)
    fig.update_xaxes(title_text="T_des [degC]", row=1, col=2)
    fig.update_yaxes(title_text="delta_q [mmol/g]", row=1, col=2)

    # Plot 3: SEC vs T_des
    fig.add_trace(go.Scatter(
        x=data["T_des_range"], y=data["sec_th"],
        name="Thermal", line=dict(color="#d62728"),
        legendgroup="sec", showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data["T_des_range"], y=data["sec_el"],
        name="Electrical", line=dict(color="#1f77b4"),
        legendgroup="sec", showlegend=True,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data["T_des_range"], y=data["sec_total"],
        name="Total SEC", line=dict(color="#333", dash="dash"),
        legendgroup="sec", showlegend=True,
    ), row=2, col=1)
    fig.update_xaxes(title_text="T_des [degC]", row=2, col=1)
    fig.update_yaxes(title_text="SEC [GJ/tCO2]", row=2, col=1)

    # Plot 4: Productivity vs T_des
    fig.add_trace(go.Scatter(
        x=data["T_des_range"], y=data["prod"],
        name="Productivity", line=dict(color="#ff7f0e"),
        showlegend=False,
    ), row=2, col=2)
    fig.update_xaxes(title_text="T_des [degC]", row=2, col=2)
    fig.update_yaxes(title_text="Productivity [kg CO2/h]", row=2, col=2)

    # Plot 5: Working capacity vs vacuum pressure
    fig.add_trace(go.Scatter(
        x=data["P_vac_range"], y=data["dq_vs_Pvac"],
        name="dq vs P_vac", line=dict(color="#9467bd"),
        showlegend=False,
    ), row=3, col=1)
    fig.update_xaxes(title_text="P_vac [atm]", row=3, col=1)
    fig.update_yaxes(title_text="delta_q [mmol/g]", row=3, col=1)

    # Plot 6: Energy breakdown bar chart
    dr = data["default_results"]
    fig.add_trace(go.Bar(
        x=["Thermal", "Electrical", "Total"],
        y=[dr["specific_thermal_GJ_tCO2"], dr["specific_electrical_GJ_tCO2"],
           dr["total_SEC_GJ_tCO2"]],
        marker_color=["#d62728", "#1f77b4", "#333"],
        showlegend=False,
    ), row=3, col=2)
    fig.update_xaxes(title_text="", row=3, col=2)
    fig.update_yaxes(title_text="SEC [GJ/tCO2]", row=3, col=2)

    fig.update_layout(
        title_text="EC201 DAC Solid Sorbent -- F2a TSA Cycle Model",
        height=1100, width=1100,
        template="plotly_white",
    )

    fig.write_html(_REPORT_PATH)
    print(f"Report saved to {_REPORT_PATH}")


if __name__ == "__main__":
    print("Running simulation scenarios ...")
    data = run_simulations()

    dr = data["default_results"]
    print(f"\n--- Default Conditions ---")
    for k, v in dr.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    generate_html(data)
