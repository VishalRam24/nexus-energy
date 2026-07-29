"""EC139 — Salinity Gradient PRO — F1a — Simulation Scenarios + HTML Report"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulation():
    model = ComponentModel()

    # Scenario 1: Net energy vs seawater salinity (C_fw fixed)
    C_sw_vals = np.linspace(25.0, 40.0, 40)
    r_sw = model.predict({"C_sw": C_sw_vals, "C_fw": 0.5})

    # Scenario 2: Net energy vs freshwater salinity (C_sw fixed)
    C_fw_vals = np.linspace(0.1, 5.0, 40)
    r_fw = model.predict({"C_sw": 35.0, "C_fw": C_fw_vals})

    # Scenario 3: Osmotic pressure map
    C_sw_g, C_fw_g = np.meshgrid(np.linspace(25, 40, 20), np.linspace(0.1, 3.0, 20))
    r_map = model.predict({"C_sw": C_sw_g, "C_fw": C_fw_g})

    # Scenario 4: Power vs flow rate
    Q_vals = np.linspace(0.1, 10.0, 40)
    r_Q = model.predict({"C_sw": 35.0, "C_fw": 0.5, "Q_feed_m3s": Q_vals})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Net Energy vs C_sw (C_fw=0.5 g/L)",
                                            "Net Energy vs C_fw (C_sw=35 g/L)",
                                            "Osmotic Pressure Map [bar]",
                                            "Power vs Feed Flow Rate"])

        fig.add_trace(go.Scatter(x=C_sw_vals, y=r_sw["net_energy_kwh_per_m3"],
                                  name="net energy", line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=C_sw_vals, y=r_sw["gibbs_energy_kwh_per_m3"],
                                  name="Gibbs (theoretical)", line=dict(dash="dash")), row=1, col=1)

        fig.add_trace(go.Scatter(x=C_fw_vals, y=r_fw["net_energy_kwh_per_m3"],
                                  name="net energy", line=dict(color="#ff7f0e")), row=1, col=2)

        fig.add_trace(go.Heatmap(z=r_map["osmotic_pressure_bar"],
                                  x=np.linspace(25, 40, 20),
                                  y=np.linspace(0.1, 3.0, 20),
                                  colorscale="Blues", showscale=True), row=2, col=1)

        fig.add_trace(go.Scatter(x=Q_vals, y=r_Q["power_kw"],
                                  name="power", line=dict(color="#2ca02c")), row=2, col=2)

        for col, title in [(1, "Net energy [kWh/m³]"), (2, "Net energy [kWh/m³]")]:
            fig.update_yaxes(title_text=title, row=1, col=col)
        fig.update_xaxes(title_text="C_sw [g/L]", row=1, col=1)
        fig.update_xaxes(title_text="C_fw [g/L]", row=1, col=2)
        fig.update_xaxes(title_text="C_sw [g/L]", row=2, col=1)
        fig.update_yaxes(title_text="C_fw [g/L]", row=2, col=1)
        fig.update_xaxes(title_text="Q_feed [m³/s]", row=2, col=2)
        fig.update_yaxes(title_text="Power [kW]", row=2, col=2)

        fig.update_layout(title="EC139 Salinity Gradient PRO F1a — Simulation Report",
                          height=800, width=1200)

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not available — skipping HTML report.")

    r_ref = model.predict({})
    print("\n=== EC139 Salinity Gradient PRO — Summary (C_sw=35g/L, C_fw=0.5g/L) ===")
    print(f"  Osmotic pressure:  {float(r_ref['osmotic_pressure_bar']):.1f} bar")
    print(f"  Gibbs energy:      {float(r_ref['gibbs_energy_kwh_per_m3']):.4f} kWh/m³")
    print(f"  Net energy:        {float(r_ref['net_energy_kwh_per_m3']):.4f} kWh/m³")
    print(f"  Power (1 m³/s):    {float(r_ref['power_kw']):.2f} kW")


if __name__ == "__main__":
    run_simulation()
