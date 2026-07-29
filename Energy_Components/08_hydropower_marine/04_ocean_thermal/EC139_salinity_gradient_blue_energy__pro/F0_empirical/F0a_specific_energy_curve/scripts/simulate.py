"""EC139 F0a — optional Plotly report (specific energy & power vs seawater conc.)."""
import numpy as np
from predict import ComponentModel


def main():
    m = ComponentModel()
    C = np.linspace(25, 40, 60)
    r = m.predict({"C_seawater_g_per_L": C})
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=C, y=r["specific_energy_kWh_per_m3"], name="Specific energy (kWh/m3)"))
        fig.add_trace(go.Scatter(x=C, y=r["net_power_kw"], name="Net power (kW)"), secondary_y=True)
        fig.update_layout(title="EC139 F0a — Salinity gradient (PRO) vs seawater conc.",
                          xaxis_title="Seawater concentration (g/L)")
        fig.write_html("simulation_report.html")
        print("wrote simulation_report.html")
    except ImportError:
        print("plotly not installed; numeric summary:")
        for c, se, p in zip(C[::10], r["specific_energy_kWh_per_m3"][::10], r["net_power_kw"][::10]):
            print(f"  Csw={c:5.1f} g/L  SE={se:.4f} kWh/m3  P={p:6.1f} kW")


if __name__ == "__main__":
    main()
