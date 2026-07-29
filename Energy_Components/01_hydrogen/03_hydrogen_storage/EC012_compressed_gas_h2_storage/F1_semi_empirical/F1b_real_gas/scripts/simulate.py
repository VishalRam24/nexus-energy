"""EC012 — Compressed Gas H2 Storage — F1b Real-Gas — Simulation Scenarios"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def run_scenarios():
    model = ComponentModel()
    m = model._model

    # Scenario 1: Z(P) at multiple temperatures
    P_arr = np.linspace(1, 900, 300)
    temps_K = [240.0, 273.15, 298.15, 323.15, 373.15]

    Z_curves = {}
    m_curves = {}
    for T in temps_K:
        r = model.predict({"P_bar": P_arr, "T_K": T})
        Z_curves[T] = r["Z"]
        m_curves[T] = r["stored_mass_kg"]

    # Scenario 2: Compression work vs pressure ratio
    P2_arr = np.linspace(50, 900, 100)
    comp_works = []
    Z_inlets = []
    for P2 in P2_arr:
        r = model.predict({"mode": "compression", "P1_bar": 30.0, "P2_bar": P2})
        comp_works.append(float(r["compression_work_kJ_per_kg"]))
        Z_inlets.append(float(r["Z_inlet"]))

    # Scenario 3: Fill transient at different T_amb
    T_amb_arr = np.array([253.15, 273.15, 293.15, 313.15, 333.15])
    dT_arr = []
    m_after_arr = []
    for T_amb in T_amb_arr:
        r = model.predict({"mode": "fill", "P1_bar": 20.0, "P2_bar": 700.0, "T_amb_K": float(T_amb)})
        dT_arr.append(float(r["dT_K"]))
        m_after_arr.append(float(r["stored_mass_kg"]))

    # Scenario 4: Cooling transient after fill
    T_post = float(model.predict({"mode": "fill", "P1_bar": 20.0, "P2_bar": 700.0,
                                    "T_amb_K": 298.15})["T_post_fill_K"])
    tau = m.thermal_equilibration_time()
    t_arr = np.linspace(0, 5 * tau, 200)
    T_cool = m.tank_temperature_cooling(T_post, 298.15, t_arr)

    # Scenario 5: Usable mass vs T_amb
    T_amb_range = np.linspace(233.15, 333.15, 50)
    usable = m.usable_mass_vs_Tamb(T_amb_range)

    # Report
    print("=== EC012 F1b Real-Gas H2 Storage — Simulation Report ===\n")
    print("Z at 298.15 K:")
    for P in [1, 50, 100, 200, 350, 700, 900]:
        r = model.predict({"P_bar": float(P), "T_K": 298.15})
        print(f"  P={P:>4d} bar  Z={float(r['Z']):.4f}  m={float(r['stored_mass_kg']):.3f} kg")

    print(f"\nCompression 30→700 bar (298 K): {comp_works[-11]:.1f} kJ/kg")
    print(f"Fill 20→700 bar transient: dT={dT_arr[2]:.1f} K, tau={tau:.0f} s")
    print(f"T_post_fill: {T_post:.1f} K,  T_amb: 298.15 K")
    print(f"Usable mass at -20°C: {float(m.usable_mass_vs_Tamb(253.15)):.2f} kg, "
          f"at +40°C: {float(m.usable_mass_vs_Tamb(313.15)):.2f} kg")

    if not HAS_PLOTLY:
        print("\nPlotly not available — skipping HTML report.")
        return

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Compressibility Z(P) at Various Temperatures",
            "Stored H2 Mass vs Pressure",
            "Compression Work vs Final Pressure",
            "Post-Fill Temperature Transient (Cooling)",
            "Temperature Rise During Fill vs T_amb",
            "Usable H2 Mass vs Ambient Temperature",
        ]
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, T in enumerate(temps_K):
        label = f"T={T - 273.15:.0f}°C"
        fig.add_trace(go.Scatter(x=P_arr, y=Z_curves[T], name=label,
                                  line=dict(color=colors[i])), row=1, col=1)
        fig.add_trace(go.Scatter(x=P_arr, y=m_curves[T], name=label,
                                  line=dict(color=colors[i]), showlegend=False), row=1, col=2)

    fig.add_trace(go.Scatter(x=P2_arr, y=comp_works, name="w_comp",
                              line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_arr / 60, y=T_cool - 273.15, name="T_tank",
                              line=dict(color="red")), row=2, col=2)
    fig.add_hline(y=298.15 - 273.15, line_dash="dash", line_color="gray", row=2, col=2)

    T_amb_C = T_amb_arr - 273.15
    fig.add_trace(go.Bar(x=T_amb_C, y=dT_arr, name="ΔT fill",
                          marker_color="crimson"), row=3, col=1)
    fig.add_trace(go.Scatter(x=T_amb_range - 273.15, y=usable, name="usable mass",
                              line=dict(color="steelblue")), row=3, col=2)

    fig.update_xaxes(title_text="Pressure [bar]", row=1, col=1)
    fig.update_xaxes(title_text="Pressure [bar]", row=1, col=2)
    fig.update_xaxes(title_text="Final Pressure [bar]", row=2, col=1)
    fig.update_xaxes(title_text="Time [min]", row=2, col=2)
    fig.update_xaxes(title_text="T_amb [°C]", row=3, col=1)
    fig.update_xaxes(title_text="T_amb [°C]", row=3, col=2)
    fig.update_yaxes(title_text="Z [-]", row=1, col=1)
    fig.update_yaxes(title_text="Mass [kg]", row=1, col=2)
    fig.update_yaxes(title_text="w_comp [kJ/kg]", row=2, col=1)
    fig.update_yaxes(title_text="T_tank [°C]", row=2, col=2)
    fig.update_yaxes(title_text="ΔT [K]", row=3, col=1)
    fig.update_yaxes(title_text="Usable mass [kg]", row=3, col=2)

    fig.update_layout(
        height=900, width=1200,
        title_text="EC012 F1b — Real-Gas Compressed H2 Storage Model",
        showlegend=True,
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    run_scenarios()
