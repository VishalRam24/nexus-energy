"""
EC208 -- CO2 Geological Sequestration -- F2a
Simulation + interactive Plotly report (optional; plotly import is guarded).

Run:  python3 scripts/simulate.py
Produces simulation_report.html in the model folder if plotly is available.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

Mt = 1e9  # kg per Mt CO2


def run():
    cm = ComponentModel()
    r = cm.predict({"P_wellhead_bar": 90.0, "injection_years": 30.0,
                    "sim_years": 1000.0, "n_points": 400})

    print("EC208 CO2 Geological Sequestration -- F2a reservoir/trapping simulation")
    print(f"  Injected total : {r['injected_cumulative_t'][-1]/Mt:7.3f} Mt")
    print(f"  Stored total   : {r['M_total_t'][-1]/Mt:7.3f} Mt")
    print(f"  Peak P_res     : {r['reservoir_pressure_bar'].max():7.1f} bar "
          f"(fracture {r['fracture_pressure_bar']:.1f} bar)")
    print(f"  Max plume rad  : {r['plume_radius_m'].max():7.1f} m")
    tf = r["trapping_fraction"]
    print("  Trapping @ 1000 yr:")
    for k in ("structural", "residual", "solubility", "mineral"):
        print(f"     {k:11s}: {tf[k][-1]*100:6.2f} %")

    _maybe_plot(r)
    return r


def _maybe_plot(r):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> skip silently
        print(f"  [plotly unavailable, skipping HTML report: {e}]")
        return

    t = r["t_years"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("CO2 mass by trapping mechanism (Mt)",
                        "Reservoir vs fracture pressure (bar)",
                        "Trapping fraction (security pyramid)",
                        "Plume radius (m) & injection rate (kg/s)"),
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
    )

    for name, key, color in [
        ("Structural (mobile)", "M_mobile_t", "#d62728"),
        ("Residual", "M_residual_t", "#ff7f0e"),
        ("Solubility", "M_dissolved_t", "#1f77b4"),
        ("Mineral", "M_mineral_t", "#2ca02c"),
    ]:
        fig.add_trace(go.Scatter(x=t, y=r[key] / Mt, name=name,
                                 stackgroup="m", line_color=color), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=r["reservoir_pressure_bar"],
                             name="P_res", line_color="#1f77b4"), row=1, col=2)
    fig.add_hline(y=r["fracture_pressure_bar"], line_dash="dash",
                  line_color="red", annotation_text="fracture P", row=1, col=2)

    tf = r["trapping_fraction"]
    for name, color in [("structural", "#d62728"), ("residual", "#ff7f0e"),
                        ("solubility", "#1f77b4"), ("mineral", "#2ca02c")]:
        fig.add_trace(go.Scatter(x=t, y=tf[name], name=f"f_{name}",
                                 stackgroup="f", line_color=color,
                                 showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=r["plume_radius_m"], name="plume radius",
                             line_color="#9467bd"), row=2, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=r["injection_rate_kg_s"], name="inj rate",
                             line_color="#8c564b"), row=2, col=2, secondary_y=True)

    fig.update_xaxes(title_text="time (years)")
    fig.update_layout(title="EC208 CO2 Geological Sequestration -- F2a Physics-Lumped",
                      height=800, template="plotly_white")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"  Wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
