"""
EC087 -- Biomass Boiler -- F2a
Optional Plotly simulation report. Plotly import is guarded so absence
does not crash. Produces simulation_report.html in the model folder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # Scenario: cold start at full load, then a load step down to 40 %.
    def plr(t):
        return 1.0 if t < 1800 else 0.4

    r = cm.predict({"PLR": plr, "T_water_init_K": 293.15,
                    "T_return_K": 323.15, "dt": 5.0, "duration_s": 3600.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"[simulate] Final T_water={r['T_water'][-1]-273.15:.1f} degC, "
              f"eta={r['efficiency'][-1]:.3f}")
        return

    t = r["t"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Block water temperature", "Heat flows",
                        "Efficiency", "Mass flows"),
    )
    fig.add_trace(go.Scatter(x=t, y=r["T_water"] - 273.15, name="T_water"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["T_flue_C"], name="T_flue"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["Q_comb"], name="Q_comb"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["Q_useful"], name="Q_useful"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["Q_stack"], name="Q_stack"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["Q_casing"], name="Q_casing"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["efficiency"], name="eta"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=r["m_fuel"], name="m_fuel"), 2, 2)
    fig.add_trace(go.Scatter(x=t, y=r["m_flue"], name="m_flue"), 2, 2)
    fig.update_layout(title="EC087 Biomass Boiler F2a — combustion + thermal-mass ODE",
                      height=750)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
