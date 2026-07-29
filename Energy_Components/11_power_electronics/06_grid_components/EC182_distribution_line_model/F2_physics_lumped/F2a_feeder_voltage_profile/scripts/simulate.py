"""
EC182 -- Distribution Line Model -- F2a Feeder Voltage-Profile ODE
Optional Plotly report: voltage profile, current profile, and ANSI band along
a radial feeder under distributed load. Plotly import guarded so absence is safe.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html="simulation_report.html"):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    cm = ComponentModel()
    cases = [
        ("Light load (300 kW, 5 km)", {"V_s_kV": 11.0, "P_total_kW": 300.0,
                                       "Q_total_kVAR": 120.0, "length_km": 5.0}),
        ("Medium load (1500 kW, 8 km)", {"V_s_kV": 11.0, "P_total_kW": 1500.0,
                                         "Q_total_kVAR": 600.0, "length_km": 8.0}),
        ("Heavy/long (3500 kW, 15 km)", {"V_s_kV": 11.0, "P_total_kW": 3500.0,
                                         "Q_total_kVAR": 1400.0, "length_km": 15.0}),
    ]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Voltage profile V(x)",
                                        "Current profile I(x)"))
    band = cm._model.ansi_band_pct / 100.0
    for label, inp in cases:
        r = cm.predict(inp)
        fig.add_trace(go.Scatter(x=r["x_km"], y=r["V_profile_kV"],
                                 mode="lines", name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["x_km"], y=r["I_profile_A"],
                                 mode="lines", name=label, showlegend=False),
                      row=1, col=2)
    # ANSI lower band line at 95% of 11 kV
    fig.add_hline(y=11.0 * (1 - band), line_dash="dash", line_color="red",
                  annotation_text="ANSI -5% band", row=1, col=1)

    fig.update_xaxes(title_text="Distance from substation [km]", row=1, col=1)
    fig.update_xaxes(title_text="Distance from substation [km]", row=1, col=2)
    fig.update_yaxes(title_text="Line-to-line voltage [kV]", row=1, col=1)
    fig.update_yaxes(title_text="Phase current [A]", row=1, col=2)
    fig.update_layout(title="EC182 F2a -- Distribution Feeder Voltage/Current Profiles "
                            f"(R/X={cm._model.r_over_x:.2f}, distributed load)")

    out_path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(out_path)
    print(f"[simulate] Report written to {out_path}")
    return out_path


if __name__ == "__main__":
    build_report()
