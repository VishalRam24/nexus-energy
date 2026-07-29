"""
EC209 -- Reverse Osmosis (RO) -- F2a Solution-Diffusion -- Simulation report.

Generates an interactive Plotly HTML report with:
  1. Water flux vs feed pressure
  2. Salt rejection vs recovery
  3. SEC vs recovery
  4. Element-by-element profiles for a reference case
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import RO_SolutionDiffusion_F2a

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly not installed. Run: pip install plotly")
    sys.exit(1)

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def load_model():
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    return RO_SolutionDiffusion_F2a(params)


def run():
    m = load_model()

    # ----- Sweep 1: Flux vs Pressure (single vessel, 35 g/L, 8 m3/h) -----
    pressures = np.linspace(30, 80, 30)
    flux_avg = []
    recovery_p = []
    sec_p = []
    rejection_p = []

    for P in pressures:
        r = m.solve_vessel(35.0, P, 8.0)
        # Average flux across elements
        avg_jw = np.mean(r["profiles"]["Jw_LMH"])
        flux_avg.append(avg_jw)
        recovery_p.append(r["recovery"] * 100)
        sec_p.append(r["SEC_kwhm3"])
        rejection_p.append(r["rejection"] * 100)

    # ----- Sweep 2: Effect of feed concentration -----
    concentrations = [10, 20, 35, 42]
    rec_by_conc = {}
    sec_by_conc = {}
    for Cf in concentrations:
        recs = []
        secs = []
        for P in pressures:
            r = m.solve_vessel(Cf, P, 8.0)
            recs.append(r["recovery"] * 100)
            secs.append(r["SEC_kwhm3"])
        rec_by_conc[Cf] = recs
        sec_by_conc[Cf] = secs

    # ----- Sweep 3: Element-by-element profiles (reference case) -----
    ref = m.solve_vessel(35.0, 60.0, 8.0)
    elem_idx = list(range(1, ref["N_elements_active"] + 1))

    # ----- Build Plotly Report -----
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Avg Water Flux vs Feed Pressure",
            "Recovery vs Feed Pressure",
            "SEC vs Feed Pressure (by salinity)",
            "Rejection vs Recovery",
            "Element Profiles: Flux & Concentration",
            "Element Profiles: Pressure & Recovery",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
    )

    # (1,1) Flux vs Pressure
    fig.add_trace(go.Scatter(x=pressures, y=flux_avg, mode="lines+markers",
                             name="Avg Jw (35 g/L)", marker=dict(size=5)),
                  row=1, col=1)
    fig.update_xaxes(title_text="Feed Pressure [bar]", row=1, col=1)
    fig.update_yaxes(title_text="Avg Water Flux [LMH]", row=1, col=1)

    # (1,2) Recovery vs Pressure by salinity
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, Cf in enumerate(concentrations):
        fig.add_trace(go.Scatter(x=pressures, y=rec_by_conc[Cf], mode="lines",
                                 name=f"{Cf} g/L", line=dict(color=colors[i])),
                      row=1, col=2)
    fig.update_xaxes(title_text="Feed Pressure [bar]", row=1, col=2)
    fig.update_yaxes(title_text="Recovery [%]", row=1, col=2)

    # (2,1) SEC vs Pressure by salinity
    for i, Cf in enumerate(concentrations):
        fig.add_trace(go.Scatter(x=pressures, y=sec_by_conc[Cf], mode="lines",
                                 name=f"SEC {Cf} g/L", line=dict(color=colors[i], dash="dash")),
                      row=2, col=1)
    fig.update_xaxes(title_text="Feed Pressure [bar]", row=2, col=1)
    fig.update_yaxes(title_text="SEC [kWh/m3]", row=2, col=1)

    # (2,2) Rejection vs Recovery
    fig.add_trace(go.Scatter(x=recovery_p, y=rejection_p, mode="lines+markers",
                             name="Rejection (35 g/L)", marker=dict(size=5, color="#9467bd")),
                  row=2, col=2)
    fig.update_xaxes(title_text="Recovery [%]", row=2, col=2)
    fig.update_yaxes(title_text="Salt Rejection [%]", row=2, col=2)

    # (3,1) Element profiles: flux and concentration
    fig.add_trace(go.Bar(x=elem_idx, y=ref["profiles"]["Jw_LMH"],
                         name="Jw [LMH]", marker_color="#1f77b4"),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=elem_idx, y=ref["profiles"]["Cf_gL"], mode="lines+markers",
                             name="Cf [g/L]", yaxis="y7", marker=dict(color="#ff7f0e")),
                  row=3, col=1)
    fig.update_xaxes(title_text="Element #", row=3, col=1)
    fig.update_yaxes(title_text="Water Flux [LMH] / Cf [g/L]", row=3, col=1)

    # (3,2) Element profiles: pressure and cumulative recovery
    fig.add_trace(go.Scatter(x=elem_idx, y=ref["profiles"]["P_bar"], mode="lines+markers",
                             name="Pressure [bar]", marker=dict(color="#2ca02c")),
                  row=3, col=2)
    cum_Qp = np.cumsum(ref["profiles"]["Qp_m3h"])
    cum_rec = cum_Qp / 8.0 * 100
    fig.add_trace(go.Scatter(x=elem_idx, y=cum_rec.tolist(), mode="lines+markers",
                             name="Cum. Recovery [%]", marker=dict(color="#d62728")),
                  row=3, col=2)
    fig.update_xaxes(title_text="Element #", row=3, col=2)
    fig.update_yaxes(title_text="Pressure [bar] / Recovery [%]", row=3, col=2)

    # Layout
    fig.update_layout(
        title_text="EC209 RO -- F2a Solution-Diffusion Model -- Simulation Report",
        height=1100, width=1100,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
        template="plotly_white",
    )

    # Add annotation with reference case summary
    ref_text = (
        f"Reference case: 35 g/L NaCl, 60 bar, 8 m3/h, 25 degC, 7 elements<br>"
        f"Recovery={ref['recovery']:.1%}, Rejection={ref['rejection']:.4f}, "
        f"SEC={ref['SEC_kwhm3']:.2f} kWh/m3, Cp={ref['Cp_gL']:.3f} g/L"
    )
    fig.add_annotation(
        text=ref_text, xref="paper", yref="paper",
        x=0.5, y=1.06, showarrow=False, font=dict(size=11),
    )

    fig.write_html(OUT_PATH, include_plotlyjs="cdn")
    print(f"Report saved to {OUT_PATH}")


if __name__ == "__main__":
    run()
