"""
EC198 -- Post-Combustion Capture (Amine Scrubbing) -- F2a Equilibrium Stage -- Plotly HTML report.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cm = ComponentModel()
m = cm._model

# ----------------------------------------------------------------
# Figure 1: Capture rate & SRD vs L/G ratio
# ----------------------------------------------------------------
L_Gs = np.linspace(1.2, 5.0, 30)
crs = []
srds = []
for lg in L_Gs:
    r = m.compute(y_CO2_in=0.12, L_G=lg)
    crs.append(r["capture_rate"] * 100)
    srds.append(r["SRD_GJ_per_tCO2"])

fig = make_subplots(rows=2, cols=2,
    subplot_titles=[
        "Capture Rate vs L/G Ratio",
        "Specific Reboiler Duty vs L/G Ratio",
        "Stage Profiles (loading & y_CO2)",
        "Capture Rate vs Number of Stages",
    ])

fig.add_trace(go.Scatter(x=L_Gs, y=crs, name="Capture Rate",
              line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig.update_xaxes(title_text="L/G Ratio (mol/mol)", row=1, col=1)
fig.update_yaxes(title_text="Capture Rate (%)", row=1, col=1)

fig.add_trace(go.Scatter(x=L_Gs, y=srds, name="SRD",
              line=dict(color="#d62728", width=2)), row=1, col=2)
fig.update_xaxes(title_text="L/G Ratio (mol/mol)", row=1, col=2)
fig.update_yaxes(title_text="SRD (GJ/tCO2)", row=1, col=2)

# ----------------------------------------------------------------
# Figure 2: Stage profiles at nominal L/G
# ----------------------------------------------------------------
r_nom = m.compute(y_CO2_in=0.12, L_G=2.5)
stages = list(range(len(r_nom["stage_loadings"])))
fig.add_trace(go.Scatter(x=stages, y=r_nom["stage_loadings"], name="Loading",
              line=dict(color="#2ca02c", width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=stages, y=r_nom["stage_y_CO2"], name="y_CO2",
              line=dict(color="#ff7f0e", width=2, dash="dash")), row=2, col=1)
fig.update_xaxes(title_text="Stage (0=top)", row=2, col=1)
fig.update_yaxes(title_text="Loading / y_CO2", row=2, col=1)

# ----------------------------------------------------------------
# Figure 3: Capture rate vs number of stages
# ----------------------------------------------------------------
N_stages_range = list(range(3, 25))
crs_n = []
for n in N_stages_range:
    r = m.compute(y_CO2_in=0.12, L_G=2.5, N_stages=n)
    crs_n.append(r["capture_rate"] * 100)

fig.add_trace(go.Scatter(x=N_stages_range, y=crs_n, name="CR vs N",
              line=dict(color="#9467bd", width=2)), row=2, col=2)
fig.update_xaxes(title_text="Number of Stages", row=2, col=2)
fig.update_yaxes(title_text="Capture Rate (%)", row=2, col=2)

fig.update_layout(
    title="<b>EC198 Post-Combustion Capture (Amine Scrubbing) -- F2a Equilibrium Stage</b>",
    height=800, template="plotly_white", showlegend=True,
)

out = os.path.join(OUTPUT_DIR, "EC198_F2a_equilibrium_stage_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
