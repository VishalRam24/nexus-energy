"""
EC179 -- Wound Rotor Synchronous Generator -- F2a dq-frame
Plotly HTML simulation report: power-angle curve, capability curve,
excitation->Q sweep, and swing-equation transient after a load step.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cm = ComponentModel()
m = cm._model

# --- Scenario 1: Power-angle curves for several excitations ---
deltas = np.radians(np.linspace(0, 90, 91))

# --- Scenario 2: Capability curve ---
cap = m.capability_curve(1.0)

# --- Scenario 3: Excitation -> reactive power sweep at fixed P ---
Efs = np.linspace(0.6, 2.2, 60)

# --- Scenario 4: Swing transient after a load step ---
r = cm.predict({"P_pu": 0.7, "Q_pu": 0.2, "simulate": True,
                "P_step": 0.15, "t_step": 0.5, "duration_s": 6.0})
tr = r["transient"]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Power-angle P=Ef·V/Xs·sinδ", "Capability curve (P-Q)",
                    "Reactive power vs excitation", "Swing transient (load step)"),
)

for Ef in [1.0, 1.5, 2.0]:
    fig.add_trace(go.Scatter(x=np.degrees(deltas),
                             y=m.active_power(Ef, deltas, 1.0),
                             name=f"Ef={Ef} pu"), row=1, col=1)

fig.add_trace(go.Scatter(x=cap["Q_armature"], y=cap["P_armature"],
                         name="armature limit"), row=1, col=2)
fig.add_trace(go.Scatter(x=cap["Q_field"], y=cap["P_field"],
                         name="field limit"), row=1, col=2)

Q_sweep = [m.reactive_power(Ef, np.radians(20), 1.0) for Ef in Efs]
fig.add_trace(go.Scatter(x=Efs, y=Q_sweep, name="Q(Ef)"), row=2, col=1)
fig.add_hline(y=0, line_dash="dash", row=2, col=1)

fig.add_trace(go.Scatter(x=tr["t"], y=tr["delta_deg"], name="δ [deg]"), row=2, col=2)

fig.update_xaxes(title_text="δ [deg]", row=1, col=1)
fig.update_yaxes(title_text="P [pu]", row=1, col=1)
fig.update_xaxes(title_text="Q [pu]", row=1, col=2)
fig.update_yaxes(title_text="P [pu]", row=1, col=2)
fig.update_xaxes(title_text="Ef [pu]", row=2, col=1)
fig.update_yaxes(title_text="Q [pu]", row=2, col=1)
fig.update_xaxes(title_text="t [s]", row=2, col=2)
fig.update_yaxes(title_text="δ [deg]", row=2, col=2)
fig.update_layout(title_text="EC179 WRSG F2a — dq-frame synchronous generator", height=800)

out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written to {out}")
