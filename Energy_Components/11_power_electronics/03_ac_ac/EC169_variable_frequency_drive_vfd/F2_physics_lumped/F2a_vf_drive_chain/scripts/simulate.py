"""
EC169 -- Variable Frequency Drive (VFD) -- F2a Physics-Lumped V/f Drive Chain
Plotly HTML simulation report generator (optional; safe if plotly absent).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: soft-start frequency ramp ---
def ramp(t):
    return min(50.0, 5.0 + 22.5 * t)  # 5 -> 50 Hz over 2 s


r1 = model.predict({"f_set": ramp, "T_load": 50.0, "dt": 0.005, "duration_s": 5.0})

# --- Scenario 2: V/f profile vs frequency (static) ---
f_sweep = np.linspace(0.0, 120.0, 200)
V_sweep = m.output_voltage(f_sweep)
vf_sweep = m.vf_ratio(f_sweep)

# --- Scenario 3: torque-speed family at several frequencies ---
ts_curves = {}
for f in [25.0, 50.0, 75.0]:
    omega_s = float(m.sync_speed_mech(f))
    omegas = np.linspace(0.0, omega_s * 0.999, 300)
    ts_curves[f] = (omegas * 60.0 / (2 * np.pi), m.motor_torque(omegas, f))


def main():
    if not HAVE_PLOTLY:
        print("plotly not installed; skipping HTML report.")
        print(f"Ramp final speed: {r1['speed_rpm'][-1]:.1f} rpm, "
              f"eta={r1['efficiency'][-1]:.3f}")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Soft-start: speed & frequency vs time",
            "V/f control profile",
            "Torque-speed family (V/f)",
            "DC-link voltage & efficiency",
        ),
    )

    fig.add_trace(go.Scatter(x=r1["t"], y=r1["speed_rpm"], name="speed [rpm]"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["f_out"] * 30.0,
                             name="f_out x30 [Hz]"), row=1, col=1)

    fig.add_trace(go.Scatter(x=f_sweep, y=V_sweep, name="V_out [V]"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=f_sweep, y=vf_sweep, name="V/f [V/Hz]"),
                  row=1, col=2)

    for f, (rpm, T) in ts_curves.items():
        fig.add_trace(go.Scatter(x=rpm, y=T, name=f"{f:.0f} Hz"), row=2, col=1)

    fig.add_trace(go.Scatter(x=r1["t"], y=r1["V_dc"], name="V_dc [V]"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["efficiency"] * 600.0,
                             name="eta x600"), row=2, col=2)

    fig.update_layout(title="EC169 VFD F2a — V/f Drive Chain", height=800)
    out = os.path.join(OUTPUT_DIR, "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
