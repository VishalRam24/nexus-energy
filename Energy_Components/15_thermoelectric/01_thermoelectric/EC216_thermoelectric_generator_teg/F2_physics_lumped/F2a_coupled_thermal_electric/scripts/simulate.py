"""
EC216 -- Thermoelectric Generator (TEG) -- F2a Coupled Thermal-Electrical -- Simulation & HTML report.

Generates interactive Plotly plots:
  1. Power vs delta-T  (at matched load)
  2. Efficiency vs delta-T
  3. I-V curves at different delta-T
  4. Power vs R_load at different delta-T
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import TEG_CoupledF2a

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("WARNING: plotly not installed. Install with: pip install plotly")

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def load_model():
    with open(_PARAMS_PATH) as f:
        params = json.load(f)
    return TEG_CoupledF2a(params)


def run_simulation():
    m = load_model()
    T_cold = 300.0  # K

    # ---- 1 & 2: Power and Efficiency vs delta-T (matched load) ----
    dTs = np.linspace(5, 280, 60)
    sweep = m.sweep_delta_T(T_cold, dTs)

    # ---- 3: I-V curves at different delta-T ----
    dT_values = [50, 100, 150, 200]
    iv_data = {}
    for dT in dT_values:
        T_hot = T_cold + dT
        iv_data[dT] = m.iv_curve(T_hot, T_cold, N_points=80)

    # ---- 4: Power vs R_load at different delta-T ----
    R_loads = np.geomspace(0.05, 50, 80)
    p_vs_r = {}
    for dT in dT_values:
        T_hot = T_cold + dT
        powers = []
        for R_L in R_loads:
            res = m.solve_steady_state(T_hot, T_cold, R_L)
            powers.append(res["P"])
        p_vs_r[dT] = np.array(powers)

    return {
        "dTs": dTs, "sweep": sweep,
        "dT_values": dT_values, "iv_data": iv_data,
        "R_loads": R_loads, "p_vs_r": p_vs_r,
        "T_cold": T_cold,
    }


def build_report(data):
    if not HAS_PLOTLY:
        print("Cannot generate report without plotly.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power vs Temperature Difference (Matched Load)",
            "Efficiency vs Temperature Difference (Matched Load)",
            "I-V Curves at Various \u0394T",
            "Power vs Load Resistance",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    # 1. Power vs dT
    fig.add_trace(
        go.Scatter(x=data["dTs"], y=data["sweep"]["P"],
                   mode="lines", name="P (matched)",
                   line=dict(color="crimson", width=2)),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="\u0394T [K]", row=1, col=1)
    fig.update_yaxes(title_text="Power [W]", row=1, col=1)

    # 2. Efficiency vs dT
    fig.add_trace(
        go.Scatter(x=data["dTs"], y=np.array(data["sweep"]["efficiency"]) * 100,
                   mode="lines", name="\u03b7 (model)",
                   line=dict(color="royalblue", width=2)),
        row=1, col=2,
    )
    # Carnot limit
    eta_carnot = (1.0 - data["T_cold"] / (data["T_cold"] + data["dTs"])) * 100
    fig.add_trace(
        go.Scatter(x=data["dTs"], y=eta_carnot,
                   mode="lines", name="\u03b7 Carnot",
                   line=dict(color="gray", width=1, dash="dash")),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="\u0394T [K]", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency [%]", row=1, col=2)

    # 3. I-V curves
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
    for i, dT in enumerate(data["dT_values"]):
        iv = data["iv_data"][dT]
        fig.add_trace(
            go.Scatter(x=iv["V"], y=iv["I"],
                       mode="lines", name=f"\u0394T={dT}K",
                       line=dict(color=colors[i % len(colors)], width=2)),
            row=2, col=1,
        )
    fig.update_xaxes(title_text="Voltage [V]", row=2, col=1)
    fig.update_yaxes(title_text="Current [A]", row=2, col=1)

    # 4. Power vs R_load
    for i, dT in enumerate(data["dT_values"]):
        fig.add_trace(
            go.Scatter(x=data["R_loads"], y=data["p_vs_r"][dT],
                       mode="lines", name=f"\u0394T={dT}K ",
                       line=dict(color=colors[i % len(colors)], width=2)),
            row=2, col=2,
        )
    fig.update_xaxes(title_text="R_load [\u03a9]", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Power [W]", row=2, col=2)

    fig.update_layout(
        title_text="EC216 TEG -- F2a Coupled Thermal-Electrical Model",
        height=850,
        width=1200,
        showlegend=True,
        template="plotly_white",
    )

    fig.write_html(_REPORT_PATH, include_plotlyjs="cdn")
    print(f"Report saved to {_REPORT_PATH}")


if __name__ == "__main__":
    print("Running EC216 TEG F2a simulation...")
    data = run_simulation()

    # Print summary
    print(f"\n--- Summary (T_cold={data['T_cold']}K, matched load) ---")
    for i in [0, len(data["dTs"]) // 4, len(data["dTs"]) // 2, -1]:
        dT = data["dTs"][i]
        P = data["sweep"]["P"][i]
        eta = data["sweep"]["efficiency"][i]
        print(f"  dT={dT:6.1f} K  |  P={P:7.3f} W  |  eta={eta:.4f}")

    build_report(data)
