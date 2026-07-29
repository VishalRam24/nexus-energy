"""EC157 — Buck Converter — F1a — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs Load Current (V_in=48V, V_out=12V)",
            "Loss Breakdown vs Load Current",
            "Efficiency vs Input Voltage (I_load=10A, V_out=12V)",
            "Duty Cycle vs V_out Target (V_in=48V)",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Efficiency vs load current
    i_range = np.linspace(0.1, 15.0, 200)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    fig.add_trace(go.Scatter(
        x=i_range, y=r["efficiency"] * 100,
        name="eta (%)", line=dict(color="#636EFA", width=2.5),
    ), row=1, col=1)
    fig.add_hline(y=95, line_dash="dash", line_color="gray",
                  annotation_text="95%", row=1, col=1)

    # Plot 2: Loss breakdown
    fig.add_trace(go.Scatter(
        x=i_range, y=r["p_conduction_w"],
        name="P_cond", line=dict(color="#EF553B"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=i_range, y=r["p_switching_w"],
        name="P_sw", line=dict(color="#FFA15A"),
        stackgroup="losses",
    ), row=1, col=2)

    # Plot 3: Efficiency vs V_in
    v_in_range = np.linspace(20.0, 80.0, 100)
    r_vin = model.predict({"v_in": v_in_range, "v_out_target": 12.0, "i_load": 10.0})
    fig.add_trace(go.Scatter(
        x=v_in_range, y=r_vin["efficiency"] * 100,
        name="eta vs V_in", line=dict(color="#00CC96", width=2),
    ), row=2, col=1)

    # Plot 4: Duty cycle vs V_out target
    v_out_range = np.linspace(2.0, 45.0, 100)
    r_vout = model.predict({"v_in": 48.0, "v_out_target": v_out_range, "i_load": 5.0})
    fig.add_trace(go.Scatter(
        x=v_out_range, y=r_vout["duty_cycle"],
        name="D = V_out/V_in", line=dict(color="#AB63FA", width=2),
    ), row=2, col=2)
    # Reference line D=0.25 at 12V
    fig.add_vline(x=12.0, line_dash="dot", line_color="red",
                  annotation_text="V_out=12V", row=2, col=2)

    fig.update_xaxes(title_text="Load Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Load Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="Input Voltage (V)", row=2, col=1)
    fig.update_xaxes(title_text="V_out Target (V)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Duty Cycle D", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} (48V→12V, 100kHz)",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary table
    print("\n--- Buck Converter Summary (V_in=48V, V_out=12V) ---")
    print(f"{'I_load(A)':>10} {'D':>6} {'V_out(V)':>9} {'eta(%)':>8} {'P_cond(W)':>10} {'P_sw(W)':>8} {'P_loss(W)':>9}")
    for i in [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]:
        rv = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i})
        print(
            f"{i:>10.1f} {float(rv['duty_cycle']):>6.4f} {float(rv['v_out']):>9.3f} "
            f"{float(rv['efficiency'])*100:>8.3f} {float(rv['p_conduction_w']):>10.4f} "
            f"{float(rv['p_switching_w']):>8.4f} {float(rv['p_loss_w']):>9.4f}"
        )


if __name__ == "__main__":
    generate_report()
