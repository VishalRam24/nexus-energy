"""EC181 — Transmission Line — F1a Pi-Model — Simulation & HTML Report"""
import sys, numpy as np
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
            "Voltage Drop vs Load (200 km, V_s=1.0 pu, pf=0.9 lag)",
            "P_loss vs Load Power",
            "Efficiency vs Line Length (P=0.6 pu, Q=0.29 pu)",
            "Receiving Voltage vs Line Length",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Voltage drop vs P_load
    P_range = np.linspace(0.0, 1.2, 100)
    Q_range = P_range * np.tan(np.arccos(0.9))  # pf=0.9 lagging
    r1 = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                        "P_load_pu": P_range, "Q_load_pu": Q_range,
                        "length_km": 200.0})
    fig.add_trace(go.Scatter(x=P_range, y=r1["voltage_drop_pu"],
                             name="V_drop (pu)", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)

    # Plot 2: P_loss vs P_load
    fig.add_trace(go.Scatter(x=P_range, y=r1["P_loss_pu"],
                             name="P_loss (pu)", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)

    # Plot 3: Efficiency vs line length
    lengths = np.linspace(50, 800, 100)
    eta_list = []
    for L in lengths:
        r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                           "P_load_pu": 0.6, "Q_load_pu": 0.29, "length_km": float(L)})
        eta_list.append(float(r["efficiency"]))
    fig.add_trace(go.Scatter(x=lengths, y=np.array(eta_list) * 100,
                             name="eta (%)", line=dict(color="#00CC96", width=2)),
                  row=2, col=1)

    # Plot 4: V_r vs line length
    Vr_list = []
    for L in lengths:
        r = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                           "P_load_pu": 0.6, "Q_load_pu": 0.29, "length_km": float(L)})
        Vr_list.append(float(r["V_r_pu"]))
    fig.add_trace(go.Scatter(x=lengths, y=Vr_list,
                             name="V_r (pu)", line=dict(color="#AB63FA", width=2)),
                  row=2, col=2)
    fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                  annotation_text="0.95 pu limit", row=2, col=2)

    fig.update_xaxes(title_text="Active Load P (pu)", row=1, col=1)
    fig.update_xaxes(title_text="Active Load P (pu)", row=1, col=2)
    fig.update_xaxes(title_text="Line Length (km)", row=2, col=1)
    fig.update_xaxes(title_text="Line Length (km)", row=2, col=2)
    fig.update_yaxes(title_text="Voltage Drop (pu)", row=1, col=1)
    fig.update_yaxes(title_text="Active Power Loss (pu)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="V_r (pu)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary table
    print("\n--- Transmission Line Summary (200 km, pf=0.9 lag) ---")
    print(f"{'P_load(pu)':>12} {'V_r(pu)':>9} {'dV(pu)':>8} {'P_loss(pu)':>11} {'eta(%)':>8}")
    for P in [0.1, 0.3, 0.5, 0.7, 1.0, 1.2]:
        Q = P * np.tan(np.arccos(0.9))
        rv = model.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                            "P_load_pu": P, "Q_load_pu": Q, "length_km": 200.0})
        print(f"{P:>12.2f} {float(rv['V_r_pu']):>9.4f} {float(rv['voltage_drop_pu']):>8.4f} "
              f"{float(rv['P_loss_pu']):>11.5f} {float(rv['efficiency'])*100:>8.2f}")


if __name__ == "__main__":
    generate_report()
