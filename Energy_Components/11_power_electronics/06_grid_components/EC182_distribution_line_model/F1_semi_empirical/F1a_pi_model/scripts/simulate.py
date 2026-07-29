"""EC182 — Distribution Line — F1a R+jX Model — Simulation & HTML Report"""
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
            "Voltage at Receiving End vs Load (5 km, 11 kV)",
            "Power Loss vs Load",
            "Voltage Drop % vs Feeder Length (P=1500 kW, pf=0.9)",
            "Efficiency vs Load Power Factor",
        ],
        vertical_spacing=0.14,
    )

    P_range = np.linspace(0, 3000, 100)
    Q_range = P_range * np.tan(np.arccos(0.85))

    r1 = model.predict({"V_s_kV": 11.0, "P_load_kW": P_range, "Q_load_kVAR": Q_range})
    fig.add_trace(go.Scatter(x=P_range / 1000, y=r1["V_r_kV"],
                             name="V_r (kV)", line=dict(color="#636EFA", width=2.5)),
                  row=1, col=1)
    fig.add_hline(y=11.0 * 0.95, line_dash="dash", line_color="red",
                  annotation_text="-5%", row=1, col=1)

    fig.add_trace(go.Scatter(x=P_range / 1000, y=r1["P_loss_kW"],
                             name="P_loss (kW)", line=dict(color="#EF553B", width=2.5)),
                  row=1, col=2)

    lengths = np.linspace(0.5, 20, 80)
    dV_list = []
    for L in lengths:
        r = model.predict({"V_s_kV": 11.0, "P_load_kW": 1500.0,
                           "Q_load_kVAR": 1500.0 * np.tan(np.arccos(0.9)),
                           "length_km": float(L)})
        dV_list.append(float(r["voltage_drop_pct"]))
    fig.add_trace(go.Scatter(x=lengths, y=dV_list,
                             name="dV%", line=dict(color="#00CC96", width=2)),
                  row=2, col=1)
    fig.add_hline(y=5.0, line_dash="dash", line_color="red",
                  annotation_text="5% limit", row=2, col=1)

    pf_range = np.linspace(0.7, 1.0, 80)
    Q_pf = 1500.0 * np.tan(np.arccos(pf_range))
    r_pf = model.predict({"V_s_kV": 11.0, "P_load_kW": 1500.0, "Q_load_kVAR": Q_pf})
    fig.add_trace(go.Scatter(x=pf_range, y=r_pf["efficiency"] * 100,
                             name="eta (%)", line=dict(color="#AB63FA", width=2)),
                  row=2, col=2)

    fig.update_xaxes(title_text="Load P (MW)", row=1, col=1)
    fig.update_xaxes(title_text="Load P (MW)", row=1, col=2)
    fig.update_xaxes(title_text="Feeder Length (km)", row=2, col=1)
    fig.update_xaxes(title_text="Power Factor", row=2, col=2)
    fig.update_yaxes(title_text="V_r (kV)", row=1, col=1)
    fig.update_yaxes(title_text="P_loss (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage Drop (%)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Distribution Line Summary (11 kV, 5 km, pf=0.85 lag) ---")
    print(f"{'P_load(kW)':>12} {'V_r(kV)':>9} {'dV(%)':>7} {'P_loss(kW)':>11} {'eta(%)':>8}")
    for P in [200, 500, 1000, 1500, 2000, 3000]:
        Q = P * np.tan(np.arccos(0.85))
        rv = model.predict({"V_s_kV": 11.0, "P_load_kW": P, "Q_load_kVAR": Q})
        print(f"{P:>12} {float(rv['V_r_kV']):>9.3f} {float(rv['voltage_drop_pct']):>7.2f} "
              f"{float(rv['P_loss_kW']):>11.2f} {float(rv['efficiency'])*100:>8.2f}")


if __name__ == "__main__":
    generate_report()
