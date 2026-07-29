"""EC079 — Molten Salt TES — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import MoltenSaltTESF1a
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    tes = MoltenSaltTESF1a(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Energy Stored vs Temperature",
            "SOC vs Temperature",
            "Heat Loss vs Temperature",
            "Diurnal Charge-Discharge Simulation",
            "dT/dt vs Charge Power (at T=427°C)",
            "24-Hour SOC Profile",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    T_range = np.linspace(290, 565, 200)

    # 1) Energy stored vs Temperature
    r_T = model.predict({"temperature": T_range, "q_charge": 0.0, "q_discharge": 0.0})
    fig.add_trace(
        go.Scatter(x=T_range, y=r_T["energy_stored_mwh"],
                   name="Energy stored", line=dict(color="firebrick", width=2)),
        row=1, col=1,
    )
    # Mark capacity
    fig.add_hline(y=tes.E_capacity_MWh, line_dash="dash", line_color="orange",
                  annotation_text=f"Capacity {tes.E_capacity_MWh:.0f} MWh",
                  annotation_position="bottom right", row=1, col=1)

    # 2) SOC vs Temperature
    fig.add_trace(
        go.Scatter(x=T_range, y=r_T["soc"],
                   name="SOC", line=dict(color="steelblue", width=2)),
        row=1, col=2,
    )

    # 3) Heat loss vs Temperature for different ambient temps
    for T_amb in [0, 15, 25, 40]:
        r_loss = model.predict({"temperature": T_range, "q_charge": 0.0, "q_discharge": 0.0,
                                "t_ambient": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=T_range, y=r_loss["heat_loss_kw"],
                       name=f"T_amb={T_amb}°C"),
            row=1, col=3,
        )

    # 4) Diurnal simulation: charge 8h, idle 4h, discharge 8h, idle 4h
    N = 24
    q_c = np.zeros(N); q_d = np.zeros(N)
    q_c[:8] = 30000.0          # 30 MW charge
    q_d[12:20] = 25000.0       # 25 MW discharge
    hours = np.arange(N + 1)
    sim = tes.simulate(290.0, q_c, q_d, dt_s=3600.0)

    fig.add_trace(
        go.Scatter(x=hours, y=sim["T_C"], name="Temperature (°C)",
                   line=dict(color="firebrick")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=np.arange(N) + 0.5, y=q_c / 1000, name="Q_charge MW",
               marker_color="orange", opacity=0.6),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=np.arange(N) + 0.5, y=-q_d / 1000, name="Q_discharge MW",
               marker_color="steelblue", opacity=0.6),
        row=2, col=1,
    )

    # 5) dT/dt vs charge power
    Q_powers = np.linspace(0, 100000, 100)  # kW
    T_mid = (tes.T_hot + tes.T_cold) / 2.0
    r_rate = model.predict({"temperature": T_mid, "q_charge": Q_powers, "q_discharge": 0.0})
    fig.add_trace(
        go.Scatter(x=Q_powers / 1000, y=r_rate["dT_dt"] * 3600,
                   name="dT/dt (K/hr)", line=dict(color="green")),
        row=2, col=2,
    )

    # 6) SOC profile
    fig.add_trace(
        go.Scatter(x=hours, y=sim["soc"] * 100,
                   name="SOC %", line=dict(color="purple", width=2)),
        row=2, col=3,
    )
    fig.add_trace(
        go.Scatter(x=hours, y=sim["energy_mwh"],
                   name="Energy MWh", line=dict(color="darkorange", width=2, dash="dash"),
                   yaxis="y6"),
        row=2, col=3,
    )

    # Axes labels
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=3)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_xaxes(title_text="Charge Power (MW)", row=2, col=2)
    fig.update_xaxes(title_text="Hour", row=2, col=3)
    fig.update_yaxes(title_text="Energy Stored (MWh)", row=1, col=1)
    fig.update_yaxes(title_text="SOC", row=1, col=2)
    fig.update_yaxes(title_text="Heat Loss (kW)", row=1, col=3)
    fig.update_yaxes(title_text="Temperature (°C) / Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="dT/dt (K/hr)", row=2, col=2)
    fig.update_yaxes(title_text="SOC (%)", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} — {info['name']} — {info['fidelity']} Fully Mixed<br>"
            f"<sup>Solar salt 60% NaNO3 + 40% KNO3 | 1000 m³ | {tes.E_capacity_MWh:.0f} MWh | "
            "290-565°C | Herrmann et al. (2004)</sup>"
        ),
        height=850,
        template="plotly_white",
        barmode="relative",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")

    print("\n=== Molten Salt TES Summary ===")
    print(f"  Mass:             {tes.mass/1e6:.2f} Mt = {tes.mass:.0f} kg")
    print(f"  cp:               {tes.cp} J/(kg·K)")
    print(f"  UA_loss:          {tes.UA} W/K")
    print(f"  T_cold / T_hot:   {tes.T_cold} / {tes.T_hot} °C")
    print(f"  E_capacity:       {tes.E_capacity_MWh:.2f} MWh")
    r_mid = model.predict({"temperature": 427.5, "q_charge": 50000.0, "q_discharge": 0.0})
    print(f"\nAt T=427.5°C, Q_charge=50 MW:")
    for k, v in r_mid.items():
        print(f"  {k}: {float(v):.5f}")


if __name__ == "__main__":
    generate_report()
