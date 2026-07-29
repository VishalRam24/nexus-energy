"""EC080 — PCM Storage — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import PCMF1a
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    pcm = PCMF1a(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Energy Stored vs Temperature & Fraction",
            "SOC vs Liquid Fraction (at T=Tm=42°C)",
            "Heat Flow Distribution",
            "Charge-Discharge Simulation (500W / 5h each)",
            "Phase Change: Temperature & Fraction vs Time",
            "dT/dt and df/dt vs Charge Power",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    # 1) Energy stored vs temperature (solid and liquid zones)
    T_solid  = np.linspace(0, 42, 100)
    T_liquid = np.linspace(42, 80, 100)
    # Solid: f=0; Liquid: f=1
    r_sol = model.predict({"temperature": T_solid,  "liquid_fraction": np.zeros(100),
                            "q_charge": 0.0, "q_discharge": 0.0})
    r_liq = model.predict({"temperature": T_liquid, "liquid_fraction": np.ones(100),
                            "q_charge": 0.0, "q_discharge": 0.0})
    # Mushy zone: T=42°C, f from 0 to 1
    f_mushy = np.linspace(0, 1, 50)
    r_mus = model.predict({"temperature": 42.0, "liquid_fraction": f_mushy,
                            "q_charge": 0.0, "q_discharge": 0.0})

    fig.add_trace(go.Scatter(x=T_solid,  y=r_sol["energy_stored_kwh"],
                             name="Solid",  line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=[42.0]*50, y=r_mus["energy_stored_kwh"],
                             name="Mushy (T=Tm)",  line=dict(color="orange", width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=T_liquid, y=r_liq["energy_stored_kwh"],
                             name="Liquid", line=dict(color="firebrick")), row=1, col=1)
    fig.add_vline(x=42.0, line_dash="dash", line_color="gray",
                  annotation_text="T_melt=42°C", row=1, col=1)

    # 2) SOC vs liquid fraction at Tm
    fig.add_trace(go.Scatter(x=f_mushy, y=r_mus["soc"],
                             name="SOC vs f", line=dict(color="purple", width=2)), row=1, col=2)

    # 3) Heat flow pie-chart-style bar at a specific charging scenario
    T_test = 42.0; f_test = 0.5; Qc_test = 500.0
    Q_loss = pcm.heat_loss_w(T_test)
    Q_storage = Qc_test - Q_loss
    fig.add_trace(go.Bar(x=["To storage", "Heat loss"],
                         y=[max(Q_storage, 0), Q_loss],
                         name="Power distribution @500W, T=42°C",
                         marker_color=["steelblue", "firebrick"]), row=1, col=3)

    # 4-5) Simulation: charge 5h, idle 1h, discharge 5h
    dt = 60.0  # 1 minute time steps
    N_charge = 300    # 5 hours
    N_idle   = 60     # 1 hour
    N_dis    = 300    # 5 hours
    N = N_charge + N_idle + N_dis

    q_c = np.zeros(N); q_d = np.zeros(N)
    q_c[:N_charge] = 500.0                       # 500 W charge
    q_d[N_charge + N_idle:] = 400.0              # 400 W discharge

    sim = pcm.simulate(15.0, 0.0, q_c, q_d, dt_s=dt)
    time_min = np.arange(N + 1) * dt / 60.0

    # 4) Energy and SOC
    fig.add_trace(go.Scatter(x=time_min, y=sim["energy_kwh"],
                             name="Energy kWh", line=dict(color="darkorange", width=2)), row=2, col=1)
    fig.add_trace(go.Bar(x=np.arange(N) * dt / 60.0, y=q_c / 1000,
                         name="Q_charge kW", marker_color="orange", opacity=0.4), row=2, col=1)
    fig.add_trace(go.Bar(x=np.arange(N) * dt / 60.0, y=-q_d / 1000,
                         name="Q_discharge kW", marker_color="steelblue", opacity=0.4), row=2, col=1)

    # 5) Temperature and liquid fraction
    fig.add_trace(go.Scatter(x=time_min, y=sim["T_C"],
                             name="Temperature (°C)", line=dict(color="firebrick", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=time_min, y=sim["f"],
                             name="Liquid fraction", line=dict(color="steelblue", width=2, dash="dash")), row=2, col=2)
    fig.add_hline(y=pcm.Tm, line_dash="dot", line_color="orange",
                  annotation_text=f"T_melt={pcm.Tm}°C", row=2, col=2)

    # 6) dT/dt and df/dt vs charge power
    Q_vals = np.linspace(0, 5000, 100)  # W
    r_solid  = model.predict({"temperature": 20.0, "liquid_fraction": 0.0,
                               "q_charge": Q_vals, "q_discharge": 0.0})
    r_mushy2 = model.predict({"temperature": 42.0, "liquid_fraction": 0.5,
                               "q_charge": Q_vals, "q_discharge": 0.0})
    r_liquid = model.predict({"temperature": 60.0, "liquid_fraction": 1.0,
                               "q_charge": Q_vals, "q_discharge": 0.0})

    fig.add_trace(go.Scatter(x=Q_vals, y=r_solid["dT_dt"]  * 3600,
                             name="dT/dt solid (K/hr)",  line=dict(color="steelblue")), row=2, col=3)
    fig.add_trace(go.Scatter(x=Q_vals, y=r_mushy2["d_fraction_dt"] * 3600,
                             name="df/dt mushy (1/hr)",  line=dict(color="orange")), row=2, col=3)
    fig.add_trace(go.Scatter(x=Q_vals, y=r_liquid["dT_dt"] * 3600,
                             name="dT/dt liquid (K/hr)", line=dict(color="firebrick")), row=2, col=3)

    # Labels
    fig.update_xaxes(title_text="Temperature (°C)",     row=1, col=1)
    fig.update_xaxes(title_text="Liquid Fraction [-]",  row=1, col=2)
    fig.update_xaxes(title_text="Component",            row=1, col=3)
    fig.update_xaxes(title_text="Time (min)",           row=2, col=1)
    fig.update_xaxes(title_text="Time (min)",           row=2, col=2)
    fig.update_xaxes(title_text="Charge Power (W)",     row=2, col=3)
    fig.update_yaxes(title_text="Energy Stored (kWh)",   row=1, col=1)
    fig.update_yaxes(title_text="SOC [-]",               row=1, col=2)
    fig.update_yaxes(title_text="Power (W)",             row=1, col=3)
    fig.update_yaxes(title_text="Energy (kWh) / Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="T (°C) / f [-]",        row=2, col=2)
    fig.update_yaxes(title_text="Rate (K/hr or 1/hr)",   row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} — {info['name']} — {info['fidelity']} Latent Heat Model<br>"
            "<sup>Paraffin RT42 | T_melt=42°C | L=174 kJ/kg | 500 kg | "
            "Three-region: solid / mushy / liquid | Mehling & Cabeza (2008)</sup>"
        ),
        height=860,
        template="plotly_white",
        barmode="relative",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")

    print("\n=== PCM Summary (RT42 Paraffin) ===")
    print(f"  T_melt:       {pcm.Tm} °C")
    print(f"  Latent heat:  {pcm.L/1000:.0f} kJ/kg")
    print(f"  Mass:         {pcm.mass} kg")
    print(f"  UA_loss:      {pcm.UA} W/K")
    print(f"  E_full:       {pcm.E_full_J/3.6e6:.2f} kWh")

    print("\n=== Three-region test at 500W charge ===")
    for T, f, label in [(20.0, 0.0, "Solid"), (42.0, 0.5, "Mushy"), (60.0, 1.0, "Liquid")]:
        r = model.predict({"temperature": T, "liquid_fraction": f,
                           "q_charge": 500.0, "q_discharge": 0.0})
        print(f"  {label} T={T}°C f={f}: dT/dt={float(r['dT_dt'])*3600:.4f} K/hr, "
              f"df/dt={float(r['d_fraction_dt'])*3600:.6f} 1/hr, "
              f"E={float(r['energy_stored_kwh']):.2f} kWh, SOC={float(r['soc']):.3f}")


if __name__ == "__main__":
    generate_report()
