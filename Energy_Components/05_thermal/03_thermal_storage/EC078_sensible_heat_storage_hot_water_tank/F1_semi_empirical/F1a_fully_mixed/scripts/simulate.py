"""EC078 — Hot Water Tank TES — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info  = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Energy Stored vs Temperature",
            "Heat Loss vs Temperature (various T_amb)",
            "dT/dt vs Q_charge at T=60C",
            "Daily Charge/Discharge Simulation",
        ],
        vertical_spacing=0.13,
    )

    T_range = np.linspace(30.0, 90.0, 200)

    # Row 1 Col 1 — energy stored vs temperature
    r = model.predict({"temperature": T_range})
    fig.add_trace(
        go.Scatter(x=T_range, y=r["energy_stored_kwh"], name="Energy Stored",
                   line=dict(color="steelblue")),
        row=1, col=1,
    )

    # Row 1 Col 2 — heat loss vs temperature at various T_amb
    for T_amb in [5, 15, 20, 30]:
        r = model.predict({"temperature": T_range, "t_ambient": float(T_amb)})
        fig.add_trace(
            go.Scatter(x=T_range, y=r["heat_loss_w"], name=f"T_amb={T_amb}C"),
            row=1, col=2,
        )

    # Row 2 Col 1 — dT/dt vs Q_charge at T=60C, Q_discharge=0
    q_range = np.linspace(0, 20000, 200)
    r = model.predict({"temperature": 60.0, "q_charge": q_range, "q_discharge": 0.0, "t_ambient": 20.0})
    fig.add_trace(
        go.Scatter(x=q_range / 1000.0, y=r["dT_dt"] * 1000.0,
                   name="dT/dt at T=60C", line=dict(color="darkorange"), showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    # Row 2 Col 2 — 24h charge/discharge simulation
    # Morning charge (0–8h), discharge (8–16h), standby (16–24h)
    dt = 60.0         # seconds
    T_sim  = [30.0]   # start cold
    t_arr  = []
    soc_arr = []
    T_amb_sim = 20.0

    total_steps = int(24 * 3600 / dt)
    for i in range(total_steps):
        t_h = i * dt / 3600.0
        if t_h < 8:          # charging period
            qin, qout = 8000.0, 0.0
        elif t_h < 16:       # discharge period
            qin, qout = 0.0, 6000.0
        else:                # standby
            qin, qout = 0.0, 0.0

        s = T_sim[-1]
        r = model.predict({"temperature": s, "q_charge": qin, "q_discharge": qout,
                           "t_ambient": T_amb_sim})
        dT = float(r["dT_dt"]) * dt
        T_new = np.clip(s + dT, 30.0, 90.0)
        T_sim.append(T_new)
        t_arr.append(t_h)
        soc_arr.append(float(r["soc"]))

    fig.add_trace(
        go.Scatter(x=t_arr, y=T_sim[:-1], name="Tank Temperature",
                   line=dict(color="firebrick"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Q_charge (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)

    fig.update_yaxes(title_text="Energy Stored (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Heat Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="dT/dt (mK/s)", row=2, col=1)
    fig.update_yaxes(title_text="Tank Temperature (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 500 L",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
