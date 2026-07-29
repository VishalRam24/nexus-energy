"""EC080 -- PCM Storage -- F1b Enthalpy Method -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import PCMStorageF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    pcm = PCMStorageF1b(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Enthalpy vs Temperature",
            "Phase Fraction vs Temperature",
            "Charging from Cold (T_pcm vs time)",
            "Energy Stored During Charge",
            "Charge/Discharge Cycle",
            "Thermal Power During Cycle",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # --- 1) Enthalpy curve ---
    T_range = np.linspace(30, 80, 500)
    h_range = pcm.enthalpy(T_range)
    fig.add_trace(
        go.Scatter(x=T_range, y=h_range / 1000, name="h(T)",
                   line=dict(color="firebrick", width=2)),
        row=1, col=1,
    )
    fig.add_vline(x=pcm.T_solidus, line_dash="dash", line_color="blue", row=1, col=1)
    fig.add_vline(x=pcm.T_liquidus, line_dash="dash", line_color="red", row=1, col=1)

    # --- 2) Phase fraction ---
    f_range = pcm.phase_fraction(T_range)
    fig.add_trace(
        go.Scatter(x=T_range, y=f_range, name="Phase fraction",
                   line=dict(color="steelblue", width=2)),
        row=1, col=2,
    )

    # --- 3) Charging simulation (cold start) ---
    N_charge = 24
    modes = ["charge"] * N_charge
    T_htf = np.full(N_charge, 80.0)
    flows = np.full(N_charge, 1.0)
    sim_charge = pcm.simulate(30.0, T_htf, flows, modes, dt_step_s=600.0)

    hours_c = np.arange(N_charge) * 600 / 3600
    fig.add_trace(
        go.Scatter(x=hours_c, y=sim_charge["T_pcm"], name="T_pcm (charge)",
                   line=dict(color="firebrick", width=2)),
        row=1, col=3,
    )
    fig.add_hline(y=pcm.T_pc, line_dash="dash", line_color="gray",
                  annotation_text="T_pc", row=1, col=3)

    # --- 4) Energy stored ---
    fig.add_trace(
        go.Scatter(x=hours_c, y=sim_charge["energy_stored_kwh"],
                   name="Energy stored", line=dict(color="darkorange", width=2)),
        row=2, col=1,
    )

    # --- 5) Charge/discharge cycle ---
    N_cycle = 48
    modes_cycle = ["charge"] * 24 + ["discharge"] * 24
    T_htf_cycle = np.concatenate([np.full(24, 80.0), np.full(24, 30.0)])
    flows_cycle = np.full(N_cycle, 1.0)
    sim_cycle = pcm.simulate(40.0, T_htf_cycle, flows_cycle, modes_cycle, dt_step_s=600.0)

    hours_cycle = np.arange(N_cycle) * 600 / 3600
    fig.add_trace(
        go.Scatter(x=hours_cycle, y=sim_cycle["T_pcm"], name="T_pcm (cycle)",
                   line=dict(color="purple", width=2)),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(x=hours_cycle, y=sim_cycle["phase_fraction"],
                   name="Phase fraction", line=dict(color="green", width=2, dash="dash")),
        row=2, col=2,
    )

    # --- 6) Thermal power ---
    fig.add_trace(
        go.Scatter(x=hours_cycle, y=sim_cycle["thermal_power_kw"],
                   name="Thermal power [kW]", line=dict(color="darkorange", width=2)),
        row=2, col=3,
    )

    # Axes
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=1, col=3)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=3)
    fig.update_yaxes(title_text="Enthalpy (kJ/kg)", row=1, col=1)
    fig.update_yaxes(title_text="Phase fraction [-]", row=1, col=2)
    fig.update_yaxes(title_text="T_pcm (degC)", row=1, col=3)
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Enthalpy Method<br>"
            f"<sup>RT58 | T_pc=58 degC | L=180 kJ/kg | 5000 kg | UA_htf=500 W/K</sup>"
        ),
        height=900,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    generate_report()
