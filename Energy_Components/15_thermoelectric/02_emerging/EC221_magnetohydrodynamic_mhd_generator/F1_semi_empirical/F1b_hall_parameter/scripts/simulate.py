"""EC221 — MHD Generator — F1b — Simulation Scenarios + HTML Report"""
import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()
    p = model.params["unit"]

    # --- Scenario 1: Power vs load factor K, varying Hall parameter ---
    K_range = np.linspace(0.01, 0.99, 200)
    betas = [0.0, 1.0, 3.0, 5.0]
    s1_powers = {}
    for beta in betas:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0,
                           "K": K_range, "beta": beta})
        s1_powers[beta] = np.asarray(r["power_elec_W"]) / 1e6  # MW

    # --- Scenario 2: Q_in stagnation vs velocity (compare with kinetic-only) ---
    u_arr = np.linspace(200.0, 1500.0, 200)
    r_s2 = model.predict({"sigma": 10.0, "u": u_arr, "B": 5.0, "K": 0.5, "beta": 3.0})
    Q_stag = np.asarray(r_s2["heat_input_stag_W"]) / 1e6
    rho = p["rho_plasma"]["value"]
    w = p["channel_width"]["value"]
    h = p["channel_height"]["value"]
    Q_kinetic = 0.5 * rho * u_arr ** 3 * w * h / 1e6

    # --- Scenario 3: Power vs magnetic field, varying beta ---
    B_arr = np.linspace(1.0, 9.0, 200)
    s3_powers = {}
    for beta in [0.0, 3.0, 6.0]:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": B_arr, "K": 0.5, "beta": beta})
        s3_powers[beta] = np.asarray(r["power_elec_W"]) / 1e6

    # --- Scenario 4: Efficiency vs plasma temperature ---
    T_arr = np.linspace(1800.0, 3500.0, 200)
    r_s4 = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5,
                           "beta": 3.0, "T_plasma_K": T_arr})
    eta_elec = np.asarray(r_s4["eta_electric"])
    sigma_T = np.asarray(r_s4["sigma_actual_Sm"])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Power vs Load Factor K (Hall Parameter Effect)",
                "Stagnation vs Kinetic Q_in (First-Law Correction)",
                "Power vs Magnetic Field (Hall Effect)",
                "Efficiency & Conductivity vs Plasma Temperature",
            ],
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, beta in enumerate(betas):
            fig.add_trace(go.Scatter(x=K_range, y=s1_powers[beta],
                                     name=f"beta={beta}",
                                     line=dict(color=colors[i]),
                                     legendgroup=f"beta{beta}"),
                          row=1, col=1)

        fig.add_trace(go.Scatter(x=u_arr, y=Q_stag, name="Q_in stagnation",
                                  line=dict(color="blue")), row=1, col=2)
        fig.add_trace(go.Scatter(x=u_arr, y=Q_kinetic, name="Q_in kinetic (0.5ρu³)",
                                  line=dict(color="red", dash="dash")), row=1, col=2)

        for i, beta in enumerate([0.0, 3.0, 6.0]):
            fig.add_trace(go.Scatter(x=B_arr, y=s3_powers[beta],
                                      name=f"beta={beta}",
                                      line=dict(color=colors[i]),
                                      legendgroup=f"beta{beta}",
                                      showlegend=False),
                          row=2, col=1)

        fig.add_trace(go.Scatter(x=T_arr, y=eta_elec * 100, name="eta_electric (%)",
                                  line=dict(color="green")), row=2, col=2)
        fig.add_trace(go.Scatter(x=T_arr, y=sigma_T, name="sigma_actual (S/m)",
                                  line=dict(color="purple"),
                                  yaxis="y4"), row=2, col=2)

        fig.update_xaxes(title_text="Load factor K [-]", row=1, col=1)
        fig.update_yaxes(title_text="Power [MW]", row=1, col=1)
        fig.update_xaxes(title_text="Plasma velocity [m/s]", row=1, col=2)
        fig.update_yaxes(title_text="Heat flux [MW]", row=1, col=2)
        fig.update_xaxes(title_text="Magnetic field B [T]", row=2, col=1)
        fig.update_yaxes(title_text="Power [MW]", row=2, col=1)
        fig.update_xaxes(title_text="Plasma temperature [K]", row=2, col=2)
        fig.update_yaxes(title_text="eta_electric [%] / sigma [S/m]", row=2, col=2)

        fig.update_layout(
            title="EC221 MHD Generator F1b — Hall Parameter + Stagnation Enthalpy Model",
            height=800,
        )

        out = Path(__file__).parent.parent / "simulation_report.html"
        fig.write_html(str(out))
        print(f"Report saved: {out}")
    except ImportError:
        print("Plotly not installed — skipping HTML report.")

    # Print design-point summary
    r0 = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": 0.5,
                        "beta": 3.0, "T_plasma_K": 2500.0})
    print("\nEC221 MHD F1b — Design Point Summary:")
    for k, v in r0.items():
        val = float(np.atleast_1d(v)[0])
        print(f"  {k:30s} = {val:.4g}")


if __name__ == "__main__":
    run_simulations()
