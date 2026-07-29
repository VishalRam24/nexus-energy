"""EC217 — TEC — F1b Temperature-Dependent Properties — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_c_fixed = 263.15  # -10C cold side
    T_h_fixed = 308.15  # 35C hot side
    Is = np.linspace(0.1, 12.0, 80)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Q_cold vs Current (at fixed T_c=-10C, T_h=35C)",
            "COP vs Current",
            "ZT(T) — Local Figure of Merit",
            "T_min vs T_hot (optimal current)",
        ],
        vertical_spacing=0.15,
    )

    # Plot 1: Q_cold vs I at different T_hot values
    for T_h in [298.15, 308.15, 323.15]:
        Q_colds = []
        for I in Is:
            r = model.predict({"T_cold_K": T_c_fixed, "T_hot_K": T_h, "I_A": float(I)})
            Q_colds.append(float(np.atleast_1d(r["Q_cold_W"])[0]))
        fig.add_trace(
            go.Scatter(x=Is, y=Q_colds, name=f"T_hot={T_h-273.15:.0f}°C", line=dict(width=2)),
            row=1, col=1
        )

    # Plot 2: COP vs I
    for T_h in [298.15, 308.15, 323.15]:
        COPs = []
        for I in Is:
            r = model.predict({"T_cold_K": T_c_fixed, "T_hot_K": T_h, "I_A": float(I)})
            COPs.append(max(0, float(np.atleast_1d(r["COP"])[0])))
        fig.add_trace(
            go.Scatter(x=Is, y=COPs, name=f"COP T_hot={T_h-273.15:.0f}°C", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: ZT(T) local
    Ts = np.linspace(230, 380, 100)
    zt_local = model._model.zt_local(Ts)
    alpha_norm = model._model.alpha(Ts) / model._model.alpha0
    k_norm = model._model.k_thermal(Ts) / model._model.k0
    sigma_norm = model._model.sigma_electrical(Ts) / model._model.sigma0
    fig.add_trace(go.Scatter(x=Ts, y=zt_local, name="ZT(T)", line=dict(width=2, color="navy")), row=2, col=1)
    for nm, vals, clr in [("alpha/alpha0", alpha_norm, "red"), ("k/k0", k_norm, "green"), ("sigma/sigma0", sigma_norm, "blue")]:
        fig.add_trace(go.Scatter(x=Ts, y=vals, name=nm, line=dict(width=1.5, dash="dot", color=clr)), row=2, col=1)

    # Plot 4: T_min achievable vs T_hot
    T_hots = np.linspace(290, 380, 50)
    T_mins = []
    for T_h in T_hots:
        I_opt = model._model.compute_optimal_current(T_c_fixed, T_h)
        r = model.predict({"T_cold_K": T_c_fixed, "T_hot_K": float(T_h), "I_A": I_opt})
        T_mins.append(float(np.atleast_1d(r["T_min_achievable_K"])[0]))
    fig.add_trace(go.Scatter(x=T_hots - 273.15, y=np.array(T_mins) - 273.15,
                             name="T_min (°C)", line=dict(width=2, color="navy")), row=2, col=2)

    fig.update_xaxes(title_text="Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_xaxes(title_text="T_hot (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Q_cold (W)", row=1, col=1)
    fig.update_yaxes(title_text="COP (-)", row=1, col=2)
    fig.update_yaxes(title_text="ZT / Normalized", row=2, col=1)
    fig.update_yaxes(title_text="T_min (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Bi2Te3 TEC | T-dependent alpha, k, sigma | Thomson correction</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
