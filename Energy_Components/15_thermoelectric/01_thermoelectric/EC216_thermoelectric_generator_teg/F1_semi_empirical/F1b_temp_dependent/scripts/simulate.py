"""EC216 — TEG — F1b Temperature-Dependent — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import TEGF1b
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    T_hots = np.linspace(323, 573, 50)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "ZT(T) — Local Figure of Merit",
            "Average ZT vs T_hot",
            "Efficiency vs T_hot",
            "Power Density vs T_hot",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Local ZT vs T
    Ts = np.linspace(273, 573, 100)
    zt_local = model._model.zt_local(Ts)
    fig.add_trace(
        go.Scatter(x=Ts, y=zt_local, name="ZT(T)", line=dict(width=2, color="navy")),
        row=1, col=1
    )

    # Also show alpha, k, sigma on secondary traces (normalized)
    alpha_norm = model._model.alpha(Ts) / model._model.alpha0
    k_norm = model._model.k_thermal(Ts) / model._model.k0
    sigma_norm = model._model.sigma_electrical(Ts) / model._model.sigma0
    for name, vals, color in [("alpha/alpha0", alpha_norm, "red"),
                               ("k/k0", k_norm, "green"),
                               ("sigma/sigma0", sigma_norm, "blue")]:
        fig.add_trace(
            go.Scatter(x=Ts, y=vals, name=name, line=dict(width=1.5, dash="dot", color=color)),
            row=1, col=1
        )

    # Plot 2: Average ZT vs T_hot at different T_cold
    for T_c in [273, 293, 313]:
        zt_avg = []
        for T_h in T_hots:
            if T_h > T_c + 10:
                r = model.predict({"T_hot_K": float(T_h), "T_cold_K": float(T_c)})
                zt_avg.append(float(np.atleast_1d(r["zt_average"])[0]))
            else:
                zt_avg.append(0.0)
        fig.add_trace(
            go.Scatter(x=T_hots, y=zt_avg, name=f"Tc={T_c}K", line=dict(width=2)),
            row=1, col=2
        )

    # Plot 3: Efficiency vs T_hot
    for T_c in [273, 293, 313]:
        etas = []
        for T_h in T_hots:
            if T_h > T_c + 10:
                r = model.predict({"T_hot_K": float(T_h), "T_cold_K": float(T_c)})
                etas.append(float(np.atleast_1d(r["efficiency"])[0]) * 100)
            else:
                etas.append(0.0)
        fig.add_trace(
            go.Scatter(x=T_hots, y=etas, name=f"eta Tc={T_c}K", line=dict(width=2)),
            row=2, col=1
        )

    # Plot 4: Power density vs T_hot
    for T_c in [273, 293, 313]:
        pd = []
        for T_h in T_hots:
            if T_h > T_c + 10:
                r = model.predict({"T_hot_K": float(T_h), "T_cold_K": float(T_c)})
                pd.append(float(np.atleast_1d(r["power_density_w_cm2"])[0]))
            else:
                pd.append(0.0)
        fig.add_trace(
            go.Scatter(x=T_hots, y=pd, name=f"PD Tc={T_c}K", line=dict(width=2)),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_xaxes(title_text="T_hot (K)", row=1, col=2)
    fig.update_xaxes(title_text="T_hot (K)", row=2, col=1)
    fig.update_xaxes(title_text="T_hot (K)", row=2, col=2)
    fig.update_yaxes(title_text="ZT / Normalized", row=1, col=1)
    fig.update_yaxes(title_text="ZT_average (-)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Power Density (W/cm2)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Bi2Te3 module | Temperature-dependent alpha, k, sigma</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
