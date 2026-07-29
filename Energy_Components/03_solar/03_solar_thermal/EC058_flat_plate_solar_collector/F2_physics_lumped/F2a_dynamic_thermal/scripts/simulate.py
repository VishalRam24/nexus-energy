"""
EC058 -- Flat Plate Solar Collector -- F2a Dynamic Thermal -- Simulation scenarios.

Generates an interactive HTML report with Plotly showing:
  1. Steady-state efficiency curve vs (T_m - T_amb)/G
  2. Dynamic cold-start response under constant irradiance
  3. Cloud passage transient (step change in irradiance)
  4. Full-day simulation with realistic solar profile
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import FlatPlateCollectorF2a

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed, skipping HTML report generation.")


def load_model():
    params_path = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
    with open(params_path) as f:
        params = json.load(f)
    return FlatPlateCollectorF2a(params)


def scenario_efficiency_curve(m):
    """Steady-state efficiency vs reduced temperature (T_in - T_amb)/G."""
    G = 800.0
    T_amb = 20.0
    T_ins = np.linspace(20, 100, 50)

    etas = []
    x_reduced = []
    for T_in in T_ins:
        r = m.steady_state(G, T_in, T_amb)
        etas.append(r["efficiency"])
        x_reduced.append((T_in - T_amb) / G)

    return np.array(x_reduced), np.array(etas), T_ins


def scenario_cold_start(m):
    """Dynamic cold start: collector at ambient, sun turns on."""
    t_end = 3600.0
    t_eval = np.linspace(0, t_end, 300)
    T_amb = 20.0

    res = m.simulate(
        t_span=(0, t_end), t_eval=t_eval, T_m0=T_amb,
        G_func=lambda t: 800.0,
        T_in_func=lambda t: T_amb,
        T_amb_func=lambda t: T_amb,
    )
    return res


def scenario_cloud_passage(m):
    """Cloud passes at t=1800s, irradiance drops for 10 min, then recovers."""
    t_end = 5400.0
    t_eval = np.linspace(0, t_end, 400)
    T_amb = 25.0
    T_in = 35.0

    def G_func(t):
        if 1800 <= t < 2400:
            return 150.0  # cloud
        return 800.0

    res = m.simulate(
        t_span=(0, t_end), t_eval=t_eval, T_m0=T_in,
        G_func=G_func,
        T_in_func=lambda t: T_in,
        T_amb_func=lambda t: T_amb,
    )

    G_profile = np.array([G_func(t) for t in t_eval])
    return res, G_profile


def scenario_full_day(m):
    """Simulated full day with bell-shaped irradiance profile."""
    t_end = 24 * 3600.0
    t_eval = np.linspace(0, t_end, 1000)

    def G_func(t):
        hour = t / 3600.0
        if 6 < hour < 18:
            return 900.0 * np.sin(np.pi * (hour - 6) / 12)
        return 0.0

    def T_amb_func(t):
        hour = t / 3600.0
        return 15.0 + 10.0 * np.sin(np.pi * (hour - 6) / 24)

    T_in = 30.0

    res = m.simulate(
        t_span=(0, t_end), t_eval=t_eval, T_m0=T_in,
        G_func=G_func,
        T_in_func=lambda t: T_in,
        T_amb_func=T_amb_func,
    )

    G_profile = np.array([G_func(t) for t in t_eval])
    T_amb_profile = np.array([T_amb_func(t) for t in t_eval])
    return res, G_profile, T_amb_profile


def build_report(m):
    if not HAS_PLOTLY:
        print("Plotly not available. Run: pip install plotly")
        return

    # ---- Scenario 1: Efficiency curve ----
    x_red, etas, T_ins = scenario_efficiency_curve(m)

    # ---- Scenario 2: Cold start ----
    res_cold = scenario_cold_start(m)

    # ---- Scenario 3: Cloud passage ----
    res_cloud, G_cloud = scenario_cloud_passage(m)

    # ---- Scenario 4: Full day ----
    res_day, G_day, T_amb_day = scenario_full_day(m)

    # Build figure
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Steady-State Efficiency Curve",
            "Cold Start: Temperature Response",
            "Cold Start: Heat Flows",
            "Cloud Passage: Irradiance & Temperature",
            "Cloud Passage: Useful Heat",
            "Full Day: Irradiance & Ambient",
            "Full Day: Collector Temperatures",
            "Full Day: Efficiency & Useful Heat",
        ),
        vertical_spacing=0.06,
    )

    # 1. Efficiency curve
    fig.add_trace(go.Scatter(
        x=x_red, y=etas, mode="lines", name="eta(x)",
        line=dict(color="firebrick", width=2),
    ), row=1, col=1)
    fig.update_xaxes(title_text="(T_in - T_amb) / G [m2.K/W]", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency [-]", row=1, col=1)

    # 2. Cold start temperatures
    t_min = res_cold["t"] / 60.0
    fig.add_trace(go.Scatter(
        x=t_min, y=res_cold["T_mean_C"], mode="lines", name="T_mean",
        line=dict(color="red"),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=t_min, y=res_cold["T_outlet_C"], mode="lines", name="T_outlet",
        line=dict(color="orange", dash="dash"),
    ), row=1, col=2)
    fig.update_xaxes(title_text="Time [min]", row=1, col=2)
    fig.update_yaxes(title_text="Temperature [C]", row=1, col=2)

    # 3. Cold start heat flows
    fig.add_trace(go.Scatter(
        x=t_min, y=res_cold["Q_solar_W"], mode="lines", name="Q_solar",
        line=dict(color="gold"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=res_cold["Q_useful_W"], mode="lines", name="Q_useful",
        line=dict(color="green"),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_min, y=res_cold["Q_loss_W"], mode="lines", name="Q_loss",
        line=dict(color="blue"),
    ), row=2, col=1)
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_yaxes(title_text="Heat [W]", row=2, col=1)

    # 4. Cloud passage
    t_min_c = res_cloud["t"] / 60.0
    fig.add_trace(go.Scatter(
        x=t_min_c, y=G_cloud, mode="lines", name="G_T",
        line=dict(color="gold"),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=t_min_c, y=res_cloud["T_mean_C"], mode="lines", name="T_mean",
        line=dict(color="red"),
        yaxis="y2",
    ), row=2, col=2)
    fig.update_xaxes(title_text="Time [min]", row=2, col=2)
    fig.update_yaxes(title_text="G [W/m2] / T [C]", row=2, col=2)

    # 5. Cloud passage useful heat
    fig.add_trace(go.Scatter(
        x=t_min_c, y=res_cloud["Q_useful_W"], mode="lines", name="Q_useful",
        line=dict(color="green"),
    ), row=3, col=1)
    fig.update_xaxes(title_text="Time [min]", row=3, col=1)
    fig.update_yaxes(title_text="Q_useful [W]", row=3, col=1)

    # 6. Full day irradiance & ambient
    t_hr = res_day["t"] / 3600.0
    fig.add_trace(go.Scatter(
        x=t_hr, y=G_day, mode="lines", name="G_T",
        line=dict(color="gold"),
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=t_hr, y=T_amb_day, mode="lines", name="T_amb",
        line=dict(color="lightblue", dash="dot"),
    ), row=3, col=2)
    fig.update_xaxes(title_text="Time [hr]", row=3, col=2)
    fig.update_yaxes(title_text="G [W/m2] / T [C]", row=3, col=2)

    # 7. Full day temperatures
    fig.add_trace(go.Scatter(
        x=t_hr, y=res_day["T_mean_C"], mode="lines", name="T_mean",
        line=dict(color="red"),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=t_hr, y=res_day["T_outlet_C"], mode="lines", name="T_outlet",
        line=dict(color="orange", dash="dash"),
    ), row=4, col=1)
    fig.update_xaxes(title_text="Time [hr]", row=4, col=1)
    fig.update_yaxes(title_text="Temperature [C]", row=4, col=1)

    # 8. Full day efficiency + Q_useful
    fig.add_trace(go.Scatter(
        x=t_hr, y=res_day["efficiency"], mode="lines", name="eta",
        line=dict(color="green"),
    ), row=4, col=2)
    fig.update_xaxes(title_text="Time [hr]", row=4, col=2)
    fig.update_yaxes(title_text="Efficiency [-]", row=4, col=2)

    fig.update_layout(
        height=1600, width=1100,
        title_text="EC058 Flat Plate Solar Collector -- F2a Dynamic Thermal Model",
        showlegend=False,
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_path)
    print(f"Report saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    m = load_model()
    build_report(m)
