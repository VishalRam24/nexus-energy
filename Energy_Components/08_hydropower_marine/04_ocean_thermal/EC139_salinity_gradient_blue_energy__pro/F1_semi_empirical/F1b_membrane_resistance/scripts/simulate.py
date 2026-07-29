"""EC139 -- PRO F1b -- Simulation scenarios and HTML report"""
import json, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_simulations():
    model = ComponentModel()

    # 1. Pressure sweep at default conditions
    dP_arr = np.linspace(1.0, 28.0, 60)
    W_d_arr, w_net_arr, Jw_arr = [], [], []
    for dP in dP_arr:
        r = model.predict({"C_sw": 35.0, "C_fw": 0.5, "dP_bar": float(dP), "T_degC": 25.0})
        W_d_arr.append(float(r["power_density_W_m2"]))
        w_net_arr.append(float(r["net_energy_kwh_per_m3"]))
        Jw_arr.append(float(r["J_w_m_s"]))
    opt_dP = model._model.optimal_pressure_bar(35.0, 0.5, 25.0)

    # 2. Temperature sweep
    T_arr = np.linspace(5.0, 35.0, 30)
    w_T, cp_icp_T, cp_ecp_T = [], [], []
    for T in T_arr:
        r = model.predict({"C_sw": 35.0, "C_fw": 0.5, "dP_bar": 12.0, "T_degC": float(T)})
        w_T.append(float(r["net_energy_kwh_per_m3"]))
        cp_icp_T.append(float(r["cp_factor_ICP"]))
        cp_ecp_T.append(float(r["cp_factor_ECP"]))

    # 3. Salinity sweep
    C_sw_arr = np.linspace(25.0, 40.0, 30)
    w_C = [float(model.predict({"C_sw": float(c), "C_fw": 0.5, "dP_bar": 12.0})
                        ["net_energy_kwh_per_m3"]) for c in C_sw_arr]

    # 4. Summary table
    summary = model.predict({"C_sw": 35.0, "C_fw": 0.5, "dP_bar": opt_dP, "T_degC": 25.0})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EC139 PRO F1b Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{font-family:Arial;margin:30px;background:#f9f9f9}}</style>
</head>
<body>
<h1>EC139 — Salinity Gradient PRO — F1b Membrane Resistance Model</h1>
<p>A-B transport model with ICP/ECP concentration polarization and temperature-dependent diffusivity.</p>
<p><strong>Energy basis:</strong> per m³ freshwater permeated (Yip &amp; Elimelech 2012 Phase 7).</p>

<h2>Summary (dP_opt = {opt_dP:.1f} bar, T=25°C, C_sw=35 g/L, C_fw=0.5 g/L)</h2>
<ul>
  <li>Water flux J_w: {float(summary["J_w_m_s"]):.2e} m/s</li>
  <li>Effective ΔΠ: {float(summary["dPi_eff_bar"]):.2f} bar</li>
  <li>Power density: {float(summary["power_density_W_m2"]):.2f} W/m²</li>
  <li>Net energy: {float(summary["net_energy_kwh_per_m3"]):.4f} kWh/m³_fw</li>
  <li>Net power (A=200 m²): {float(summary["power_kw"]):.2f} kW</li>
  <li>ICP factor: {float(summary["cp_factor_ICP"]):.3f}</li>
  <li>ECP factor: {float(summary["cp_factor_ECP"]):.3f}</li>
</ul>

<div id="p1"></div>
<div id="p2"></div>
<div id="p3"></div>

<script>
Plotly.newPlot('p1', [
  {{x:{list(dP_arr)}, y:{W_d_arr}, name:'Power density W/m²', line:{{color:'blue'}}}},
  {{x:[{opt_dP},{opt_dP}], y:[0, {max(W_d_arr)*1.05}], mode:'lines', line:{{color:'red',dash:'dash'}}, name:'Optimal dP'}}
], {{title:'Power Density vs Hydraulic Pressure', xaxis:{{title:'dP [bar]'}}, yaxis:{{title:'W/m²'}}}});

Plotly.newPlot('p2', [
  {{x:{list(T_arr)}, y:{w_T}, name:'Net energy kWh/m³_fw', line:{{color:'green'}}}},
  {{x:{list(T_arr)}, y:{cp_icp_T}, name:'ICP factor', line:{{color:'orange',dash:'dot'}}, yaxis:'y2'}},
  {{x:{list(T_arr)}, y:{cp_ecp_T}, name:'ECP factor', line:{{color:'red',dash:'dot'}}, yaxis:'y2'}}
], {{title:'Temperature Effects on PRO Performance', xaxis:{{title:'T [°C]'}},
    yaxis:{{title:'Net energy [kWh/m³_fw]'}},
    yaxis2:{{title:'CP factor',overlaying:'y',side:'right'}}}});

Plotly.newPlot('p3', [
  {{x:{list(C_sw_arr)}, y:{w_C}, name:'Net energy kWh/m³_fw', line:{{color:'purple'}}}}
], {{title:'Net Energy vs Seawater Salinity (dP=12 bar, T=25°C)',
    xaxis:{{title:'C_sw [g/L]'}}, yaxis:{{title:'kWh/m³ freshwater'}}}});
</script>
</body></html>"""

    out = Path(__file__).parent.parent / "simulation_report.html"
    out.write_text(html)
    print(f"Report written: {out}")
    return {"optimal_dP_bar": opt_dP, "net_energy_at_opt": float(summary["net_energy_kwh_per_m3"])}


if __name__ == "__main__":
    results = run_simulations()
    print(json.dumps(results, indent=2))
