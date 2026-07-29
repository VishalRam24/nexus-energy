"""
EC207 — CO2 Compression & Pipeline — F1b Injection Pressure Effects + Leakage Losses

Extends F1a compression/pipeline model with:
  1. Injection pressure effects: higher injection pressure → higher compression SEC.
     Varies P_outlet to match reservoir injection requirements.
  2. Pipeline leakage losses: exponential with pipeline age and segment count.
     L(t) = m_dot * leakage_rate_base * t_years (linear for small rates).
  3. Compressor degradation: polytropic efficiency degrades with hours
     (fouling, seal wear): eta_p(t) = eta0 * (1 - eta_deg_rate * t).
  4. Recompression energy: boosted pressure for long pipelines with pressure loss.
  5. Net CO2 delivered vs total injected (accounts for leakage).

References:
    IPCC (2005). Special Report on CCS, Chapter 4.
    IEA GHG (2007). Pipeline Transmission of CO2 and Energy. Report 2007/18.
    McCoy, S.T. & Rubin, E.S. (2008). Energy & Environmental Science, 1, 707.
"""

import numpy as np


class CO2CompressionPipelineF1b:
    """CO2 compression + pipeline with injection pressure effects and leakage model."""

    def __init__(self, params: dict):
        c = params["compressor"]
        g = params["co2"]
        p = params["pipeline"]

        self.N = int(c["n_stages"]["value"])
        self.n_poly0 = c["polytropic_index"]["value"]
        self.eta_p0 = c["eta_polytropic"]["value"]
        self.eta_m = c["eta_mech"]["value"]
        self.T_inlet_default = c["T_inlet"]["value"]
        self.P_inlet_default = c["P_inlet"]["value"]
        self.seal_deg_rate = c["seal_degradation_rate"]["value"]
        self.eta_deg_rate = c["eta_poly_degradation"]["value"]

        self.M_CO2 = g["molar_mass"]["value"]
        self.R_s = g["R_specific"]["value"]
        self.gamma = g["gamma"]["value"]
        self.T_crit = g["T_critical"]["value"]
        self.P_crit = g["P_critical"]["value"]
        self.rho_dense = g["rho_dense_phase"]["value"]
        self.mu_dense = g["viscosity_dense"]["value"]
        self.leakage_base = g["leakage_rate_base"]["value"]

        self.roughness = p["roughness"]["value"]

    # ── Compressor degradation ────────────────────────────────────────────────

    def polytropic_efficiency(self, operating_hours):
        """eta_p(t) = eta0 * (1 - eta_deg_rate * t), floored at 0.60."""
        t = np.asarray(operating_hours, dtype=float)
        eta = self.eta_p0 * (1.0 - self.eta_deg_rate * t)
        return np.clip(eta, 0.60, self.eta_p0)

    def seal_leakage_fraction(self, operating_hours):
        """Compressor seal leakage fraction (of throughput).
        f_seal(t) = seal_deg_rate * t, capped at 5%.
        """
        t = np.asarray(operating_hours, dtype=float)
        return np.clip(self.seal_deg_rate * t, 0.0, 0.05)

    # ── Compression (with degradation) ───────────────────────────────────────

    def _stage_pressure_ratio(self, P_in, P_out):
        return (P_out / P_in) ** (1.0 / self.N)

    def sec_kwh_per_tco2(self, P_in, P_out, T_in=None, operating_hours=0):
        """SEC with degraded efficiency [kWh/tCO2]."""
        T1 = self.T_inlet_default if T_in is None else np.asarray(T_in, dtype=float)
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)
        eta_p = self.polytropic_efficiency(operating_hours)
        n = self.n_poly0

        PRs = self._stage_pressure_ratio(P_in, P_out)
        exponent = (n - 1.0) / n
        w_stage = (n / (n - 1.0)) * self.R_s * T1 * (PRs ** exponent - 1.0) / eta_p
        w_total = self.N * w_stage / self.eta_m  # J/kg
        return w_total / 3.6e6 * 1000.0  # kWh/tCO2

    def shaft_power_kw(self, m_dot_kg_s, P_in, P_out, T_in=None, operating_hours=0):
        """Shaft power [kW]."""
        m_dot = np.asarray(m_dot_kg_s, dtype=float)
        T1 = self.T_inlet_default if T_in is None else np.asarray(T_in, dtype=float)
        eta_p = self.polytropic_efficiency(operating_hours)
        n = self.n_poly0
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)

        PRs = self._stage_pressure_ratio(P_in, P_out)
        exponent = (n - 1.0) / n
        w_stage = (n / (n - 1.0)) * self.R_s * T1 * (PRs ** exponent - 1.0) / eta_p
        w_total = self.N * w_stage / self.eta_m
        return m_dot * w_total / 1000.0

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _friction_factor_churchill(self, Re, D):
        eps = self.roughness
        A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * eps / D))) ** 16
        B = (37530.0 / Re) ** 16
        return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)

    def pipeline_pressure_drop_bar(self, m_dot_kg_s, length_km, diameter_m,
                                    rho_kg_m3=None):
        """Dense-phase CO2 pipeline pressure drop [bar]."""
        m = np.asarray(m_dot_kg_s, dtype=float)
        L = np.asarray(length_km, dtype=float) * 1000.0
        D = np.asarray(diameter_m, dtype=float)
        rho = self.rho_dense if rho_kg_m3 is None else np.asarray(rho_kg_m3, dtype=float)
        A_pipe = np.pi / 4.0 * D ** 2
        v = m / (rho * A_pipe)
        Re = np.maximum(rho * v * D / self.mu_dense, 1.0)
        f = self._friction_factor_churchill(Re, D)
        dP_Pa = f * (L / D) * (rho * v ** 2 / 2.0)
        return dP_Pa / 1e5

    # ── Leakage ───────────────────────────────────────────────────────────────

    def pipeline_leakage_fraction(self, operating_hours, length_km=100.0):
        """Pipeline CO2 leakage fraction (of throughput) per year equivalent.
        Scales with pipeline age and length.
        L_frac = leakage_base * (t_years) * (L/100 km)
        (Simplified from IEA GHG 2007: 0.001-0.01%/year for modern steel pipelines)
        """
        t_years = np.asarray(operating_hours, dtype=float) / 8760.0
        L_norm = np.asarray(length_km, dtype=float) / 100.0
        frac = self.leakage_base * t_years * L_norm
        return np.clip(frac, 0.0, 0.10)

    def net_co2_delivered_kg_s(self, m_dot_kg_s, operating_hours, length_km=100.0):
        """Net CO2 delivered (kg/s) after accounting for seal and pipeline leakage."""
        m = np.asarray(m_dot_kg_s, dtype=float)
        f_seal = self.seal_leakage_fraction(operating_hours)
        f_pipe = self.pipeline_leakage_fraction(operating_hours, length_km)
        return m * (1.0 - f_seal) * (1.0 - f_pipe)

    def injection_pressure_sec_kwh_tco2(self, P_reservoir_bar, depth_m,
                                          rho_co2_kg_m3=None, operating_hours=0):
        """SEC adjusted for injection well pressure requirements.
        P_required = P_reservoir + surface_loss = typically 20-50% above reservoir P.
        Compressor must deliver P_outlet = P_required.
        """
        P_res = np.asarray(P_reservoir_bar, dtype=float)
        d = np.asarray(depth_m, dtype=float)
        rho = rho_co2_kg_m3 if rho_co2_kg_m3 is not None else self.rho_dense
        # Surface injection pressure accounting for hydrostatic head and friction
        P_wellhead = P_res - rho * 9.81 * d / 1e5  # subtract hydrostatic benefit
        P_wellhead = np.clip(P_wellhead, 80.0, 250.0)
        return self.sec_kwh_per_tco2(
            self.P_inlet_default, P_wellhead, operating_hours=operating_hours
        )

    def compute(self, P_in, P_out, m_dot_kg_s, T_in, pipeline_length_km,
                pipeline_diameter_m, operating_hours):
        """Full computation returning all outputs."""
        sec = self.sec_kwh_per_tco2(P_in, P_out, T_in, operating_hours)
        power = self.shaft_power_kw(m_dot_kg_s, P_in, P_out, T_in, operating_hours)
        dp = self.pipeline_pressure_drop_bar(m_dot_kg_s, pipeline_length_km,
                                              pipeline_diameter_m)
        P_out_pipe = np.maximum(np.asarray(P_out, dtype=float) - dp, 0.0)
        eta_p = self.polytropic_efficiency(operating_hours)
        f_seal = self.seal_leakage_fraction(operating_hours)
        f_pipe = self.pipeline_leakage_fraction(operating_hours, pipeline_length_km)
        m_net = self.net_co2_delivered_kg_s(m_dot_kg_s, operating_hours,
                                             pipeline_length_km)

        return {
            "sec_kwh_per_tco2": sec,
            "shaft_power_kw": power,
            "pipeline_dp_bar": dp,
            "pipeline_outlet_P_bar": P_out_pipe,
            "polytropic_efficiency": eta_p,
            "seal_leakage_fraction": f_seal,
            "pipeline_leakage_fraction": f_pipe,
            "net_co2_delivered_kg_s": m_net,
        }
