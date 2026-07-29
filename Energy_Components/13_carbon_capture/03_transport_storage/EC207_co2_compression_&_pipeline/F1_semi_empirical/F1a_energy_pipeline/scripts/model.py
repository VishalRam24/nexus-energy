"""
EC207 — CO2 Compression & Pipeline — F1a Energy & Pipeline Model

Two sub-models:

A. CO2 COMPRESSION (1.5 bar capture → 150 bar supercritical):
   Same polytropic framework as EC016/EC191 but for CO2:
   - M = 44.01 g/mol, R_s = 188.9 J/(kg·K), γ ≈ 1.29-1.30
   - Must remain above T_critical = 304.2 K throughout (intercooling to 35°C)
   - SEC benchmark: ~100 kWh/tCO2 (IPCC 2005 full compression 1→150 bar)

B. DENSE-PHASE CO2 PIPELINE pressure drop:
   Darcy-Weisbach for incompressible dense-phase CO2:
       ΔP = f × (L/D) × (ρ × v²/2)
   where v = m_dot / (ρ × A), f from Colebrook-White (Churchill approximation)
   Minimum inlet pressure for pipeline: ≥ P_critical + margin = ~80 bar

CO2 Phase Note:
   - Subcritical (T < 304.2 K or P < 73.8 bar): gas/liquid two-phase risk
   - Supercritical / dense-phase: T > 304.2 K AND P > 73.8 bar → treat as dense liquid
   - Pipeline transport requires dense-phase to avoid two-phase flow

References:
    IPCC (2005). Special Report on Carbon Dioxide Capture and Storage. Chapter 4.
    McCoy, S.T. & Rubin, E.S. (2008). Energy & Environmental Science, 1, 707.
    IEA GHG (2011). Rotating Machinery for CO2 Compression in CCS Systems.
"""

import numpy as np


class CO2CompressionPipelineF1a:
    """CO2 compression to supercritical + dense-phase pipeline model."""

    def __init__(self, params: dict):
        c = params["compressor"]
        g = params["co2"]
        p = params["pipeline"]

        self.N = int(c["n_stages"]["value"])
        self.n_poly = c["polytropic_index"]["value"]
        self.eta_p = c["eta_polytropic"]["value"]
        self.eta_m = c["eta_mech"]["value"]
        self.T_inlet_default = c["T_inlet"]["value"]
        self.P_inlet_default = c["P_inlet"]["value"]
        self.P_sc = c["P_supercritical"]["value"]

        self.M_CO2 = g["molar_mass"]["value"]
        self.R_s = g["R_specific"]["value"]
        self.gamma = g["gamma"]["value"]
        self.T_crit = g["T_critical"]["value"]
        self.P_crit = g["P_critical"]["value"]
        self.rho_dense = g["rho_dense_phase"]["value"]
        self.mu_dense = g["viscosity_dense"]["value"]
        self.SEC_ref = g["SEC_ref"]["value"]
        self.JT_dense = g["JT_coefficient_dense"]["value"]  # K/Pa (negative for dense CO2)

        self.roughness = p["roughness"]["value"]

    # ── Compression ──────────────────────────────────────────────────────────

    def stage_pressure_ratio(self, P_in, P_out):
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)
        return (P_out / P_in) ** (1.0 / self.N)

    def stage_discharge_temperature(self, T_in, P_in, P_out):
        T_in = np.asarray(T_in, dtype=float)
        PRs = self.stage_pressure_ratio(P_in, P_out)
        exponent = (self.n_poly - 1.0) / self.n_poly
        return T_in * PRs ** exponent

    def specific_work_compression(self, P_in, P_out, T_in=None):
        """Specific polytropic shaft work for CO2 compression [J/kg]."""
        T1 = self.T_inlet_default if T_in is None else np.asarray(T_in, dtype=float)
        P_in = np.asarray(P_in, dtype=float)
        P_out = np.asarray(P_out, dtype=float)
        PRs = self.stage_pressure_ratio(P_in, P_out)
        exponent = (self.n_poly - 1.0) / self.n_poly
        w_stage = (self.n_poly / (self.n_poly - 1.0)) * self.R_s * T1 * (PRs ** exponent - 1.0) / self.eta_p
        return self.N * w_stage / self.eta_m

    def sec_kwh_per_tco2(self, P_in, P_out, T_in=None):
        """Specific energy consumption [kWh/tonne CO2]."""
        w_J_per_kg = self.specific_work_compression(P_in, P_out, T_in)
        return w_J_per_kg / 3.6e6 * 1000.0  # J/kg → kWh/tonne (1 t = 1000 kg)

    def shaft_power_kw(self, m_dot_kg_s, P_in, P_out, T_in=None):
        """Compression shaft power [kW]."""
        m_dot = np.asarray(m_dot_kg_s, dtype=float)
        return m_dot * self.specific_work_compression(P_in, P_out, T_in) / 1000.0

    # ── Pipeline (dense-phase) ───────────────────────────────────────────────

    def _friction_factor_churchill(self, Re, D):
        """Churchill (1977) friction factor approximation (valid all Re)."""
        eps = self.roughness
        # Churchill (1977) formula
        A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * eps / D))) ** 16
        B = (37530.0 / Re) ** 16
        f = 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
        return f

    def pipeline_pressure_drop_bar(self, m_dot_kg_s, length_km, diameter_m,
                                    rho_kg_m3=None):
        """
        Dense-phase CO2 pipeline pressure drop via Darcy-Weisbach [bar].

        Args:
            m_dot_kg_s  : mass flow rate [kg/s]
            length_km   : pipeline length [km]
            diameter_m  : internal diameter [m]
            rho_kg_m3   : density [kg/m³], defaults to dense-phase reference (800)
        """
        m = np.asarray(m_dot_kg_s, dtype=float)
        L = np.asarray(length_km, dtype=float) * 1000.0  # km → m
        D = np.asarray(diameter_m, dtype=float)
        rho = self.rho_dense if rho_kg_m3 is None else np.asarray(rho_kg_m3, dtype=float)
        mu = self.mu_dense

        A_pipe = np.pi / 4.0 * D ** 2
        v = m / (rho * A_pipe)
        Re = rho * v * D / mu
        Re = np.maximum(Re, 1.0)  # avoid div-by-zero
        f = self._friction_factor_churchill(Re, D)
        dP_Pa = f * (L / D) * (rho * v ** 2 / 2.0)
        return dP_Pa / 1e5  # Pa → bar

    def is_supercritical(self, T_K, P_bar):
        """True where CO2 is in dense/supercritical phase."""
        T = np.asarray(T_K, dtype=float)
        P = np.asarray(P_bar, dtype=float)
        return (T > self.T_crit) & (P > self.P_crit)
