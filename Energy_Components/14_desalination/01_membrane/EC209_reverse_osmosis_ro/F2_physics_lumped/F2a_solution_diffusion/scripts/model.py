"""
EC209 -- Reverse Osmosis (RO) -- F2a Solution-Diffusion Membrane Model

First-principles solution-diffusion transport model with concentration
polarization for spiral-wound RO membrane elements.

Physics:
  Water flux:   Jw = A * (dP - d_pi)        [LMH]
  Salt flux:    Js = B * (Cf_m - Cp)         [g/m2/h]
  Osmotic pressure:  pi = 0.7 * C            [bar, C in g/L -- simplified van't Hoff for NaCl]
  Concentration polarization:
      Cf_m = Cp + (Cf - Cp) * exp(Jw / k)
      where k = mass transfer coefficient [m/s]
  Mass transfer:
      k = Sh * D / dh
      Sh = 0.04 * Re^0.75 * Sc^0.33  (turbulent spacer-filled channel)
  Recovery: R = Qp / Qf
  SEC = dP / (eta_pump * R * 36)  with ERD correction on brine

The element model is implicit because Jw appears on both sides (via CP).
Solved by fixed-point iteration.

Vessel model chains N elements in series: the concentrate of element i
becomes the feed of element i+1, with a small pressure drop per element.

References:
    Geise, G. M. et al. (2011). J. Membrane Science, 369, 130-138.
    Wijmans, J. G. & Baker, R. W. (1995). J. Membrane Science, 107, 1-21.
    Schock, G. & Miquel, A. (1987). Desalination, 64, 339-352.
"""

import numpy as np


class RO_SolutionDiffusion_F2a:
    """Solution-diffusion RO membrane model with concentration polarization."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.A = u["A"]["value"]                     # LMH/bar
        self.B = u["B"]["value"]                     # LMH (salt permeability)
        self.Am = u["membrane_area_m2"]["value"]     # m2 per element
        self.N_elements = int(u["N_elements"]["value"])
        self.eta_pump = u["eta_pump"]["value"]
        self.eta_ERD = u["eta_ERD"]["value"]
        self.pi_coeff = u["pi_coeff"]["value"]       # bar per g/L
        self.dP_element = u["dP_element_bar"]["value"]  # bar per element
        self.D = u["D_NaCl"]["value"]                # m2/s
        self.rho = u["rho_water"]["value"]           # kg/m3
        self.mu = u["mu_water"]["value"]             # Pa.s
        self.dh = u["dh_mm"]["value"] * 1e-3         # m
        self.L_elem = u["L_element_m"]["value"]      # m
        self.W_elem = u["W_element_m"]["value"]      # m
        self.h_spacer = u["feed_spacer_thickness_mm"]["value"] * 1e-3  # m

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def osmotic_pressure(self, C_gL):
        """Osmotic pressure [bar] from concentration [g/L]."""
        return self.pi_coeff * np.asarray(C_gL, dtype=float)

    def _mass_transfer_coeff(self, Qf_m3h):
        """Mass transfer coefficient k [m/s] using Schock & Miquel correlation
        for spacer-filled channels.

        Sh = 0.04 * Re^0.75 * Sc^0.33
        k = Sh * D / dh
        """
        # Cross-section area of feed channel
        A_cross = self.W_elem * self.h_spacer  # m2
        # Feed velocity [m/s]
        v = (Qf_m3h / 3600.0) / A_cross
        v = max(v, 1e-6)

        Re = self.rho * v * self.dh / self.mu
        Sc = self.mu / (self.rho * self.D)
        Sh = 0.04 * Re**0.75 * Sc**0.33
        k = Sh * self.D / self.dh  # m/s
        return k

    def solve_element(self, Cf_gL, P_bar, Qf_m3h, T_degC=25.0,
                      max_iter=50, tol=1e-6):
        """Solve a single membrane element using fixed-point iteration.

        Parameters
        ----------
        Cf_gL : float
            Feed concentration [g/L]
        P_bar : float
            Feed-side pressure [bar]
        Qf_m3h : float
            Feed volumetric flow rate [m3/h]
        T_degC : float
            Feed temperature [degC] (used for temperature correction)
        max_iter : int
            Maximum iterations for convergence
        tol : float
            Convergence tolerance on Jw [LMH]

        Returns
        -------
        dict with keys: Jw, Js, Qp_m3h, Qc_m3h, Cp_gL, Cc_gL, Cf_m_gL,
                        recovery, rejection, P_out_bar
        """
        Cf = float(Cf_gL)
        P = float(P_bar)
        Qf = float(Qf_m3h)

        # Temperature correction factor (Arrhenius-type, ref 25C)
        T_K = T_degC + 273.15
        T_ref = 298.15
        temp_factor = np.exp(2500.0 * (1.0 / T_ref - 1.0 / T_K))

        A_eff = self.A * temp_factor  # LMH/bar
        B_eff = self.B * temp_factor  # LMH (salt permeability also increases slightly)

        # Mass transfer coefficient
        k = self._mass_transfer_coeff(Qf)
        k_LMH = k * 3.6e6  # convert m/s to L/(m2*h) ... actually to m/h then *1000
        # k [m/s] -> k [m/h] = k*3600; Jw is in LMH = L/(m2*h)
        # Jw/k must be dimensionless: Jw [L/(m2*h)] / (k [m/s]*3600*1000 [L/(m2*h)])
        # Actually Jw [LMH] = Jw [L/(m2*h)]; convert to m/s: Jw_ms = Jw / (3.6e6)
        # ratio = Jw_ms / k = Jw / (k * 3.6e6)
        # Let's define k_norm = k * 3.6e6 so ratio = Jw / k_norm
        k_norm = k * 3.6e6  # converts k [m/s] to same units as Jw [LMH]

        # Feed osmotic pressure
        pi_f = self.osmotic_pressure(Cf)

        # Initial guess: no CP
        Jw = A_eff * max(P - pi_f, 0.0)

        for _ in range(max_iter):
            # Permeate concentration estimate
            if Jw > 1e-8:
                Cp = B_eff * Cf / (Jw + B_eff)
            else:
                Cp = Cf  # no separation at zero flux

            # Concentration polarization: membrane surface concentration
            if k_norm > 0 and Jw > 0:
                Cf_m = Cp + (Cf - Cp) * np.exp(Jw / k_norm)
            else:
                Cf_m = Cf

            # Osmotic pressures
            pi_m = self.osmotic_pressure(Cf_m)
            pi_p = self.osmotic_pressure(Cp)
            d_pi = pi_m - pi_p

            # Net driving pressure
            NDP = P - d_pi
            Jw_new = A_eff * max(NDP, 0.0)

            if abs(Jw_new - Jw) < tol:
                Jw = Jw_new
                break
            # Damped update for stability
            Jw = 0.5 * Jw + 0.5 * Jw_new

        # Final permeate concentration
        if Jw > 1e-8:
            Cp = B_eff * Cf_m / (Jw + B_eff)
        else:
            Cp = Cf

        # Salt flux
        Js = B_eff * (Cf_m - Cp)  # g/(m2*h) since B is LMH and C is g/L

        # Permeate flow [m3/h]
        Qp = Jw * self.Am / 1000.0  # LMH * m2 = L/h; /1000 = m3/h
        Qp = min(Qp, Qf * 0.95)  # cap element recovery at 95%
        Qp = max(Qp, 0.0)

        # Concentrate flow
        Qc = Qf - Qp

        # Concentrate concentration (mass balance)
        if Qc > 1e-10:
            Cc = (Cf * Qf - Cp * Qp) / Qc
        else:
            Cc = Cf

        # Element recovery
        recovery = Qp / Qf if Qf > 0 else 0.0

        # Salt rejection
        rejection = 1.0 - Cp / Cf if Cf > 0 else 0.0

        # Pressure at outlet (pressure drop across element)
        P_out = P - self.dP_element

        return {
            "Jw_LMH": Jw,
            "Js_gm2h": Js,
            "Qp_m3h": Qp,
            "Qc_m3h": Qc,
            "Cp_gL": Cp,
            "Cc_gL": Cc,
            "Cf_m_gL": Cf_m,
            "recovery": recovery,
            "rejection": rejection,
            "P_out_bar": P_out,
            "NDP_bar": P - (self.osmotic_pressure(Cf_m) - self.osmotic_pressure(Cp)),
        }

    def solve_vessel(self, Cf_gL, P_bar, Qf_m3h, T_degC=25.0,
                     N_elements=None):
        """Solve a pressure vessel with N elements in series.

        Parameters
        ----------
        Cf_gL : float   Feed concentration [g/L]
        P_bar : float   Inlet pressure [bar]
        Qf_m3h : float  Feed flow [m3/h]
        T_degC : float   Temperature [degC]
        N_elements : int  Override number of elements (default: self.N_elements)

        Returns
        -------
        dict with vessel-level results and element-by-element profiles
        """
        N = N_elements or self.N_elements

        # Element-by-element profiles
        profiles = {
            "Jw_LMH": [], "Cp_gL": [], "Cf_gL": [], "Cc_gL": [],
            "Cf_m_gL": [], "Qp_m3h": [], "Qf_m3h": [], "recovery": [],
            "P_bar": [], "NDP_bar": [],
        }

        Cf_i = float(Cf_gL)
        P_i = float(P_bar)
        Qf_i = float(Qf_m3h)

        total_Qp = 0.0
        total_salt_perm = 0.0  # g/h of salt in permeate

        for i in range(N):
            res = self.solve_element(Cf_i, P_i, Qf_i, T_degC)

            profiles["Jw_LMH"].append(res["Jw_LMH"])
            profiles["Cp_gL"].append(res["Cp_gL"])
            profiles["Cf_gL"].append(Cf_i)
            profiles["Cc_gL"].append(res["Cc_gL"])
            profiles["Cf_m_gL"].append(res["Cf_m_gL"])
            profiles["Qp_m3h"].append(res["Qp_m3h"])
            profiles["Qf_m3h"].append(Qf_i)
            profiles["recovery"].append(res["recovery"])
            profiles["P_bar"].append(P_i)
            profiles["NDP_bar"].append(res["NDP_bar"])

            total_Qp += res["Qp_m3h"]
            total_salt_perm += res["Cp_gL"] * res["Qp_m3h"]  # g/h

            # Next element input
            Cf_i = res["Cc_gL"]
            P_i = res["P_out_bar"]
            Qf_i = res["Qc_m3h"]

            if Qf_i < 0.1 or P_i < 1.0:
                break

        # Vessel-level results
        Qf_total = float(Qf_m3h)
        vessel_recovery = total_Qp / Qf_total if Qf_total > 0 else 0.0
        Cp_blend = total_salt_perm / total_Qp if total_Qp > 0 else 0.0
        vessel_rejection = 1.0 - Cp_blend / float(Cf_gL) if float(Cf_gL) > 0 else 0.0

        # Brine conditions
        Qc_vessel = Qf_total - total_Qp
        if Qc_vessel > 1e-10:
            Cc_vessel = (float(Cf_gL) * Qf_total - Cp_blend * total_Qp) / Qc_vessel
        else:
            Cc_vessel = float(Cf_gL)

        # Specific energy consumption with ERD
        P_feed = float(P_bar)
        P_brine = P_i  # pressure at vessel exit
        sec_numerator = P_feed / self.eta_pump - P_brine * (1.0 - vessel_recovery) * self.eta_ERD
        sec_bar = sec_numerator / vessel_recovery if vessel_recovery > 0 else 0.0
        sec_kwh = sec_bar / 36.0  # 1 bar*m3 = 1/36 kWh

        return {
            "Qp_m3h": total_Qp,
            "Qc_m3h": Qc_vessel,
            "Cp_gL": Cp_blend,
            "Cc_gL": Cc_vessel,
            "recovery": vessel_recovery,
            "rejection": vessel_rejection,
            "SEC_kwhm3": max(sec_kwh, 0.0),
            "P_brine_bar": P_i,
            "profiles": profiles,
            "N_elements_active": len(profiles["Jw_LMH"]),
        }

    def compute(self, feed_concentration_gL, feed_pressure_bar,
                feed_flow_m3h, temperature_degC=25.0, N_elements=None):
        """Full computation -- convenience wrapper around solve_vessel.

        Parameters
        ----------
        feed_concentration_gL : float  [g/L]
        feed_pressure_bar : float  [bar]
        feed_flow_m3h : float  [m3/h]
        temperature_degC : float  [degC]
        N_elements : int or None

        Returns
        -------
        dict: vessel-level outputs
        """
        return self.solve_vessel(
            Cf_gL=feed_concentration_gL,
            P_bar=feed_pressure_bar,
            Qf_m3h=feed_flow_m3h,
            T_degC=temperature_degC,
            N_elements=N_elements,
        )
