"""
EC146 -- Torrefaction Reactor -- F2a Two-Step Arrhenius Kinetics + Reactor ODE

Physics-lumped (0D) model of mild thermal pretreatment of biomass
(~250-300 degC, inert atmosphere). The volatile-yielding decomposition is
dominated by hemicellulose, which Di Blasi & Lanzetta (1997) describe with a
TWO-STEP first-order Arrhenius mechanism:

        k1                          k2
    B  ---->  c1 * C1 + (1-c1) V1 ;  C1  ---->  c2 * C2 + (1-c2) V2

    B  = virgin (hemicellulose-rich) biomass
    C1 = intermediate solid (active char)
    C2 = final torrefied char
    V1, V2 = released volatiles (torgas: CO2, H2O, acetic acid, light organics)

Only the THERMALLY REACTIVE fraction of the feed (hemicellulose + amorphous
cellulose, `reactive_frac`) undergoes this decomposition; the balance
(crystalline cellulose + lignin) is treated as an INERT solid `I` that
survives mild torrefaction (Bergman 2005; Prins 2006). The Di Blasi-Lanzetta
kinetics were measured on xylan/hemicellulose, so applying them to the
reactive fraction is physically faithful.

Mass ODEs (per kg of dry feed, all in kg):
    dB /dt = -k1 * B
    dC1/dt =  c1 * k1 * B  - k2 * C1
    dC2/dt =  c2 * k2 * C1
    dV /dt = (1-c1)*k1*B + (1-c2)*k2*C1            (released volatiles)
    I      =  (1 - reactive_frac) * B0  = const    (inert solid)

    Solid mass  S(t) = I + B + C1 + C2
    Mass yield  Y_m  = S(t) / B0
    Conservation:  I + B + C1 + C2 + V == B0   (enforced to machine precision)

Arrhenius rate (s^-1):
    k_i(T) = A_i * exp(-E_i / (R T))

Coupled reactor energy balance (lumped solid, batch / per-kg basis):
    (S*cp) dT/dt = hA (T_wall - T) - dh_rxn * dV/dt
    where dV/dt > 0 is the volatile release rate and dh_rxn<0 (exothermic)
    adds heat. Solved together with the mass ODEs via scipy.solve_ivp.

Energy yield (densification — the value proposition of torrefaction):
    E_solid(t) = C2*LHV_char + (B+C1)*LHV_raw          [MJ]
    Y_e = E_solid(t) / (B0 * LHV_raw)
    Because LHV_char > LHV_raw and low-energy oxygenates leave with the
    volatiles, Y_e > Y_m always (energy concentrates in the solid).
    HHV/LHV upgrade ratio = LHV_solid_product / LHV_raw > 1.

References:
    Di Blasi, C. & Lanzetta, M. (1997). Intrinsic kinetics of isothermal
        xylan degradation in inert atmosphere. J. Anal. Appl. Pyrolysis
        40-41, 287-303.
    Bergman, P.C.A. et al. (2005). Torrefaction for biomass co-firing in
        existing coal-fired power stations. ECN-C-05-073.
    Prins, M.J. et al. (2006). Torrefaction of wood. Part 1 & 2.
        J. Anal. Appl. Pyrolysis 77, 28-43.
    Bates, R.B. & Ghoniem, A.F. (2012). Biomass torrefaction: Modeling of
        volatile and solid product evolution kinetics. Bioresour. Technol.
        124, 460-469.
"""

import numpy as np
from scipy.integrate import solve_ivp


class TorrefactionF2a:
    """Two-step Arrhenius torrefaction kinetics coupled to a reactor energy balance."""

    R = 8.314  # J/(mol.K)

    def __init__(self, params: dict):
        u = params["unit"]
        # Kinetics (Di Blasi & Lanzetta 1997)
        self.A1 = float(u["A1"]["value"])
        self.E1 = float(u["E1"]["value"])
        self.A2 = float(u["A2"]["value"])
        self.E2 = float(u["E2"]["value"])
        self.c1 = float(u["c1_solid_frac"]["value"])   # solid fraction step 1
        self.c2 = float(u["c2_solid_frac"]["value"])   # solid fraction step 2
        self.reactive_frac = float(u["reactive_frac"]["value"])  # reactive mass frac

        # Energy / LHV
        self.LHV_raw = float(u["LHV_raw_MJ_kg"]["value"])
        self.LHV_char = float(u["LHV_char_MJ_kg"]["value"])
        self.LHV_vol = float(u["LHV_volatiles_MJ_kg"]["value"])

        # Thermal
        self.dh_rxn = float(u["delta_h_rxn_kJ_kg"]["value"]) * 1e3  # J/kg volatiles
        self.m_feed = float(u["m_feed_kg"]["value"])
        self.cp = float(u["cp_solid_J_kgK"]["value"])
        self.hA = float(u["hA_heat_W_K"]["value"])
        self.T_wall_default = float(u["T_wall_degC"]["value"]) + 273.15
        self.T_feed_default = float(u["T_feed_degC"]["value"]) + 273.15

    # ------------------------------------------------------------------
    # Arrhenius rate constants
    # ------------------------------------------------------------------
    def k1(self, T_K):
        """Step-1 rate constant [1/s] (B -> C1 + V1)."""
        return self.A1 * np.exp(-self.E1 / (self.R * T_K))

    def k2(self, T_K):
        """Step-2 rate constant [1/s] (C1 -> C2 + V2)."""
        return self.A2 * np.exp(-self.E2 / (self.R * T_K))

    # ------------------------------------------------------------------
    # LHV of the current solid product (intensive, MJ/kg)
    # ------------------------------------------------------------------
    def solid_LHV(self, I, B, C1, C2):
        """Mass-weighted LHV of the solid [MJ/kg]. Char is energy-dense."""
        S = I + B + C1 + C2
        if S <= 1e-12:
            return self.LHV_char
        # inert + virgin + active solid carry raw LHV; final char carries upgraded LHV
        return (C2 * self.LHV_char + (I + B + C1) * self.LHV_raw) / S

    # ------------------------------------------------------------------
    # Coupled ODE RHS:  state = [B, C1, C2, V, T]   (I is constant)
    # ------------------------------------------------------------------
    def _rhs(self, t, y, T_wall, I):
        B, C1, C2, V, T = y
        B = max(B, 0.0)
        C1 = max(C1, 0.0)

        k1 = self.k1(T)
        k2 = self.k2(T)

        r1 = k1 * B          # B consumption rate
        r2 = k2 * C1         # C1 consumption rate

        dB = -r1
        dC1 = self.c1 * r1 - r2
        dC2 = self.c2 * r2
        dV = (1.0 - self.c1) * r1 + (1.0 - self.c2) * r2   # volatile release rate

        # Reactor energy balance on lumped solid (inert + reacting solid)
        S = I + B + C1 + C2
        heat_cap = max(S * self.cp, 1e-9)
        # dh_rxn<0 (exothermic) -> -dh_rxn*dV > 0 adds heat
        dT = (self.hA * (T_wall - T) - self.dh_rxn * dV) / heat_cap

        return [dB, dC1, dC2, dV, dT]

    # ------------------------------------------------------------------
    # Full transient simulation
    # ------------------------------------------------------------------
    def simulate(self, T_set_degC=280.0, residence_time_min=30.0,
                 T0_degC=25.0, dt_s=5.0):
        """
        Integrate the coupled kinetic + thermal ODEs over the residence time.

        Parameters
        ----------
        T_set_degC : float   Reactor wall / heating-medium temperature [degC]
        residence_time_min : float  Residence (reaction) time [min]
        T0_degC : float      Initial solid temperature [degC]
        dt_s : float         Output sampling interval [s]

        Returns
        -------
        dict with time-series arrays and final scalar metrics.
        """
        T_wall = float(T_set_degC) + 273.15
        T0 = float(T0_degC) + 273.15
        t_end = float(residence_time_min) * 60.0

        feed0 = self.m_feed
        I = (1.0 - self.reactive_frac) * feed0       # inert solid (constant)
        B0 = self.reactive_frac * feed0               # reactive virgin biomass
        y0 = [B0, 0.0, 0.0, 0.0, T0]

        t_eval = np.arange(0.0, t_end + 1e-9, dt_s)
        if t_eval[-1] < t_end - 1e-9:
            t_eval = np.append(t_eval, t_end)

        sol = solve_ivp(
            self._rhs, (0.0, t_end), y0, t_eval=t_eval,
            args=(T_wall, I), method="LSODA", rtol=1e-7, atol=1e-10,
        )

        B, C1, C2, V, T = sol.y
        S = I + B + C1 + C2

        mass_yield = S / feed0
        # intensive solid LHV at each time
        LHV_solid = np.array([self.solid_LHV(I, b, c1, c2)
                              for b, c1, c2 in zip(B, C1, C2)])
        E_solid = S * LHV_solid                 # MJ (per kg feed -> MJ/kg_feed basis)
        E_in = feed0 * self.LHV_raw
        energy_yield = E_solid / E_in
        hhv_upgrade = LHV_solid / self.LHV_raw   # densification ratio (>1)
        conversion = V / feed0                   # fraction of feed mass released as volatiles

        return {
            "t": sol.t / 60.0,                   # min
            "B": B, "C1": C1, "C2": C2,
            "solid_mass": S,
            "volatiles_mass": V,
            "temperature": T,                    # K
            "temperature_degC": T - 273.15,
            "mass_yield": mass_yield,
            "energy_yield": energy_yield,
            "LHV_solid": LHV_solid,              # MJ/kg
            "hhv_upgrade": hhv_upgrade,
            "conversion": conversion,
            # final-state scalars
            "mass_yield_final": float(mass_yield[-1]),
            "energy_yield_final": float(energy_yield[-1]),
            "LHV_solid_final": float(LHV_solid[-1]),
            "hhv_upgrade_final": float(hhv_upgrade[-1]),
            "conversion_final": float(conversion[-1]),
            "temperature_final_degC": float(T[-1] - 273.15),
            "mass_balance_residual": float(np.max(np.abs(S + V - feed0))),
            "success": bool(sol.success),
        }


if __name__ == "__main__":
    import json, os
    p = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
    with open(p) as f:
        params = json.load(f)
    m = TorrefactionF2a(params)
    r = m.simulate(T_set_degC=280.0, residence_time_min=30.0)
    print(f"mass_yield={r['mass_yield_final']:.3f}  "
          f"energy_yield={r['energy_yield_final']:.3f}  "
          f"HHV_upgrade={r['hhv_upgrade_final']:.3f}  "
          f"T_final={r['temperature_final_degC']:.1f} degC  "
          f"mass_residual={r['mass_balance_residual']:.2e}")
