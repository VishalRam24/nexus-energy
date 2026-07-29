"""
EC210 -- Electrodialysis (ED) -- F2a Ion-Transport Stack Model

Physics-lumped (1D plug-flow) model of an electrodialysis stack. Ions are driven
through alternating cation- (CEM) and anion-exchange (AEM) membranes by an applied
current. The diluate stream is progressively desalted along the flow path while the
concentrate stream is enriched; salt is conserved between the two.

Core physics
------------
1. Faradaic ion flux through the membranes (Nernst-Planck migration term):
       J_salt = xi * I / (z * F)          [mol/s per cell pair]
   where xi is the *current efficiency* -- the fraction of charge carried by the
   desired counter-ions through both membranes:
       xi <= current_efficiency_max,  bounded above by the membrane transport numbers
       (xi_membrane = (t_CEM - t_sol) + (t_AEM - (1 - t_sol)))  (Strathmann 2004, Eq. 5-?).

2. Limiting current density (concentration polarization). The counter-ion
   concentration at the depleted membrane face cannot fall below zero, giving the
   classic limiting current (Cowan-Brown / Pierce form, Strathmann 2010):
       i_lim = (z F D c_dil) / (delta_bl * (t_mem - t_sol))      [A/cm2]
   The current density is capped at a fraction of i_lim; above i_lim, water
   splitting and large polarization losses set in.

3. Salt mass balance along the flow path (the lumped ODE, integrated in space).
   With diluate volumetric flow Q_d and N cell pairs, marching distance x:
       dc_dil/dx = - N * i(x) * xi * A_strip / (z F Q_d)
       dc_con/dx = + N * i(x) * xi * A_strip / (z F Q_c)
   driven by the local (polarization-limited) current density i(x).

4. Stack voltage = ohmic (membranes + diluate + concentrate solution resistance)
   + membrane (Donnan / Nernst concentration) potential across each pair:
       U_pair = i * (R_CEM + R_AEM + R_dil(c) + R_con(c))
                + 2 * (R T / z F) * ln(c_con / c_dil)
       U_stack = N * U_pair

Outputs: SEC (kWh/m3 diluate produced), current efficiency, recovery, stack
voltage, product (diluate) concentration, limiting-current margin.

References
----------
Strathmann, H. (2004). Ion-Exchange Membrane Separation Processes. Elsevier.
Strathmann, H. (2010). Electrodialysis, a mature technology with a multitude of
    new applications. Desalination 264, 268-288.
Ortiz, J.M. et al. (2005). Brackish water desalination by electrodialysis:
    batch recirculation operation modeling. J. Membrane Sci. 252, 65-75.
Nikonenko, V.V. et al. (2014). Desalination at overlimiting currents.
    Adv. Colloid Interface Sci. 235, 233-246.
"""

import numpy as np
from scipy.integrate import solve_ivp

F = 96485.0      # Faraday constant, C/mol
R = 8.314        # gas constant, J/(mol.K)


class ElectrodialysisF2a:
    """ED stack ion-transport model (1D plug-flow ODE in space)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.N = u["N_cell_pairs"]["value"]
        self.A_cm2 = u["A_membrane_cm2"]["value"]            # cm2 per cell pair
        self.h_cm = u["h_cell_cm"]["value"]                  # channel thickness, cm
        self.t_CEM = u["t_transport_CEM"]["value"]
        self.t_AEM = u["t_transport_AEM"]["value"]
        self.t_sol = u["t_transport_sol"]["value"]
        self.D = u["D_salt_cm2_s"]["value"]                  # cm2/s
        self.delta = u["delta_bl_cm"]["value"]               # cm
        self.R_CEM = u["R_area_CEM_ohm_cm2"]["value"]        # ohm.cm2
        self.R_AEM = u["R_area_AEM_ohm_cm2"]["value"]        # ohm.cm2
        self.Lambda0 = u["Lambda_eq_S_cm2_mol"]["value"]     # S.cm2/mol
        self.lam_f = u["lambda_factor"]["value"]
        self.xi_max = u["current_efficiency_max"]["value"]
        self.T = u["T_K"]["value"]
        self.z = u["z_ion"]["value"]

        # Membrane-pair current efficiency from transport numbers
        # (counter-ion flux gain across CEM for cations + across AEM for anions)
        xi_mem = (self.t_CEM - self.t_sol) + (self.t_AEM - (1.0 - self.t_sol))
        self.xi = float(min(self.xi_max, max(0.0, xi_mem)))

    # ------------------------------------------------------------------
    # Current efficiency
    # ------------------------------------------------------------------
    def current_efficiency(self):
        """Stack current (utilization) efficiency [-]."""
        return self.xi

    # ------------------------------------------------------------------
    # Limiting current density (concentration polarization)
    # ------------------------------------------------------------------
    def limiting_current_density(self, c_dil_mol_m3):
        """
        Limiting current density [A/cm2] at the depleted membrane surface.

        i_lim = z F D c / (delta * (t_mem - t_sol))

        c is converted mol/m3 -> mol/cm3 to keep CGS-electrical units (A/cm2).
        """
        c = np.maximum(np.asarray(c_dil_mol_m3, dtype=float), 1e-9)
        c_cm3 = c * 1e-6                          # mol/m3 -> mol/cm3
        dt_mem = max(self.t_CEM - self.t_sol, 1e-6)
        i_lim = (self.z * F * self.D * c_cm3) / (self.delta * dt_mem)
        return i_lim                              # A/cm2

    # ------------------------------------------------------------------
    # Solution conductivity / resistance of a channel
    # ------------------------------------------------------------------
    def solution_resistance(self, c_mol_m3):
        """
        Areal resistance of one solution channel [ohm.cm2].

        kappa = Lambda * c   (kappa in S/cm with c in mol/cm3),
        R_area = h / kappa.
        """
        c = np.maximum(np.asarray(c_mol_m3, dtype=float), 1e-6)
        c_cm3 = c * 1e-6                          # mol/cm3
        kappa = self.lam_f * self.Lambda0 * c_cm3  # S/cm
        kappa = np.maximum(kappa, 1e-9)
        return self.h_cm / kappa                  # ohm.cm2

    # ------------------------------------------------------------------
    # Membrane (concentration / Donnan) potential across a cell pair
    # ------------------------------------------------------------------
    def membrane_potential(self, c_dil_mol_m3, c_con_mol_m3):
        """Concentration potential across CEM+AEM of one cell pair [V]."""
        c_d = np.maximum(np.asarray(c_dil_mol_m3, dtype=float), 1e-6)
        c_c = np.maximum(np.asarray(c_con_mol_m3, dtype=float), 1e-6)
        # Two membranes (CEM + AEM), apparent permselectivity ~ avg transport no.
        perm = 0.5 * (self.t_CEM + self.t_AEM)
        return 2.0 * perm * (R * self.T) / (self.z * F) * np.log(c_c / c_d)

    # ------------------------------------------------------------------
    # Operating current density (capped below limiting current)
    # ------------------------------------------------------------------
    def operating_current_density(self, i_applied_A_m2, c_dil_mol_m3,
                                  limiting_fraction=0.8):
        """
        Local current density [A/cm2], capped at limiting_fraction * i_lim.

        i_applied given in A/m2 (engineering convention) -> A/cm2.
        """
        i_app = np.asarray(i_applied_A_m2, dtype=float) * 1e-4   # A/m2 -> A/cm2
        i_lim = self.limiting_current_density(c_dil_mol_m3)
        i_cap = limiting_fraction * i_lim
        return np.minimum(i_app, i_cap)

    # ------------------------------------------------------------------
    # Cell-pair voltage at a local state
    # ------------------------------------------------------------------
    def cell_pair_voltage(self, i_A_cm2, c_dil_mol_m3, c_con_mol_m3):
        """Voltage across one cell pair [V] = ohmic + concentration potential."""
        R_dil = self.solution_resistance(c_dil_mol_m3)
        R_con = self.solution_resistance(c_con_mol_m3)
        R_tot = self.R_CEM + self.R_AEM + R_dil + R_con     # ohm.cm2
        eta_ohm = i_A_cm2 * R_tot                           # V
        E_mem = self.membrane_potential(c_dil_mol_m3, c_con_mol_m3)
        return eta_ohm + E_mem

    # ------------------------------------------------------------------
    # 1D plug-flow simulation along the stack (spatial ODE)
    # ------------------------------------------------------------------
    def simulate(self, current_density_A_m2, feed_conc_mol_m3,
                 flow_velocity_cm_s, stack_length_cm,
                 conc_feed_mol_m3=None, recovery_ratio=1.0,
                 limiting_fraction=0.8, n_eval=120):
        """
        Integrate salt mass balance along the diluate flow path.

        Parameters
        ----------
        current_density_A_m2 : float
            Applied current density [A/m2] (will be capped below i_lim).
        feed_conc_mol_m3 : float
            Diluate inlet salt concentration [mol/m3].
        flow_velocity_cm_s : float
            Linear velocity in the diluate channel [cm/s].
        stack_length_cm : float
            Flow-path length of the stack [cm].
        conc_feed_mol_m3 : float, optional
            Concentrate inlet concentration (default = feed).
        recovery_ratio : float
            Q_diluate / Q_concentrate ratio (default 1.0, symmetric flow).
        limiting_fraction : float
            Fraction of i_lim allowed locally.
        n_eval : int
            Number of spatial output points.

        Returns
        -------
        dict with spatial arrays and lumped performance metrics.
        """
        c0 = float(feed_conc_mol_m3)
        cc0 = float(conc_feed_mol_m3) if conc_feed_mol_m3 is not None else c0

        # --- Geometry in SI (the mass balance is integrated in SI units) ---
        # A = width * length  ->  width = A / length.  Convert cm -> m.
        L_m = stack_length_cm * 1e-2
        A_m2 = self.A_cm2 * 1e-4                            # membrane area per pair, m2
        h_m = self.h_cm * 1e-2                              # channel thickness, m
        width_m = A_m2 / L_m                                # m
        x_area_m2 = width_m * h_m                           # flow cross-section, m2
        v_m_s = flow_velocity_cm_s * 1e-2                   # m/s
        Q_d = v_m_s * x_area_m2                             # m3/s, diluate per cell pair
        Q_c = Q_d / max(recovery_ratio, 1e-6)              # m3/s, concentrate per cell pair

        zF = self.z * F

        def rhs(x, y):
            # x is position in metres; c in mol/m3
            c_d, c_c = y
            c_d = max(c_d, 1e-6)
            c_c = max(c_c, 1e-6)
            # local current density [A/cm2], polarization-capped on diluate side
            i_cm2 = float(self.operating_current_density(
                current_density_A_m2, c_d, limiting_fraction))
            i_m2 = i_cm2 * 1e4                              # A/cm2 -> A/m2
            # salt molar flux per membrane area: xi*i/zF  [mol/(s.m2)]
            # removal per unit flow-length = flux * width [m] -> mol/(s.m)
            flux_lin = self.xi * i_m2 / zF * width_m        # mol/(s.m)
            dcd = -flux_lin / Q_d                           # mol/m3 per m
            dcc = +flux_lin / Q_c
            return [dcd, dcc]

        x_eval = np.linspace(0.0, L_m, n_eval)
        sol = solve_ivp(rhs, (0.0, L_m), [c0, cc0],
                        t_eval=x_eval, method="RK45",
                        rtol=1e-8, atol=1e-10, max_step=L_m / 50.0)

        x = sol.t * 1e2                                     # m -> cm for reporting
        c_dil = np.maximum(sol.y[0], 1e-6)
        c_con = np.maximum(sol.y[1], 1e-6)

        # Local current density and voltage profiles
        N = len(x)
        i_local = np.zeros(N)
        i_lim = np.zeros(N)
        U_pair = np.zeros(N)
        for k in range(N):
            i_local[k] = float(self.operating_current_density(
                current_density_A_m2, c_dil[k], limiting_fraction))
            i_lim[k] = float(self.limiting_current_density(c_dil[k]))
            U_pair[k] = float(self.cell_pair_voltage(i_local[k], c_dil[k], c_con[k]))

        U_stack = self.N * U_pair                          # V (full stack)

        # --- Lumped performance metrics ---
        # Diluate volumetric production rate for full stack [m3/s] (Q_d already SI)
        Q_d_stack_m3s = Q_d * self.N                       # m3/s

        # Average current density and power along the path.
        # i in A/cm2, area per pair in cm2 -> I per pair = i * A.
        # U_stack = N * U_pair (full stack); the SAME current flows through the
        # series electrical path, so stack power = U_stack * I_pair.
        I_pair = i_local * self.A_cm2                      # A per cell pair
        P_elec = U_stack * I_pair                          # W (whole stack)
        # Average electrical power over the flow path:
        P_avg_W = float(np.trapz(P_elec, x) / (x[-1] - x[0])) if x[-1] > x[0] else float(P_elec[0])

        # SEC = energy per m3 of diluate produced.
        # Energy rate [W] / production rate [m3/s] = J/m3 -> /3.6e6 = kWh/m3
        if Q_d_stack_m3s > 0:
            SEC_kWh_m3 = P_avg_W / Q_d_stack_m3s / 3.6e6
        else:
            SEC_kWh_m3 = float("nan")

        salt_removed_frac = float((c0 - c_dil[-1]) / c0) if c0 > 0 else 0.0
        polarization_margin = float(np.min(i_lim - i_local))   # >0 means below limit

        return {
            "x": x,                                  # cm, position along stack
            "c_diluate": c_dil,                      # mol/m3
            "c_concentrate": c_con,                  # mol/m3
            "current_density_local": i_local,        # A/cm2
            "limiting_current_density": i_lim,       # A/cm2
            "cell_pair_voltage": U_pair,             # V
            "stack_voltage": U_stack,                # V
            "SEC_kWh_m3": SEC_kWh_m3,                # kWh per m3 diluate
            "current_efficiency": self.xi,           # -
            "salt_removed_fraction": salt_removed_frac,  # -
            "product_concentration_mol_m3": float(c_dil[-1]),
            "diluate_flow_m3_s": Q_d_stack_m3s,      # m3/s (whole stack)
            "polarization_margin_A_cm2": polarization_margin,
            "below_limiting_current": bool(polarization_margin > -1e-9),
        }
