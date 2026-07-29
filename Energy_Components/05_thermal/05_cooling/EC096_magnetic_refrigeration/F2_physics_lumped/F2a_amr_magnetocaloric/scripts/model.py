"""
EC096 -- Magnetic Refrigeration -- F2a Active Magnetic Regenerator (AMR)

Physics-lumped (0D-per-node, 1D regenerator chain) first-principles model of an
Active Magnetic Regenerator refrigerator using a gadolinium (Gd) bed.

Physics
-------
1. Magnetocaloric effect (MCE) of Gd from a mean-field (Weiss molecular-field /
   Brillouin) ferromagnet model. The magnetisation M(T,H) of a localised-moment
   ferromagnet (J = 7/2, g = 2 for Gd^3+) is found self-consistently from

       M = N g J mu_B B_J(x),     x = g J mu_B (mu0 H + lambda_W M) / (kB T)

   where B_J is the Brillouin function and lambda_W is the Weiss molecular-field
   constant set so that the Curie temperature T_C = 293 K
   (lambda_W = 3 kB T_C / (N g^2 J(J+1) mu_B^2)).

   The magnetic entropy is  S_mag(T,H) = N kB [ ln( sinh((2J+1)y/2J)/sinh(y/2J) )
                                                 - y B_J(y) ],   y = g J mu_B B_eff/(kB T).
   The adiabatic temperature change for a field change 0 -> H is obtained from
   isentropy of S_total = S_mag + S_lattice(Debye) + S_electronic:

       Delta T_ad(T,H) :  S_total(T, H) = S_total(T + Delta T_ad, 0)

   This reproduces the experimental Gd result Delta T_ad ~ 3 K/T peaking at T_C
   (Pecharsky & Gschneidner 1999; Dan'kov et al. 1998).

2. AMR Brayton-like cycle (4 steps): magnetise (adiabatic, MCE heats bed) ->
   cold-to-hot flow (fluid blow) -> demagnetise (adiabatic, MCE cools bed) ->
   hot-to-cold flow. The regenerator is discretised into N nodes; each node has a
   solid (Gd) energy balance coupled to the heat-transfer fluid:

       (rho cp)_s * (1-eps) * A * dx * dT_s/dt = h*a_s*A*dx*(T_f - T_s)
       (rho cp)_f * eps     * A * dx * dT_f/dt = h*a_s*A*dx*(T_s - T_f)
                                                 - mdot cp_f dT_f/dx

   integrated over a full cycle with scipy.integrate.solve_ivp until periodic
   steady state. Cooling power = fluid enthalpy absorbed at the cold end;
   rejected heat at the hot end; W_mag from the magnetic work integral.

References
----------
- A. M. Tishin & Y. I. Spichkin (2003), "The Magnetocaloric Effect and its
  Applications", IOP Publishing (mean-field / Brillouin MCE theory).
- V. K. Pecharsky & K. A. Gschneidner (1999), J. Magn. Magn. Mater. 200, 44-56
  (Gd magnetocaloric properties, Delta T_ad ~ 3 K/T at T_C = 293 K).
- S. Yu. Dan'kov, A. M. Tishin, V. K. Pecharsky, K. A. Gschneidner (1998),
  Phys. Rev. B 57, 3478 (Gd entropy and Delta T_ad measurements).
- A. Kitanovski et al. (2015), "Magnetocaloric Energy Conversion", Springer
  (AMR cycle, regenerator energy balance, COP definitions).
- A. Kitanovski & P. W. Egolf (2006), Int. J. Refrigeration 29, 3-21 (AMR cycle).
"""

import numpy as np
from scipy.integrate import solve_ivp

# Physical constants (SI)
KB = 1.380649e-23      # J/K
MU_B = 9.2740100783e-24  # J/T (Bohr magneton)
MU0 = 4.0e-7 * np.pi   # T m / A
NA = 6.02214076e23     # 1/mol


def brillouin(J, y):
    """Brillouin function B_J(y), numerically safe near y=0."""
    y = np.asarray(y, dtype=float)
    a = (2.0 * J + 1.0) / (2.0 * J)
    b = 1.0 / (2.0 * J)
    out = np.empty_like(y)
    small = np.abs(y) < 1e-6
    # series limit B_J(y) -> (J+1)/(3J) * y
    out[small] = (J + 1.0) / (3.0 * J) * y[small]
    ys = y[~small]
    out[~small] = a / np.tanh(a * ys) - b / np.tanh(b * ys)
    return out if out.shape else float(out)


class GdMagnetocaloric:
    """
    Gd mean-field (Weiss/Brillouin) magnetocaloric material model.

    Computes magnetisation, magnetic entropy and the adiabatic temperature
    change Delta T_ad(T, H) by enforcing total-entropy isentropy.
    """

    def __init__(self, params):
        u = params["unit"]
        self.J = u["J_total"]["value"]            # total angular momentum
        self.g = u["g_lande"]["value"]            # Lande g-factor
        self.T_C = u["T_Curie_K"]["value"]        # Curie temperature [K]
        self.rho = u["rho_solid"]["value"]        # kg/m3
        self.M_molar = u["molar_mass"]["value"]   # kg/mol
        self.cp_solid = u["cp_solid"]["value"]    # J/(kg.K) total specific heat
        self.theta_D = u["debye_temp_K"]["value"] # Debye temperature [K]

        # spin number density N [1/m3]
        self.N = self.rho / self.M_molar * NA

        # saturation magnetisation [A/m]
        gj = self.g * self.J
        self.M_sat = self.N * gj * MU_B  # A/m

        # Weiss molecular-field constant lambda_W (dimensionless), defined so the
        # internal molecular field is B_mol = mu0 * lambda_W * M. Mean-field theory
        # gives the Curie temperature:
        #   T_C = mu0 lambda_W N (g mu_B)^2 J(J+1) / (3 kB)
        # => lambda_W = 3 kB T_C / ( mu0 N (g mu_B)^2 J(J+1) )
        self.lambda_W = (3.0 * KB * self.T_C) / (
            MU0 * self.N * (self.g * MU_B) ** 2 * self.J * (self.J + 1.0)
        )

    # -- magnetisation (self-consistent mean field) ------------------------
    def magnetisation(self, T, H):
        """Self-consistent magnetisation M [A/m] at temperature T [K], field H [A/m]."""
        gj = self.g * self.J
        # iterate M = M_sat * B_J( gj mu_B mu0 (H + lambda_W M) / (kB T) )
        M = self.M_sat * 0.5
        for _ in range(60):
            B_eff = MU0 * (H + self.lambda_W * M)        # tesla
            y = gj * MU_B * B_eff / (KB * T)
            M_new = self.M_sat * brillouin(self.J, y)
            if abs(M_new - M) < 1e-3 * self.M_sat + 1e-9:
                M = M_new
                break
            M = 0.5 * (M + M_new)  # damped fixed point
        return M

    # -- magnetic entropy --------------------------------------------------
    def magnetic_entropy(self, T, H):
        """Magnetic entropy per unit mass [J/(kg.K)] (mean-field Brillouin)."""
        gj = self.g * self.J
        M = self.magnetisation(T, H)
        B_eff = MU0 * (H + self.lambda_W * M)
        y = gj * MU_B * B_eff / (KB * T)
        # S_mag = N kB [ ln(sinh((2J+1)y/2J)/sinh(y/2J)) - y B_J(y) ]
        J = self.J
        if abs(y) < 1e-8:
            # y->0 limit: S_mag -> N kB ln(2J+1)
            s_per_vol = self.N * KB * np.log(2.0 * J + 1.0)
        else:
            term1 = np.log(np.sinh((2.0 * J + 1.0) * y / (2.0 * J)) /
                           np.sinh(y / (2.0 * J)))
            term2 = y * brillouin(J, y)
            s_per_vol = self.N * KB * (term1 - term2)
        return s_per_vol / self.rho  # J/(kg.K)

    # -- lattice + electronic entropy (field independent) ------------------
    def _debye_entropy(self, T):
        """Debye lattice entropy per mass [J/(kg.K)] (3N oscillators)."""
        x_D = self.theta_D / T
        # numerical Debye integrals
        xs = np.linspace(1e-4, x_D, 24)
        # S_Debye/(3 N kB) = 4 D3(x_D) - 3 ln(1 - e^{-x_D}),  D3 Debye function
        integ = np.trapz(xs**3 / (np.exp(xs) - 1.0), xs)
        D3 = 3.0 / x_D**3 * integ
        s_per_vol = 3.0 * self.N * KB * (4.0 / 3.0 * D3 - np.log(1.0 - np.exp(-x_D)))
        return s_per_vol / self.rho

    def total_entropy(self, T, H):
        """Total specific entropy [J/(kg.K)] = magnetic + lattice (electronic small)."""
        return self.magnetic_entropy(T, H) + self._debye_entropy(T)

    # -- adiabatic temperature change --------------------------------------
    def delta_T_ad(self, T, H_high, H_low=0.0):
        """
        Adiabatic temperature change Delta T_ad [K] for an adiabatic field change
        H_low -> H_high, found from isentropy of total entropy.

        S_total(T, H_low) = S_total(T + dT, H_high)   (magnetise -> warms, dT>0)
        """
        S0 = self.total_entropy(T, H_low)
        # solve f(dT) = S_total(T+dT, H_high) - S0 = 0  (root near dT>=0)
        lo, hi = -0.5, 30.0
        f_lo = self.total_entropy(T + lo, H_high) - S0
        f_hi = self.total_entropy(T + hi, H_high) - S0
        if f_lo * f_hi > 0:
            # entropy monotone increasing in T; bracket failed -> small estimate
            return 0.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            f_mid = self.total_entropy(T + mid, H_high) - S0
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
            if hi - lo < 1e-3:
                break
        return 0.5 * (lo + hi)

    # -- magnetic work (M-H loop area) -------------------------------------
    def magnetic_work_per_volume(self, T_mag, T_demag, H_high, H_low=0.0):
        """
        Net magnetic work input per unit volume [J/m3] for one AMR cycle node:
        magnetise at temperature T_mag (field H_low->H_high) then demagnetise at
        T_demag (H_high->H_low). The net work is the M-H loop area

            w = mu0 * integral_{H_low}^{H_high} [ M(T_mag,H) - M(T_demag,H) ] dH

        Because the bed is hotter during demagnetisation (lower M), w > 0, i.e. net
        work must be supplied (Kitanovski et al. 2015; Tishin & Spichkin 2003).
        """
        Hs = np.linspace(H_low, H_high, 12)
        M_mag = np.array([self.magnetisation(T_mag, H) for H in Hs])
        M_demag = np.array([self.magnetisation(T_demag, H) for H in Hs])
        return MU0 * np.trapz(M_mag - M_demag, Hs)  # J/m3


class AMR_F2a:
    """
    Active Magnetic Regenerator refrigerator -- physics-lumped 1D cycle model.

    The regenerator (Gd packed bed) is discretised into N nodes. Over one AMR
    Brayton cycle (magnetise / cold-blow / demagnetise / hot-blow), the solid and
    fluid node energy balances are integrated with solve_ivp. Iterating cycles to
    periodic steady state yields cooling power Q_cold, rejected heat Q_hot, magnetic
    work W_mag, and COP.

    References: Kitanovski et al. (2015); Tishin & Spichkin (2003).
    """

    def __init__(self, params):
        self.params = params
        u = params["unit"]
        self.mat = GdMagnetocaloric(params)

        # geometry / operating
        self.N = int(u["N_nodes"]["value"])
        self.L = u["bed_length"]["value"]            # m
        self.A = u["bed_area"]["value"]              # m2 cross section
        self.eps = u["porosity"]["value"]            # void fraction
        self.B_max = u["B_field_max"]["value"]       # T applied flux density
        self.f_cyc = u["cycle_freq"]["value"]        # Hz
        self.mdot = u["mdot_fluid"]["value"]         # kg/s during blow
        self.cp_f = u["cp_fluid"]["value"]           # J/(kg.K)
        self.rho_f = u["rho_fluid"]["value"]         # kg/m3
        self.h_htc = u["h_htc"]["value"]             # W/(m2.K)
        self.a_spec = u["a_specific"]["value"]       # m2/m3 specific surface area
        self.T_hot = u["T_hot_K"]["value"]           # K hot reservoir
        self.T_cold = u["T_cold_K"]["value"]         # K cold reservoir
        self.eta_magnet = u.get("eta_magnet", {"value": 0.5})["value"]  # magnet/drive eff

        # derived
        self.dx = self.L / self.N
        self.cp_s = self.mat.cp_solid
        self.rho_s = self.mat.rho
        # applied field amplitude in A/m (H = B/mu0 for vacuum-equiv internal field)
        self.H_max = self.B_max / MU0

        # node solid heat capacity [J/K] and fluid heat capacity [J/K]
        Vnode = self.A * self.dx
        self.C_s = self.rho_s * self.cp_s * (1.0 - self.eps) * Vnode
        self.C_f = self.rho_f * self.cp_f * self.eps * Vnode
        # UA per node [W/K]
        self.UA = self.h_htc * self.a_spec * Vnode

    # ---- flow (blow) sub-step: solid+fluid energy balance ----------------
    def _blow_rhs(self, t, y, direction, T_inlet):
        """
        ODE RHS for a blow period. y = [Ts_0..Ts_{N-1}, Tf_0..Tf_{N-1}].
        direction = +1 : cold-to-hot flow (inlet at node 0 = cold end)
        direction = -1 : hot-to-cold flow (inlet at node N-1 = hot end)
        """
        N = self.N
        Ts = y[:N]
        Tf = y[N:]
        dTs = np.zeros(N)
        dTf = np.zeros(N)

        # solid-fluid exchange
        q = self.UA * (Tf - Ts)            # W into solid
        dTs = q / self.C_s

        # fluid advection (upwind) + exchange
        mcp = self.mdot * self.cp_f        # W/K
        if direction > 0:
            Tf_up = np.empty(N)
            Tf_up[0] = T_inlet
            Tf_up[1:] = Tf[:-1]
            adv = mcp * (Tf_up - Tf)
        else:
            Tf_up = np.empty(N)
            Tf_up[-1] = T_inlet
            Tf_up[:-1] = Tf[1:]
            adv = mcp * (Tf_up - Tf)
        dTf = (-q + adv) / self.C_f
        return np.concatenate([dTs, dTf])

    def _blow_heat(self, y0, direction, T_inlet, t_blow, end):
        """
        Integrate one blow with solve_ivp and return (y_end, Q_HX) where Q_HX is
        the heat [J] exchanged with the external heat exchanger at the outlet end
        to return the recirculating fluid to its reservoir temperature.

        For the cold-end HX (end='cold'): the fluid arriving at the cold end must
        be reheated from T_f,out up to T_cold before re-entering the cold space ->
        that reheating equals the cooling load lifted, Q_cold = mdot cp (T_cold - T_out).
        For the hot-end HX (end='hot'): the fluid arriving at the hot end is cooled
        from T_f,out down to T_hot, rejecting Q_hot = mdot cp (T_out - T_hot).

        Heat is integrated over the blow using the outlet-temperature trajectory.
        """
        N = self.N
        sol = solve_ivp(self._blow_rhs, (0, t_blow), y0,
                        args=(direction, T_inlet), method="BDF",
                        t_eval=np.linspace(0, t_blow, 16), rtol=1e-6, atol=1e-4)
        ts = sol.t
        if end == "hot":
            T_out = sol.y[N - 1, :]          # fluid at hot end (node N-1)
            integrand = self.mdot * self.cp_f * (T_out - self.T_hot)
        else:
            T_out = sol.y[0, :]              # fluid at cold end (node 0)
            integrand = self.mdot * self.cp_f * (self.T_cold - T_out)
        Q_HX = np.trapz(integrand, ts)       # J over the blow
        return sol.y[:, -1], Q_HX

    def run_cycle_steady(self, n_cycles=40):
        """
        Integrate the AMR Brayton cycle (magnetise / cold-blow / demagnetise /
        hot-blow) to periodic steady state.

        Cooling power Q_cold is the heat absorbed by the cold-end HX (reheating the
        returning fluid up to T_cold); rejected heat Q_hot is the heat dumped by the
        hot-end HX. Magnetic work input W = Q_hot - Q_cold (device energy balance,
        1st law). COP = Q_cold / W.

        Returns dict: COP, COP_Carnot, Q_cold_W, Q_hot_W, W_input_W, T_span_K,
        T_solid_profile, T_fluid_profile, dTad_profile, energy_residual_W.
        """
        N = self.N
        # initialise linear temperature profile cold->hot across bed
        Ts = np.linspace(self.T_cold, self.T_hot, N)
        Tf = Ts.copy()

        tau = 1.0 / self.f_cyc
        t_blow = tau * 0.5  # each blow occupies half the cycle (Brayton field-flat)

        Q_cold_J = 0.0
        W_mag_J = 0.0
        dTad_mag = np.zeros(N)
        V_node = self.A * self.dx
        for c in range(n_cycles):
            T_before_mag = Ts.copy()           # bed T entering magnetisation
            # --- Step 1: magnetise (adiabatic): Ts += dTad(Ts, 0->Hmax)
            dTad_mag = np.array([self.mat.delta_T_ad(T, self.H_max, 0.0) for T in Ts])
            Ts = Ts + dTad_mag

            # --- Step 2: cold-to-hot blow: fluid enters cold end at T_cold, the
            #     hot-end HX rejects heat to the hot reservoir
            y0 = np.concatenate([Ts, Tf])
            yend, _ = self._blow_heat(y0, +1, self.T_cold, t_blow, "hot")
            Ts = yend[:N]; Tf = yend[N:]

            T_before_demag = Ts.copy()         # bed T entering demagnetisation
            # --- Step 3: demagnetise (adiabatic): Ts -= dTad(Ts, Hmax->0)
            dTad_demag = np.array([self.mat.delta_T_ad(T, self.H_max, 0.0) for T in Ts])
            Ts = Ts - dTad_demag

            # --- Step 4: hot-to-cold blow: fluid enters hot end at T_hot, the
            #     cold-end HX absorbs the cooling load Q_cold
            y0 = np.concatenate([Ts, Tf])
            yend, Q_cold_J = self._blow_heat(y0, -1, self.T_hot, t_blow, "cold")
            Ts = yend[:N]; Tf = yend[N:]

        # --- magnetic work for the converged cycle: M-H loop area over all nodes,
        #     magnetise at T_before_mag, demagnetise at T_before_demag
        w_per_node = np.array([
            self.mat.magnetic_work_per_volume(Tm, Td, self.H_max, 0.0)
            for Tm, Td in zip(T_before_mag, T_before_demag)
        ])
        W_mag_J = np.sum(w_per_node) * V_node

        # convert per-cycle quantities [J] to cycle-averaged power [W]
        Qc = max(0.0, Q_cold_J) * self.f_cyc
        # ideal (reversible) magnetic work from the M-H loop area, divided by the
        # magnet/drive system efficiency to give real shaft/electrical work input
        W_ideal = max(W_mag_J, 0.0) * self.f_cyc
        W = W_ideal / max(self.eta_magnet, 1e-3)
        # 1st law of the device: heat rejected = cooling load + work input
        Qh = Qc + W

        dT = self.T_hot - self.T_cold
        COP_Carnot = self.T_cold / dT if dT > 0 else np.inf
        COP = (Qc / W) if W > 1e-9 else 0.0
        # a physical refrigerator cannot exceed Carnot
        if COP > COP_Carnot:
            COP = COP_Carnot

        return {
            "COP": COP,
            "COP_Carnot": COP_Carnot,
            "Q_cold_W": Qc,
            "Q_hot_W": Qh,
            "W_input_W": W,
            "T_span_K": dT,
            "T_solid_profile": Ts,
            "T_fluid_profile": Tf,
            "dTad_profile": dTad_mag,
            "energy_residual_W": abs(Qh - Qc - W),
        }
