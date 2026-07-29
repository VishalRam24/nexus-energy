"""
EC142 -- Biogas Upgrading to Biomethane -- F2a Physics-Lumped Water Scrubbing

High-pressure water scrubbing (HPWS): CO2 is separated from raw biogas
(CH4 + CO2) by physical absorption into pressurised cold water, exploiting
the ~25x higher Henry's-law solubility of CO2 vs CH4. The small amount of
co-absorbed CH4 leaves with the stripped CO2 stream and constitutes the
"methane slip".

Physics-lumped (0D) model
-------------------------
The counter-current packed absorption column is lumped to a single transfer-
unit description. Each species i in {CO2, CH4} is characterised by:

  - Henry's-law solubility  H_cp_i(T)  [mol/(m3.Pa)] with van't Hoff T
    correction (cold water dissolves more CO2; Sander 2015).
  - Number of transfer units  NTU_i = kLa_i * V_L / L_water   (column height
    / efficiency measure; kLa_CH4 = selectivity * kLa_CO2).
  - Absorption factor  A_i = H_cp_i * P_col * L_water / G   (G = gas molar
    throughput per liquid volume), the solvent capacity for species i.

The steady-state gas-phase removal fraction follows the Colburn/Kremser
counter-current relation:

    E_i = (1 - exp(-NTU*(1-1/A))) / (1 - (1/A)*exp(-NTU*(1-1/A)))

CO2 (A >> 1, high NTU) is almost fully stripped (E -> ~1); CH4 (A << 1,
poorly soluble) has E -> small, which IS the methane-slip mechanism.

Transient (the lumped ODE): each species removal fraction r_i(t) relaxes
first-order toward its steady E_i with time constant tau = V_L/L_water (the
liquid residence time), integrated by scipy.solve_ivp:

    dr_i/dt = (E_i - r_i) / tau

Gas-side species mass balance (per unit time):
    n_i,out_gas = n_i,in_gas - N_i_absorbed,   N_i_absorbed = r_i * n_i,in
Conservation: in = out_gas + absorbed holds exactly for each species.

Outputs: product (raffinate) gas composition, CH4 purity, CH4 recovery,
methane slip, CO2 removal, and specific energy demand (pump compression of
the recirculating water against column pressure).

References
----------
Bauer, F. et al. (2013). Biogas upgrading -- Review of commercial technologies.
    Bioresour. Technol. 122, 145-159.
Sun, Q. et al. (2015). Selection of appropriate biogas upgrading technology.
    Renew. Sustain. Energy Rev. 51, 521-532.
Sander, R. (2015). Compilation of Henry's law constants. Atmos. Chem. Phys. 15, 4399.
Cussler, E.L. (2009). Diffusion: Mass Transfer in Fluid Systems, 3rd ed., CUP.
"""

import numpy as np
from scipy.integrate import solve_ivp


class BiogasUpgradingF2a:
    """High-pressure water scrubbing -- lumped transient absorption column."""

    R = 8.314           # J/(mol.K)  universal gas constant
    T_std = 273.15      # K          standard temperature for Nm3
    P_std = 101325.0    # Pa         standard pressure for Nm3
    T_ref_H = 298.15    # K          reference T for Henry constants

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_L = float(u["V_liquid"]["value"])             # m3
        self.L_water = float(u["L_water"]["value"])          # m3/s
        self.P_col = float(u["P_col"]["value"]) * 1e5        # Pa
        self.T_col = float(u["T_col"]["value"])              # K
        self.kLa_CO2 = float(u["kLa_CO2"]["value"])          # 1/s
        self.H_cp_CO2_ref = float(u["H_cp_CO2"]["value"])    # mol/(m3.Pa)
        self.H_cp_CH4_ref = float(u["H_cp_CH4"]["value"])    # mol/(m3.Pa)
        self.dlnH_CO2 = float(u["dlnH_CO2"]["value"])        # K
        self.dlnH_CH4 = float(u["dlnH_CH4"]["value"])        # K
        self.sel = float(u["selectivity_CH4_CO2"]["value"])  # -
        self.eta_pump = float(u["eta_pump"]["value"])        # -

    # ------------------------------------------------------------------
    # Henry's-law solubility with van't Hoff temperature correction
    # ------------------------------------------------------------------
    def henry_cp(self, species, T):
        """Henry solubility H^cp [mol/(m3.Pa)] at temperature T.

        van't Hoff: H(T) = H_ref * exp( dlnH * (1/T - 1/T_ref) ).
        Lower T -> larger H -> more dissolved gas (cold water absorbs more).
        """
        if species == "CO2":
            return self.H_cp_CO2_ref * np.exp(self.dlnH_CO2 * (1.0 / T - 1.0 / self.T_ref_H))
        elif species == "CH4":
            return self.H_cp_CH4_ref * np.exp(self.dlnH_CH4 * (1.0 / T - 1.0 / self.T_ref_H))
        raise ValueError(species)

    def kLa(self, species):
        """Volumetric mass-transfer coefficient [1/s] per species."""
        if species == "CO2":
            return self.kLa_CO2
        return self.sel * self.kLa_CO2     # CH4 absorbs slower -> selectivity

    # ------------------------------------------------------------------
    # Molar flow helper
    # ------------------------------------------------------------------
    def nm3h_to_mol_s(self, Q_Nm3_h):
        """Convert a volumetric flow [Nm3/h] to molar flow [mol/s] (ideal gas)."""
        n_per_Nm3 = self.P_std / (self.R * self.T_std)     # mol/m3 at STP
        return Q_Nm3_h / 3600.0 * n_per_Nm3

    def mol_s_to_nm3h(self, n_mol_s):
        n_per_Nm3 = self.P_std / (self.R * self.T_std)
        return n_mol_s * 3600.0 / n_per_Nm3

    # ------------------------------------------------------------------
    # Counter-current column removal (NTU / transfer-unit model)
    # ------------------------------------------------------------------
    def NTU(self, species):
        """Number of liquid-phase transfer units for the packed column.

        NTU_i = kLa_i * V_L / L_water  (dimensionless). This is the lumped
        measure of column performance: a tall/efficient column has many
        transfer units. kLa already embeds the species selectivity, so
        CH4 (low kLa) has a much smaller NTU than CO2 -> low methane slip.
        """
        return self.kLa(species) * self.V_L / self.L_water

    def removal_efficiency(self, species, T, x_i):
        """Steady-state gas-phase removal fraction for a counter-current
        absorber, from the Colburn/Kremser relation.

        E_i = (1 - exp(-NTU*(1-A_inv))) / (1 - A_inv*exp(-NTU*(1-A_inv)))

        where the absorption factor A = L/(H*P*G) measures solvent capacity
        relative to the gas load (Treybal; Sun 2015). For a strong solvent
        (A >> 1, CO2) E -> 1; for a poorly-soluble species (A << 1, CH4)
        E -> small, which is exactly the methane-slip mechanism. Returned
        value is clipped to (0,1).
        """
        N = self.NTU(species)
        H = self.henry_cp(species, T)               # mol/(m3.Pa)
        # gas molar flux scale per unit column volume basis cancels in A:
        # absorption factor A = (L_water * H * P_col) / (gas molar throughput
        # per transfer-unit basis). Use A = L_water*H*P_col / G_ref with the
        # convective wash-out rate as reference; expressed via NTU it reduces
        # to A = H * P_col * V_L * kLa / (L_water * x-independent) -> use the
        # dimensionless solvent capacity below.
        A = H * self.P_col * self.L_water / max(self._G_mol_m3, 1e-30)
        if abs(A - 1.0) < 1e-6:
            E = N / (1.0 + N)
        else:
            num = 1.0 - np.exp(-N * (1.0 - 1.0 / A))
            den = 1.0 - (1.0 / A) * np.exp(-N * (1.0 - 1.0 / A))
            E = num / den if den != 0 else 1.0
        return float(np.clip(E, 0.0, 0.999999))

    def _rhs(self, t, y, E_CO2, E_CH4):
        """Transient lumped ODE for the column removal state.

        Each species' instantaneous removal fraction r_i relaxes (first-order,
        time constant tau = V_L/L_water = liquid residence time) toward its
        steady counter-current value E_i. This captures the column fill/
        wash-out transient while the steady state reproduces the NTU column
        performance.
        """
        r_CO2, r_CH4 = y
        tau = self.V_L / self.L_water
        return [(E_CO2 - r_CO2) / tau, (E_CH4 - r_CH4) / tau]

    def simulate(self, biogas_flow_Nm3_per_h, CH4_fraction_in=0.60,
                 T_col_K=None, dt=2.0, duration_s=300.0):
        """
        Simulate the absorption column transient to steady state.

        Parameters
        ----------
        biogas_flow_Nm3_per_h : float    raw biogas feed [Nm3/h]
        CH4_fraction_in       : float    CH4 mole fraction of raw biogas [-]
        T_col_K               : float    column temperature [K] (default param)
        dt, duration_s        : float    output step and horizon [s]

        Returns
        -------
        dict of time-series arrays + steady-state scalar performance.
        """
        T = self.T_col if T_col_K is None else float(T_col_K)
        Q = max(float(biogas_flow_Nm3_per_h), 1e-9)
        x_CH4 = float(np.clip(CH4_fraction_in, 1e-6, 1.0 - 1e-6))
        x_CO2 = 1.0 - x_CH4

        # Gas-side molar feed rates [mol/s]
        n_in_total = self.nm3h_to_mol_s(Q)
        n_in_CO2 = n_in_total * x_CO2
        n_in_CH4 = n_in_total * x_CH4

        # Gas molar throughput per column liquid volume [mol/(m3.s)] -> the
        # reference scale that sets the absorption factor A = H*P*L / G.
        self._G_mol_m3 = n_in_total / self.V_L

        # Steady-state counter-current removal efficiencies per species.
        E_CO2 = self.removal_efficiency("CO2", T, x_CO2)
        E_CH4 = self.removal_efficiency("CH4", T, x_CH4)

        t_eval = np.arange(0.0, duration_s + dt * 0.5, dt)
        t_eval = t_eval[t_eval <= duration_s]

        sol = solve_ivp(
            self._rhs, (0.0, duration_s), [0.0, 0.0],
            t_eval=t_eval, args=(E_CO2, E_CH4),
            method="RK45", rtol=1e-8, atol=1e-12, max_step=dt,
        )

        t_out = sol.t
        r_CO2 = np.clip(sol.y[0], 0.0, 1.0)     # instantaneous CO2 removal fraction
        r_CH4 = np.clip(sol.y[1], 0.0, 1.0)     # instantaneous CH4 removal (slip) fraction
        N = len(t_out)

        # Dissolved liquid concentrations [mol/m3] implied by the removal rate
        # (absorbed flux / water flow). Reported as diagnostic state.
        C_CO2 = (r_CO2 * n_in_CO2) / self.L_water
        C_CH4 = (r_CH4 * n_in_CH4) / self.L_water

        # Absorbed molar rates [mol/s]
        Nabs_CO2 = r_CO2 * n_in_CO2
        Nabs_CH4 = r_CH4 * n_in_CH4

        # Product (raffinate) gas leaving the top of the column [mol/s]
        n_out_CO2 = n_in_CO2 - Nabs_CO2
        n_out_CH4 = n_in_CH4 - Nabs_CH4
        n_out_total = n_out_CO2 + n_out_CH4

        purity_CH4 = np.where(n_out_total > 0, n_out_CH4 / n_out_total, 0.0)
        CH4_recovery = n_out_CH4 / n_in_CH4
        CH4_slip = Nabs_CH4 / n_in_CH4               # fraction of feed CH4 lost
        CO2_removal = Nabs_CO2 / n_in_CO2

        biomethane_Nm3_h = self.mol_s_to_nm3h(n_out_total)

        # Specific energy demand: pump work to pressurise recirculating water
        # against the column pressure. W_pump = L_water * dP / eta  [W].
        dP = self.P_col - self.P_std
        W_pump_W = self.L_water * dP / self.eta_pump
        # specific energy per Nm3 of product biomethane (steady-state value)
        bm_ss = biomethane_Nm3_h[-1] if N > 0 and biomethane_Nm3_h[-1] > 0 else np.nan
        SEC_kWh_per_Nm3 = (W_pump_W / 1000.0) / max(bm_ss, 1e-9)

        return {
            "t": t_out,
            "C_CO2_liquid": C_CO2,
            "C_CH4_liquid": C_CH4,
            "purity_CH4": purity_CH4,
            "CH4_recovery": CH4_recovery,
            "CH4_slip": CH4_slip,
            "CO2_removal": CO2_removal,
            "biomethane_Nm3_per_h": biomethane_Nm3_h,
            "n_in_CO2": n_in_CO2,
            "n_in_CH4": n_in_CH4,
            "n_out_CO2": n_out_CO2,
            "n_out_CH4": n_out_CH4,
            "Nabs_CO2": Nabs_CO2,
            "Nabs_CH4": Nabs_CH4,
            "W_pump_kW": W_pump_W / 1000.0,
            "SEC_kWh_per_Nm3": float(SEC_kWh_per_Nm3),
            "purity_CH4_ss": float(purity_CH4[-1]) if N else 0.0,
            "CH4_recovery_ss": float(CH4_recovery[-1]) if N else 0.0,
            "CH4_slip_ss": float(CH4_slip[-1]) if N else 0.0,
            "CO2_removal_ss": float(CO2_removal[-1]) if N else 0.0,
        }

    # ------------------------------------------------------------------
    # Mass-balance residual (for conservation checking)
    # ------------------------------------------------------------------
    def mass_balance_residual(self, result, idx=-1):
        """Per-species |in - (out_gas + absorbed)| relative residual at index."""
        in_CO2 = result["n_in_CO2"]
        in_CH4 = result["n_in_CH4"]
        res_CO2 = abs(in_CO2 - (result["n_out_CO2"][idx] + result["Nabs_CO2"][idx])) / in_CO2
        res_CH4 = abs(in_CH4 - (result["n_out_CH4"][idx] + result["Nabs_CH4"][idx])) / in_CH4
        return {"CO2": float(res_CO2), "CH4": float(res_CH4)}
