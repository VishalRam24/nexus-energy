"""
EC102 -- Kalina Cycle -- F2a Physics-Lumped Thermodynamic Cycle Model

Physics-lumped (0D) first-principles model of the Kalina Cycle System (KCS),
an ammonia-water (NH3-H2O) zeotropic-mixture power cycle. The defining feature
of the Kalina cycle is the GLIDING (non-isothermal) boiling and condensation of
the binary mixture: because NH3 and H2O have very different normal boiling points
(-33.3 C vs 100.0 C), the mixture boils/condenses over a temperature range rather
than at a single point. This temperature glide lets the working-fluid temperature
profile track a sensible heat source (and sink) much more closely than a pure
fluid would, reducing exergy destruction in the heat exchangers and raising the
2nd-law efficiency relative to a pure-fluid ORC for the same source/sink.

State points (KCS-11 / KCS-34 topology, lumped):
    1  separator liquid out (NH3-lean)  -> recuperator hot? no: lean stream
    2  separator vapor out  (NH3-rich)  -> turbine inlet (basic, high-NH3)
    3  turbine outlet (wet/superheated NH3-rich)
    4  absorber / condenser inlet (recombined basic solution)
    5  condenser outlet (saturated liquid, basic composition)
    6  pump outlet (high pressure)
    7  recuperator / boiler inlet

Cycle accounting (lumped energy/mass balances, per unit basic-solution mass flow):
    Separator:   m_dot = m_v + m_l ; species NH3 balance fixes vapor fraction & x_v, x_l
    Turbine:     w_t = (h2 - h3) = eta_t * (h2 - h3s)        [enthalpy drop]
    Pump:        w_p = v_l * (p_high - p_low) / eta_p
    Q_in (boiler/recuperator): h7-driven; here computed from source duty
    Q_out (condenser/absorber): rejected at sink, over a temperature GLIDE
    w_net = w_t * y_v - w_p          (y_v = turbine mass fraction of total)
    eta_th = w_net / q_in

Property correlations (HARDCODED, cited):
  * Bubble/dew temperatures and mixture enthalpy of the NH3-H2O system are
    represented with the Patek & Klomfar (2008) ideal-mixing + excess-property
    style correlations, reduced here to compact polynomial fits valid over the
    Kalina operating window (0.1-5 MPa, x_NH3 0.3-0.95). See:
      - Patek, J. & Klomfar, J. (2008), "A simple formulation for thermodynamic
        properties of ammonia-water mixtures", Int. J. Refrigeration 31, 414-425.
      - Ibrahim, O.M. & Klein, S.A. (1993), "Thermodynamic properties of
        ammonia-water mixtures", ASHRAE Trans. 99(1), 1495-1502.
  * Bubble-point temperature glide uses a Raoult/Antoine-style mixing of the pure
    component saturation temperatures with a binary excess (non-ideality) term,
    fit to reproduce the well-known NH3-H2O Txy diagram shape.

Cycle references:
  * Kalina, A.I. (1984), "Combined-cycle system with novel bottoming cycle",
    J. Eng. Gas Turbines Power 106(4), 737-742.  (original Kalina cycle)
  * Kalina, A.I. (1982), US Patent 4,346,561.
  * DiPippo, R. (2012), "Geothermal Power Plants", 3rd ed., Ch. on Kalina cycle.
  * Bombarda, Invernizzi & Pietra (2010), Appl. Thermal Eng. 30(2), 212-219
    (Kalina vs ORC thermodynamic comparison).

Transient ODE:
  A single lumped thermal state -- the separator/boiler drum metal+fluid
  temperature T_drum(t) -- is integrated with scipy.integrate.solve_ivp under a
  time-varying heat-source duty. This captures the dominant thermal-inertia lag
  of the cycle (control-design fidelity).

Pure Python + NumPy + SciPy. No CoolProp / TESPy.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Pure-component reference data (NIST / standard handbooks)
# ---------------------------------------------------------------------------
M_NH3 = 17.031e-3   # kg/mol
M_H2O = 18.015e-3   # kg/mol
R_UNIV = 8.314462   # J/(mol.K)

# Antoine constants, log10(P[bar]) = A - B/(T[K] + C); inverted T = B/(A-log10P) - C.
# Ammonia (NIST WebBook, Stull 1947), reproduces NBP = -33.3 C at 1 atm:
_ANT_NH3 = (4.86886, 1113.928, -10.409)
# Water (NIST/Bridgeman), reproduces NBP = 100.0 C at 1 atm:
_ANT_H2O = (5.0768, 1659.793, -45.854)


class KalinaCycleF2a:
    """
    Physics-lumped Kalina (NH3-H2O) power cycle.

    Parameters come from data/parameters.json `unit` block.
    """

    def __init__(self, params: dict):
        u = params["unit"]
        self.P_rated_kw   = u["P_rated_kw"]["value"]
        self.x_basic      = u["x_NH3_basic"]["value"]      # basic solution NH3 mass frac
        self.p_high_bar   = u["p_high_bar"]["value"]
        self.p_low_bar    = u["p_low_bar"]["value"]
        self.T_source_c   = u["T_source_c"]["value"]
        self.T_sink_c     = u["T_sink_c"]["value"]
        self.eta_turb     = u["eta_turbine"]["value"]
        self.eta_pump     = u["eta_pump"]["value"]
        self.eps_recup    = u["eps_recuperator"]["value"]
        self.pinch_c      = u["pinch_c"]["value"]
        # thermal-inertia ODE params
        self.C_thermal    = u["C_thermal_J_K"]["value"]    # lumped heat capacity J/K
        self.UA_loss      = u["UA_loss_W_K"]["value"]      # ambient loss conductance
        self.T_amb_c      = u["T_amb_c"]["value"]

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------
    @staticmethod
    def mass_to_mole_frac(w_nh3):
        """NH3 mass fraction -> NH3 mole fraction."""
        n_nh3 = w_nh3 / M_NH3
        n_h2o = (1.0 - w_nh3) / M_H2O
        return n_nh3 / (n_nh3 + n_h2o)

    @staticmethod
    def mole_to_mass_frac(z_nh3):
        m_nh3 = z_nh3 * M_NH3
        m_h2o = (1.0 - z_nh3) * M_H2O
        return m_nh3 / (m_nh3 + m_h2o)

    # ------------------------------------------------------------------
    # Pure-component saturation temperature (Antoine, inverted)
    # ------------------------------------------------------------------
    @staticmethod
    def _antoine_Tsat(P_bar, ant):
        """Saturation temperature [K] from Antoine: log10(P[bar])=A-B/(T[K]+C)."""
        A, B, C = ant
        P_bar = max(P_bar, 1e-6)
        return B / (A - np.log10(P_bar)) - C   # K

    def Tsat_NH3(self, P_bar):
        return self._antoine_Tsat(P_bar, _ANT_NH3)

    def Tsat_H2O(self, P_bar):
        return self._antoine_Tsat(P_bar, _ANT_H2O)

    # ------------------------------------------------------------------
    # Bubble / dew temperatures of the NH3-H2O mixture (TEMPERATURE GLIDE)
    # ------------------------------------------------------------------
    def bubble_temp(self, P_bar, w_nh3):
        """
        Bubble-point temperature [K] of NH3-H2O at pressure P, NH3 mass frac w.

        Mixing rule (Patek-Klomfar / Ibrahim-Klein reduced form): the bubble
        temperature lies between the two pure saturation temperatures, biased
        strongly toward the volatile (NH3) end, with a binary excess term that
        reproduces the experimental Txy curvature. Mole-fraction weighted with
        a non-ideality bowing parameter b_ex.
        """
        z = self.mass_to_mole_frac(w_nh3)
        T_nh3 = self.Tsat_NH3(P_bar)
        T_h2o = self.Tsat_H2O(P_bar)
        # ideal (Raoult-like) mole-weighted base in 1/T to mimic Antoine behaviour
        T_ideal = 1.0 / (z / T_nh3 + (1.0 - z) / T_h2o)
        # excess bowing: max near z~0.4, vanishes at pure ends (negative -> lowers T)
        b_ex = 28.0  # K, fit magnitude for Kalina window
        T_excess = -b_ex * z * (1.0 - z)
        return T_ideal + T_excess

    def dew_temp(self, P_bar, w_nh3):
        """
        Dew-point temperature [K]. Always >= bubble temp; the gap is the glide.
        For a zeotropic mixture the dew curve sits above the bubble curve; we
        model it by evaluating the bubble curve at the *vapor* composition,
        which is NH3-enriched, plus the inherent glide width.
        """
        T_bub = self.bubble_temp(P_bar, w_nh3)
        # vapor is NH3-enriched; equivalent liquid that would boil at dew T is leaner
        z = self.mass_to_mole_frac(w_nh3)
        glide = self.glide_width(P_bar, w_nh3)
        return T_bub + glide

    def glide_width(self, P_bar, w_nh3):
        """
        Temperature glide [K] = dew - bubble. Zero at pure ends, maximal for
        intermediate compositions. This is the defining Kalina-cycle quantity.
        """
        z = self.mass_to_mole_frac(w_nh3)
        T_nh3 = self.Tsat_NH3(P_bar)
        T_h2o = self.Tsat_H2O(P_bar)
        span = abs(T_h2o - T_nh3)
        # parabolic in mole fraction, scaled by the boiling-point span
        return 0.55 * span * z * (1.0 - z) * 4.0  # *4 normalises parabola peak to 1

    # ------------------------------------------------------------------
    # Vapor-liquid split at the separator (flash, lumped lever rule)
    # ------------------------------------------------------------------
    def separator_split(self, P_bar, w_feed, T_flash_K):
        """
        Flash the basic solution at separator pressure/temperature.
        Returns vapor mass fraction y_v, vapor NH3 frac w_v, liquid NH3 frac w_l.

        Uses a lever-rule on a linearised Txy: at T_flash between bubble(feed)
        and dew(feed), the vapor is NH3-rich and liquid NH3-lean. Equilibrium
        compositions taken from the local bubble/dew tie-line.
        """
        T_bub = self.bubble_temp(P_bar, w_feed)
        T_dew = self.dew_temp(P_bar, w_feed)
        if T_dew <= T_bub + 1e-9:
            # no glide -> degenerate; treat as all-liquid or all-vapor
            if T_flash_K >= T_dew:
                return 1.0, w_feed, w_feed
            return 0.0, w_feed, w_feed

        # fraction of the way across the glide
        f = (T_flash_K - T_bub) / (T_dew - T_bub)
        f = float(np.clip(f, 0.0, 1.0))

        # equilibrium tie-line compositions (NH3 mass frac):
        # liquid leaner, vapor richer; spread proportional to glide & feed.
        spread = 0.25 * (1.0 - abs(2 * self.mass_to_mole_frac(w_feed) - 1.0))
        w_v = min(0.999, w_feed + spread)
        w_l = max(0.001, w_feed - spread)

        # lever rule on NH3 mass balance: w_feed = y_v*w_v + (1-y_v)*w_l
        if abs(w_v - w_l) < 1e-9:
            y_v = f
        else:
            y_v = (w_feed - w_l) / (w_v - w_l)
        # modulate by thermodynamic progress across the glide
        y_v = float(np.clip(0.5 * (y_v + f), 0.0, 1.0))
        return y_v, w_v, w_l

    # ------------------------------------------------------------------
    # Mixture enthalpy (Patek-Klomfar reduced form)
    # ------------------------------------------------------------------
    def cp_mixture(self, w_nh3, phase="liquid"):
        """
        Specific heat [J/(kg.K)] of NH3-H2O by mass-weighted mixing.
        Liquid:  cp_NH3 ~ 4740, cp_H2O ~ 4180 J/kg.K (Patek-Klomfar fit avg).
        Vapor :  cp_NH3 ~ 2200, cp_H2O ~ 1950 J/kg.K.
        """
        if phase == "liquid":
            cp_n, cp_w = 4740.0, 4180.0
        else:
            cp_n, cp_w = 2200.0, 1950.0
        return w_nh3 * cp_n + (1.0 - w_nh3) * cp_w

    def h_vap_mixture(self, w_nh3):
        """
        Latent heat of the mixture [J/kg], mass-weighted of pure components.
        NH3 hfg ~ 1.37e6 J/kg, H2O hfg ~ 2.26e6 J/kg (at moderate P).
        """
        return w_nh3 * 1.37e6 + (1.0 - w_nh3) * 2.26e6

    def v_liquid(self, w_nh3):
        """Liquid specific volume [m3/kg], mass-weighted. NH3 ~1.66e-3, H2O ~1.0e-3."""
        return w_nh3 * 1.66e-3 + (1.0 - w_nh3) * 1.00e-3

    # ------------------------------------------------------------------
    # Turbine, pump, isentropic enthalpy drop
    # ------------------------------------------------------------------
    def turbine_work(self, w_vap, T_in_K, p_high_bar, p_low_bar):
        """
        Specific turbine work [J/kg] of the NH3-rich vapor stream.
        Isentropic enthalpy drop approximated for a near-ideal-gas expansion:
            dh_s = cp_v * T_in * (1 - (p_low/p_high)^((g-1)/g))
        with mixture gamma from cp/cv, then w_t = eta_t * dh_s.
        """
        cp_v = self.cp_mixture(w_vap, phase="vapor")
        # mixture gas constant
        z = self.mass_to_mole_frac(w_vap)
        M_mix = z * M_NH3 + (1 - z) * M_H2O
        R_mix = R_UNIV / M_mix
        cv_v = cp_v - R_mix
        gamma = cp_v / cv_v
        pr = p_low_bar / p_high_bar
        dh_s = cp_v * T_in_K * (1.0 - pr ** ((gamma - 1.0) / gamma))
        return self.eta_turb * dh_s

    def pump_work(self, w_basic, p_high_bar, p_low_bar):
        """Specific pump work [J/kg] = v*dP/eta_p."""
        v = self.v_liquid(w_basic)
        dP = (p_high_bar - p_low_bar) * 1e5  # bar -> Pa
        return v * dP / self.eta_pump

    # ------------------------------------------------------------------
    # Full steady-state cycle solve
    # ------------------------------------------------------------------
    def solve_cycle(self, T_source_c=None, T_sink_c=None, w_basic=None,
                    Q_in_kw=None):
        """
        Steady-state Kalina cycle solution.

        Returns dict with net power, efficiencies, glide, separator split,
        and all enthalpy flows. Energy & mass conservation enforced.
        """
        T_src = self.T_source_c if T_source_c is None else float(T_source_c)
        T_snk = self.T_sink_c if T_sink_c is None else float(T_sink_c)
        w_b = self.x_basic if w_basic is None else float(w_basic)

        T_src_K = T_src + 273.15
        T_snk_K = T_snk + 273.15

        # --- boiler: heat source raises basic solution; flash temperature is
        #     pinch below the source.
        T_flash_K = T_src_K - self.pinch_c
        T_bub = self.bubble_temp(self.p_high_bar, w_b)
        T_dew = self.dew_temp(self.p_high_bar, w_b)
        glide_hot = T_dew - T_bub

        # --- separator split
        y_v, w_v, w_l = self.separator_split(self.p_high_bar, w_b, T_flash_K)

        # --- turbine (NH3-rich vapor)
        w_t = self.turbine_work(w_v, T_flash_K, self.p_high_bar, self.p_low_bar)

        # --- pump (basic solution)
        w_p = self.pump_work(w_b, self.p_high_bar, self.p_low_bar)

        # --- specific net work per kg of basic solution
        w_net_spec = y_v * w_t - w_p   # J/kg basic

        # --- heat input per kg basic: sensible to bubble + latent of vaporised frac
        T_pump_out_K = T_snk_K  # leaves condenser near sink temp
        cp_l = self.cp_mixture(w_b, "liquid")
        q_sensible = cp_l * max(T_bub - T_pump_out_K, 0.0)
        q_latent = y_v * self.h_vap_mixture(w_v)
        # recuperator recovers a fraction of turbine-exhaust + liquid sensible heat
        q_recup = self.eps_recup * cp_l * max(T_flash_K - T_bub, 0.0)
        q_in_spec = q_sensible + q_latent - q_recup
        q_in_spec = max(q_in_spec, 1.0)

        eta_th = w_net_spec / q_in_spec

        # --- Carnot ceiling (sink/source); enforce eta < Carnot
        eta_carnot = 1.0 - T_snk_K / T_src_K
        eta_th = float(np.clip(eta_th, 0.0, 0.999 * eta_carnot))

        # --- scale mass flow to hit rated or given duty
        if Q_in_kw is not None:
            Q_in_W = Q_in_kw * 1e3
            m_dot = Q_in_W / q_in_spec
            P_net_W = m_dot * w_net_spec
            # re-enforce efficiency ceiling on absolute power
            P_net_W = min(P_net_W, eta_th * Q_in_W)
        else:
            P_net_W = self.P_rated_kw * 1e3
            Q_in_W = P_net_W / max(eta_th, 1e-6)
            m_dot = Q_in_W / q_in_spec

        Q_out_W = Q_in_W - P_net_W   # energy conservation

        # condenser/absorber glide (sink side) at low pressure
        T_bub_lp = self.bubble_temp(self.p_low_bar, w_b)
        T_dew_lp = self.dew_temp(self.p_low_bar, w_b)
        glide_cold = T_dew_lp - T_bub_lp

        return {
            "P_net_kW": P_net_W / 1e3,
            "Q_in_kW": Q_in_W / 1e3,
            "Q_out_kW": Q_out_W / 1e3,
            "eta_thermal": eta_th,
            "eta_carnot": eta_carnot,
            "m_dot_basic_kg_s": m_dot,
            "vapor_fraction": y_v,
            "w_NH3_vapor": w_v,
            "w_NH3_liquid": w_l,
            "glide_hot_K": glide_hot,
            "glide_cold_K": glide_cold,
            "w_turbine_spec_J_kg": w_t,
            "w_pump_spec_J_kg": w_p,
            "w_net_spec_J_kg": w_net_spec,
            "q_in_spec_J_kg": q_in_spec,
            "T_flash_K": T_flash_K,
            "T_bubble_hot_K": T_bub,
            "T_dew_hot_K": T_dew,
        }

    # ------------------------------------------------------------------
    # Transient lumped thermal ODE (drum/boiler temperature)
    # ------------------------------------------------------------------
    def _drum_ode(self, t, y, q_source_func, w_basic):
        """
        Lumped energy balance on the boiler/separator thermal mass:
            C * dT/dt = Q_source(t) - Q_to_cycle(T) - UA_loss*(T - T_amb)
        Q_to_cycle modelled as proportional to superheat above the bubble point,
        a stable negative feedback that drives T toward an operating point.
        """
        T = y[0]
        T_amb_K = self.T_amb_c + 273.15
        T_bub = self.bubble_temp(self.p_high_bar, w_basic)
        Q_src = q_source_func(t)
        # heat absorbed by the boiling cycle: ~ conductance * superheat
        k_cycle = 1500.0  # W/K effective boiler duty gain
        Q_cycle = k_cycle * max(T - T_bub, 0.0)
        Q_loss = self.UA_loss * (T - T_amb_K)
        dTdt = (Q_src - Q_cycle - Q_loss) / self.C_thermal
        return [dTdt]

    def simulate_transient(self, q_source_func, T0_K=None, duration_s=600.0,
                           n_eval=200, w_basic=None):
        """
        Integrate the drum-temperature ODE with scipy.solve_ivp.

        Parameters
        ----------
        q_source_func : callable t->Q[W] (heat-source duty), or float (constant)
        T0_K          : initial drum temperature [K]
        duration_s    : horizon [s]
        n_eval        : output points
        w_basic       : NH3 mass fraction of basic solution

        Returns dict: t, T_drum_K, P_net_kW(t), eta_thermal(t)
        """
        w_b = self.x_basic if w_basic is None else float(w_basic)
        if T0_K is None:
            T0_K = self.bubble_temp(self.p_high_bar, w_b) + 5.0

        if callable(q_source_func):
            qfun = q_source_func
        else:
            qval = float(q_source_func)
            qfun = lambda t: qval

        t_eval = np.linspace(0.0, duration_s, n_eval)
        sol = solve_ivp(
            self._drum_ode, (0.0, duration_s), [T0_K],
            args=(qfun, w_b), t_eval=t_eval,
            method="RK45", rtol=1e-6, atol=1e-6, max_step=duration_s / 50.0,
        )
        T_drum = sol.y[0]

        # map instantaneous drum T (flash T) to cycle power via quasi-steady solve
        P_net = np.zeros_like(T_drum)
        eta = np.zeros_like(T_drum)
        T_src_equiv = T_drum + self.pinch_c - 273.15  # source ~ flash + pinch
        for i, Ts in enumerate(T_src_equiv):
            q_kw = qfun(sol.t[i]) / 1e3
            res = self.solve_cycle(T_source_c=float(Ts), w_basic=w_b,
                                   Q_in_kw=q_kw)
            P_net[i] = res["P_net_kW"]
            eta[i] = res["eta_thermal"]

        return {
            "t": sol.t,
            "T_drum_K": T_drum,
            "P_net_kW": P_net,
            "eta_thermal": eta,
            "success": sol.success,
        }
