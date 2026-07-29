"""
EC221 -- Magnetohydrodynamic (MHD) Generator -- F2a Physics-Lumped Channel Model

A 1D lumped quasi-one-dimensional model of a Faraday-type MHD power channel.
The plasma (seeded combustion gas) flows along x through a transverse magnetic
field B. The motional EMF u*B drives a current density J = sigma*(u*B - E)
across segmented electrodes; the load factor K = E/(u*B) sets how much of the
EMF appears across the external load. The conducting fluid does work against the
J x B retarding (Lorentz) force, decelerating and cooling as electrical energy
is extracted, while ohmic (Joule) dissipation reheats it. Plasma electrical
conductivity is obtained from a Saha-equation ionization balance of the alkali
seed, and the Hall effect (beta = mu_e * B) reduces the effective Faraday
conductivity.

State vector integrated spatially along the channel with scipy.solve_ivp:
    y = [u, T, p]   (velocity [m/s], static temperature [K], pressure [Pa])

Governing quasi-1D conservation laws (Sutton & Sherman 1965, Ch. 9;
Rosa 1987, Ch. 3). With constant mass flow mdot = rho*u*A(x):

  Mass:      rho * u * A = mdot = const  ->  rho(x) = mdot / (u * A(x))
  Momentum:  rho*u du/dx = -dp/dx - J*B        (J*B is the Lorentz body force)
  Energy:    rho*u d/dx(cp*T + u^2/2) = J*E_internal_loss - J*E_load
             i.e. the flow loses the extracted electrical work and keeps Joule heat:
             rho*u d/dx(h0) = -J*E   (work delivered to the load, per unit volume)
             where the Joule heat J^2/sigma is retained internally.

Electrodynamics (Faraday channel, segmented electrodes):
  E_field   = K * u * B                          (transverse load electric field)
  sigma_eff = sigma / (1 + beta^2)               (Hall-reduced Faraday conductivity)
  J         = sigma_eff * (u*B - E) = sigma_eff*u*B*(1-K)   (Faraday current density)
  p_elec    = J * E   = sigma_eff*u^2*B^2*K*(1-K)  (delivered electrical power density)
  p_joule   = J^2/sigma_eff                        (ohmic dissipation, reheats gas)
  p_body    = J * B   (Lorentz force density opposing the flow)

Conductivity (Saha ionization of alkali seed -- Sutton & Sherman 1965, Eq. 5-?):
  n_e^2/(n_seed - n_e) = (2*Z_i/Z_0)*(2*pi*m_e*k*T/h^2)^1.5 * exp(-E_ion/(k*T))
  sigma = n_e * e^2 / (m_e * nu_collision)  ~  sigma_scale * n_e * e * mu_e

References:
  Rosa, R.J. (1987). Magnetohydrodynamic Energy Conversion. McGraw-Hill.
  Sutton, G.W. & Sherman, A. (1965). Engineering Magnetohydrodynamics. McGraw-Hill.
  Messerle, H.K. (1995). MHD Electrical Power Generation. Wiley.
"""

import numpy as np
from scipy.integrate import solve_ivp


class MHD_F2a:
    """Quasi-1D lumped Faraday MHD channel with Saha conductivity + Hall effect."""

    # Physical constants (SI)
    k_B = 1.380649e-23      # J/K  Boltzmann
    e_ch = 1.602176634e-19  # C    elementary charge
    m_e = 9.1093837015e-31  # kg   electron mass
    h_pl = 6.62607015e-34   # J*s  Planck

    def __init__(self, params: dict):
        u = params["unit"]
        self.B0 = u["B_field"]["value"]
        self.K0 = u["K_load"]["value"]
        self.L = u["channel_length"]["value"]
        self.w = u["channel_width"]["value"]          # electrode spacing d
        self.h_in = u["channel_height"]["value"]
        self.area_ratio = u["area_ratio"]["value"]
        self.u_in = u["u_inlet"]["value"]
        self.T_in = u["T_inlet"]["value"]
        self.p_in = u["p_inlet"]["value"]
        self.mdot = u["mdot"]["value"]
        self.cp = u["cp_plasma"]["value"]
        self.gamma = u["gamma_plasma"]["value"]
        self.M_gas = u["M_gas"]["value"]
        self.seed_frac = u["seed_fraction"]["value"]
        self.E_ion_eV = u["E_ion_seed"]["value"]
        self.sigma_floor = u["sigma_floor"]["value"]
        self.sigma_scale = u["sigma_scale"]["value"]
        self.mu_e = u["mu_e"]["value"]

        self.R_specific = 8.314462618 / self.M_gas   # J/(kg*K)
        self.A_in = self.w * self.h_in               # inlet cross-section [m^2]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def area(self, x):
        """Channel cross-sectional area A(x) [m^2], linear divergence inlet->outlet."""
        frac = np.clip(x / self.L, 0.0, 1.0)
        return self.A_in * (1.0 + (self.area_ratio - 1.0) * frac)

    # ------------------------------------------------------------------
    # Plasma conductivity from Saha ionization
    # ------------------------------------------------------------------
    def electron_density(self, T, p):
        """Equilibrium electron number density [1/m^3] from Saha balance.

        Saha equation for single ionization of the alkali seed:
            n_e * n_i / n_n = (2 Z_i / Z_0)(2 pi m_e k T / h^2)^1.5 exp(-E_ion/kT)
        Charge neutrality n_e = n_i, and total seed n_seed = n_n + n_i give:
            n_e^2 / (n_seed - n_e) = S(T)
        solved with the quadratic formula. (Sutton & Sherman 1965.)
        """
        T = np.asarray(T, dtype=float)
        p = np.asarray(p, dtype=float)
        # total number density of gas from ideal gas: n = p/(kT)
        n_total = p / (self.k_B * T)
        n_seed = self.seed_frac * n_total

        E_ion_J = self.E_ion_eV * self.e_ch
        # Saha right-hand side S(T); statistical weight ratio 2*Z_i/Z_0 ~ 1
        # (alkali: ground neutral 2S_1/2 g=2, ion closed-shell g=1 -> 2*1/2 = 1)
        S = (2.0 * np.pi * self.m_e * self.k_B * T / self.h_pl ** 2) ** 1.5 \
            * np.exp(-E_ion_J / (self.k_B * T))
        # n_e^2 + S*n_e - S*n_seed = 0  ->  n_e = (-S + sqrt(S^2 + 4 S n_seed))/2
        n_e = 0.5 * (-S + np.sqrt(S ** 2 + 4.0 * S * n_seed))
        return np.maximum(n_e, 0.0)

    def conductivity(self, T, p):
        """Scalar plasma electrical conductivity sigma [S/m].

        sigma = sigma_scale * n_e * e * mu_e
        with electron mobility mu_e (set so beta = mu_e*B in the Hall regime).
        Clamped to a floor for numerical robustness in cold regions.
        """
        n_e = self.electron_density(T, p)
        sigma = self.sigma_scale * n_e * self.e_ch * self.mu_e
        return np.maximum(sigma, self.sigma_floor)

    def hall_parameter(self, B):
        """Hall parameter beta = omega_e * tau_e = mu_e * B [-]."""
        return self.mu_e * B

    def sigma_effective(self, sigma, B):
        """Hall-reduced effective Faraday conductivity sigma/(1+beta^2)."""
        beta = self.hall_parameter(B)
        return sigma / (1.0 + beta ** 2)

    # ------------------------------------------------------------------
    # Local electrodynamics at a station
    # ------------------------------------------------------------------
    def local_electrics(self, u, T, p, B, K):
        """Return dict of local electrodynamic quantities at one channel station."""
        sigma = self.conductivity(T, p)
        sigma_eff = self.sigma_effective(sigma, B)
        EMF_field = u * B                     # V/m  (motional field)
        E_load = K * EMF_field                # V/m  (load electric field)
        J = sigma_eff * (EMF_field - E_load)  # A/m^2 Faraday current density
        p_elec = J * E_load                   # W/m^3 delivered electrical power
        p_joule = J ** 2 / sigma_eff          # W/m^3 ohmic dissipation
        f_body = J * B                        # N/m^3 Lorentz retarding force density
        return {
            "sigma": sigma, "sigma_eff": sigma_eff,
            "EMF_field": EMF_field, "E_load": E_load,
            "J": J, "p_elec": p_elec, "p_joule": p_joule, "f_body": f_body,
        }

    # ------------------------------------------------------------------
    # Quasi-1D ODE right-hand side  dy/dx
    # ------------------------------------------------------------------
    def _rhs(self, x, y, B, K):
        u, T, p = y
        u = max(u, 1.0)
        T = max(T, 300.0)
        p = max(p, 1e3)
        A = self.area(x)
        rho = self.mdot / (u * A)             # mass conservation

        el = self.local_electrics(u, T, p, B, K)
        f_body = el["f_body"]                 # Lorentz force density opposing flow
        p_elec = el["p_elec"]                 # electrical power density extracted

        # Momentum:  rho*u du/dx = -dp/dx - f_body
        # Energy (stagnation enthalpy):  rho*u d/dx(cp*T + u^2/2) = -p_elec
        #   (work delivered to the load leaves the flow; Joule heat is retained
        #    inside the fluid so it does not appear as a stagnation-enthalpy sink).
        # Close the system with the ideal-gas relation p = rho*R*T, differentiated:
        #   dp/dx = R*(T*drho/dx + rho*dT/dx), and drho/dx from rho = mdot/(u*A).
        #
        # Solve the linear 2x2 system for du/dx, dT/dx, then dp/dx.
        R = self.R_specific
        dA_dx = (self.area_ratio - 1.0) * self.A_in / self.L
        # drho/dx = -rho*(1/u du/dx + 1/A dA/dx)
        # Momentum:  rho*u*du + dp = -f_body dx
        # dp = R*(T*drho + rho*dT)
        #    = R*T*(-rho/u*du - rho/A*dA) + R*rho*dT
        # => rho*u*du + R*rho*dT - (R*rho*T/u)*du = -f_body + (R*rho*T/A)*dA
        # Energy:  rho*u*(cp*dT + u*du) = -p_elec dx
        # Two equations in (du/dx, dT/dx):
        a11 = rho * u - R * rho * T / u
        a12 = R * rho
        b1 = -f_body + (R * rho * T / A) * dA_dx
        a21 = rho * u * u
        a22 = rho * u * self.cp
        b2 = -p_elec

        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-12:
            det = 1e-12 if det >= 0 else -1e-12
        du_dx = (b1 * a22 - a12 * b2) / det
        dT_dx = (a11 * b2 - b1 * a21) / det

        # dp/dx from ideal gas differential
        drho_dx = -rho * (du_dx / u + dA_dx / A)
        dp_dx = R * (T * drho_dx + rho * dT_dx)

        return [du_dx, dT_dx, dp_dx]

    # ------------------------------------------------------------------
    # Spatial integration along the channel
    # ------------------------------------------------------------------
    def simulate(self, B=None, K=None, u_in=None, T_in=None, p_in=None, n_points=200):
        """Integrate the quasi-1D channel from inlet (x=0) to outlet (x=L).

        Returns dict of arrays over x plus integral plant quantities.
        """
        B = self.B0 if B is None else B
        K = self.K0 if K is None else K
        u0 = self.u_in if u_in is None else u_in
        T0 = self.T_in if T_in is None else T_in
        p0 = self.p_in if p_in is None else p_in

        x_eval = np.linspace(0.0, self.L, n_points)
        sol = solve_ivp(
            self._rhs, (0.0, self.L), [u0, T0, p0],
            args=(B, K), t_eval=x_eval, method="RK45",
            rtol=1e-7, atol=1e-7, max_step=self.L / 50.0,
        )
        x = sol.t
        u = sol.y[0]
        T = sol.y[1]
        p = sol.y[2]
        A = self.area(x)
        rho = self.mdot / (u * A)

        # Local electrodynamics along the channel
        el = self.local_electrics(u, T, p, B, K)
        # Volumetric quantities -> power per unit length = density * A
        dPelec_dx = el["p_elec"] * A           # W/m
        P_elec = np.trapezoid(dPelec_dx, x)    # W  total electrical power

        # Joule heat retained
        dPjoule_dx = el["p_joule"] * A
        P_joule = np.trapezoid(dPjoule_dx, x)

        # EMF integrated transversely (over electrode spacing w)
        EMF_terminal = el["EMF_field"] * self.w      # V  open-circuit per station
        V_load_terminal = el["E_load"] * self.w      # V  loaded voltage per station

        # Stagnation enthalpy flux at inlet/outlet (first-law check)
        h0_in = self.cp * T[0] + 0.5 * u[0] ** 2
        h0_out = self.cp * T[-1] + 0.5 * u[-1] ** 2
        H_in = self.mdot * h0_in                # W
        H_out = self.mdot * h0_out
        dH = H_in - H_out                        # enthalpy delivered to channel

        # Enthalpy-extraction ratio: fraction of inlet stagnation enthalpy flux
        # converted to *stagnation-enthalpy drop* (single-channel MHD ~ a few %).
        eta_enthalpy_extraction = dH / H_in if H_in > 0 else 0.0

        # Electrical (Faraday channel) efficiency: useful electric output over
        # the total electromagnetic power P_elec + P_joule. For a Faraday channel
        #   P_elec = sigma_eff*u^2*B^2*K*(1-K),  P_joule = sigma_eff*u^2*B^2*(1-K)^2
        #   => eta_electric = K  (identically). Maximum *power* is at K=0.5.
        # (Rosa 1987; Sutton & Sherman 1965.)
        denom = P_elec + P_joule
        eta_electric = P_elec / denom if denom > 1e-9 else 0.0
        eta_electric = float(np.clip(eta_electric, 0.0, 1.0))

        return {
            "x": x, "u": u, "T": T, "p": p, "rho": rho, "area": A,
            "sigma": el["sigma"], "sigma_eff": el["sigma_eff"],
            "J": el["J"], "E_load_field": el["E_load"],
            "EMF_field": el["EMF_field"],
            "power_density": el["p_elec"],
            "EMF_terminal": EMF_terminal,
            "V_load_terminal": V_load_terminal,
            "P_elec_W": P_elec,
            "P_joule_W": P_joule,
            "H_in_W": H_in, "H_out_W": H_out, "dH_W": dH,
            "eta_enthalpy_extraction": eta_enthalpy_extraction,
            "eta_electric": eta_electric,
            "B": B, "K": K,
            "beta_hall": self.hall_parameter(B),
        }
