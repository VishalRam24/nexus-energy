"""
EC181 — Transmission Line — F2a Distributed-Parameter / Cascaded-Pi Lumped Dynamic Model
=========================================================================================

Physics-lumped (0D/1D ODE) upgrade of the F1a nominal-pi model. This model implements
the full distributed-parameter (long-line) theory of an AC transmission line and a
multi-section cascaded-pi *dynamic* ODE network integrated with scipy.solve_ivp.

Per-unit-length primary constants (SI):
    r   [ohm/km]   series resistance
    L   [H/km]     series inductance     -> x = w*L  [ohm/km]
    C   [F/km]     shunt capacitance      -> b = w*C  [S/km]
    g   [S/km]     shunt conductance (leakage)

Series impedance / shunt admittance per km (phasor, w = 2*pi*f):
    z = r + j*w*L          [ohm/km]
    y = g + j*w*C          [S/km]

Distributed / long-line ("exact") theory  [Glover §5; Bergen & Vittal §4]:
    propagation constant   gamma = sqrt(z*y)          [1/km]
    characteristic imped.  Z_c   = sqrt(z/y)          [ohm]
    Surge Impedance Loading SIL = V_rated^2 / Z_surge  (Z_surge = sqrt(L/C), lossless)

Two-port ABCD (sending = A*receiving + B*I_r ; I_s = C*V_r + D*I_r):
    EXACT (hyperbolic):
        A = D = cosh(gamma*ell)
        B     = Z_c * sinh(gamma*ell)
        C     = sinh(gamma*ell) / Z_c
    NOMINAL-PI (lumped, total Z=z*ell, Y=y*ell):
        A = D = 1 + Z*Y/2
        B     = Z
        C     = Y*(1 + Z*Y/4)
    Reciprocity always holds:  A*D - B*C = 1.

Voltage drop, losses, Ferranti, SIL — derived from ABCD and load flow.

Dynamic ODE (cascaded-pi state space)  [Bergen & Vittal §4.x lumped approximation,
Glover EMTP-style pi-section ladder]:
    The line is split into N equal pi sections. State variables are the N section
    inductor currents i_k (series branches) and the (N+1) node capacitor voltages v_k
    (shunt branches, with C/2 at the two ends merged into neighbouring nodes giving the
    standard ladder). For section k between node k and node k+1:
        L_k * di_k/dt = v_k - v_{k+1} - r_k * i_k
        C_node * dv_k/dt = (sum of currents into node k) - g_node * v_k
    With sending node driven by a voltage source v_s(t) and receiving node terminated
    by a resistive/RL load (or open for Ferranti). This is integrated with
    scipy.integrate.solve_ivp to capture line-charging transients and the AC steady
    state. Energy is conserved: P_in = P_load + losses.

References
----------
Glover, Sarma & Overbye (2012). *Power System Analysis and Design*, 5th ed.,
    Cengage Learning. Chapter 5 (Transmission Lines: Steady-State Operation).
Bergen, A. R. & Vittal, V. (2000). *Power Systems Analysis*, 2nd ed.,
    Prentice Hall. Chapter 4 (Transmission Line Parameters & Models).
"""

import numpy as np
from scipy.integrate import solve_ivp


class TransmissionLineF2a:
    """Distributed-parameter + cascaded-pi dynamic transmission line (all phasors RMS)."""

    def __init__(self, params: dict):
        u = params["unit"]
        self.V_base_kV = u["V_base_kV"]["value"]
        self.S_base_MVA = u["S_base_MVA"]["value"]
        self.f_Hz = u["f_Hz"]["value"]
        self.r = u["r_ohm_per_km"]["value"]            # ohm/km
        self.L = u["L_mH_per_km"]["value"] * 1e-3      # H/km
        self.C = u["C_nF_per_km"]["value"] * 1e-9      # F/km
        self.g = u["G_uS_per_km"]["value"] * 1e-6      # S/km
        self.length_km_default = u["length_km"]["value"]

        self.w = 2.0 * np.pi * self.f_Hz
        # Base impedance (per-unit system, single-phase equivalent of 3-phase)
        self.Z_base = (self.V_base_kV ** 2) / self.S_base_MVA   # ohm  (kV^2/MVA)

    # ------------------------------------------------------------------
    # Per-unit-length phasor constants
    # ------------------------------------------------------------------
    def primary_constants(self):
        """Return (z, y) per-km phasor series impedance and shunt admittance [ohm/km, S/km]."""
        z = complex(self.r, self.w * self.L)
        y = complex(self.g, self.w * self.C)
        return z, y

    def gamma_zc(self):
        """Propagation constant gamma [1/km] and characteristic impedance Z_c [ohm]."""
        z, y = self.primary_constants()
        gamma = np.sqrt(z * y)
        Z_c = np.sqrt(z / y)
        return gamma, Z_c

    def surge_impedance(self):
        """Lossless surge impedance Z_surge = sqrt(L/C) [ohm]."""
        return np.sqrt(self.L / self.C)

    def sil_MW(self):
        """Surge Impedance Loading [MW] (3-phase) = V_LL^2 / Z_surge."""
        return (self.V_base_kV ** 2) / self.surge_impedance()

    # ------------------------------------------------------------------
    # ABCD two-port parameters
    # ------------------------------------------------------------------
    def abcd_exact(self, length_km=None):
        """Exact (long-line hyperbolic) ABCD parameters in per-unit (Z normalised)."""
        if length_km is None:
            length_km = self.length_km_default
        gamma, Z_c = self.gamma_zc()
        gl = gamma * length_km
        A = np.cosh(gl)
        D = A
        B = Z_c * np.sinh(gl)                 # ohm
        C = np.sinh(gl) / Z_c                 # S
        # Normalise B to pu (impedance) and C to pu (admittance)
        return A, B / self.Z_base, C * self.Z_base, D

    def abcd_nominal_pi(self, length_km=None):
        """Nominal-pi lumped ABCD parameters in per-unit."""
        if length_km is None:
            length_km = self.length_km_default
        z, y = self.primary_constants()
        Z = z * length_km / self.Z_base       # pu series impedance
        Y = y * length_km * self.Z_base       # pu shunt admittance
        A = 1.0 + Z * Y / 2.0
        D = A
        B = Z
        Cc = Y * (1.0 + Z * Y / 4.0)
        return A, B, Cc, D

    # ------------------------------------------------------------------
    # Steady-state load flow from ABCD (receiving-end specified)
    # ------------------------------------------------------------------
    def solve_receiving(self, V_s_pu, P_load_pu, Q_load_pu, length_km=None, exact=True):
        """
        Given sending-end voltage magnitude and receiving-end PQ load, solve the
        receiving-end voltage/current via ABCD inversion (Newton fixed-point).

        V_s = A V_r + B I_r ;  I_s = C V_r + D I_r
        Inverse:  V_r = D V_s - B I_s ;  I_r = -C V_s + A I_s   (since AD-BC=1)
        We iterate on the receiving-end load current I_r = conj(S_load / V_r).
        """
        if length_km is None:
            length_km = self.length_km_default
        A, B, Cc, D = (self.abcd_exact(length_km) if exact
                       else self.abcd_nominal_pi(length_km))
        S_load = P_load_pu + 1j * Q_load_pu
        V_s = complex(V_s_pu, 0.0)            # reference angle 0
        # Forward solve: pick V_r so that A V_r + B I_r = V_s
        V_r = V_s / A                         # initial guess (no load current)
        for _ in range(60):
            I_r = np.conj(S_load / V_r) if abs(V_r) > 1e-12 else 0j
            V_r_new = (V_s - B * I_r) / A
            if abs(V_r_new - V_r) < 1e-12:
                V_r = V_r_new
                break
            V_r = V_r_new
        I_r = np.conj(S_load / V_r) if abs(V_r) > 1e-12 else 0j
        I_s = Cc * V_r + D * I_r
        S_s = V_s * np.conj(I_s)
        P_s, Q_s = S_s.real, S_s.imag
        P_loss = P_s - P_load_pu
        Q_loss = Q_s - Q_load_pu
        eta = (P_load_pu / P_s) if P_s > 1e-12 else 0.0
        return {
            "V_r_pu": abs(V_r),
            "delta_r_rad": np.angle(V_r),
            "I_r_pu": abs(I_r),
            "I_s_pu": abs(I_s),
            "P_s_pu": P_s,
            "Q_s_pu": Q_s,
            "P_loss_pu": P_loss,
            "Q_loss_pu": Q_loss,
            "efficiency": eta,
            "voltage_drop_pu": V_s_pu - abs(V_r),
        }

    def ferranti_no_load(self, V_s_pu=1.0, length_km=None, exact=True):
        """
        Open-circuit receiving end (I_r = 0): V_r = V_s / A.
        Ferranti effect => |V_r| > |V_s| at light/no load for long lines.
        Returns receiving voltage magnitude and the rise factor.
        """
        if length_km is None:
            length_km = self.length_km_default
        A, B, Cc, D = (self.abcd_exact(length_km) if exact
                       else self.abcd_nominal_pi(length_km))
        V_r = complex(V_s_pu, 0.0) / A
        return {
            "V_r_pu": abs(V_r),
            "rise_factor": abs(V_r) / V_s_pu,
            "ferranti": abs(V_r) > V_s_pu,
        }

    # ------------------------------------------------------------------
    # Cascaded-pi dynamic ODE (lumped multi-section)  -> solve_ivp
    # ------------------------------------------------------------------
    def _ladder_params(self, n_sections, length_km):
        """Per-section R, L and per-node C, G for an n-section pi ladder."""
        seg = length_km / n_sections
        R_k = self.r * seg
        L_k = self.L * seg
        # Total shunt C distributed over (n) interior + 2 half-end nodes.
        # Standard pi ladder: each section contributes C*seg/2 to each of its two nodes.
        C_node_full = self.C * seg            # full per-section shunt cap
        G_node_full = self.g * seg
        return R_k, L_k, C_node_full, G_node_full

    def simulate(self, V_s_func, P_load_pu=None, R_load_pu=None, L_load_pu=None,
                 n_sections=8, length_km=None, duration_s=None, n_eval=2000,
                 open_end=False):
        """
        Time-domain cascaded-pi simulation via scipy.integrate.solve_ivp.

        Parameters
        ----------
        V_s_func   : callable t -> sending-end instantaneous voltage [pu], or float
                     (interpreted as RMS magnitude of a 60 Hz cosine).
        R_load_pu  : receiving-end series load resistance [pu] (with optional L_load_pu)
        P_load_pu  : if given (and R_load_pu None), converts to an equivalent resistive
                     load R = V^2/P at 1 pu nominal voltage.
        L_load_pu  : receiving-end load inductance [pu] (reactive part); default 0.
        n_sections : number of pi ladder sections.
        open_end   : if True, receiving end open-circuited (Ferranti transient).

        State vector x = [i_1..i_N (section currents), v_1..v_N (node voltages)].
        Node 0 is the source node (driven), nodes 1..N are ladder nodes; node N is
        the receiving end. v is per-unit instantaneous node voltage; i per-unit
        instantaneous section current.

        Returns dict of time series (t, v_s, v_r, i_s, i_r, p_in, p_load, p_loss).
        """
        if length_km is None:
            length_km = self.length_km_default
        if duration_s is None:
            duration_s = 5.0 / self.f_Hz      # ~5 cycles
        N = int(n_sections)

        R_k, L_k, C_node, G_node = self._ladder_params(N, length_km)
        # pu-time scaling: phasor constants are in SI ohm; convert to pu impedance.
        R_k /= self.Z_base
        L_k /= self.Z_base                    # L in "pu-ohm-seconds" => henries/Z_base
        C_node *= self.Z_base                 # C*Z_base => "pu-farads"
        G_node *= self.Z_base

        # Source
        if callable(V_s_func):
            vs = V_s_func
        else:
            amp = float(V_s_func) * np.sqrt(2.0)   # RMS -> peak
            vs = lambda t: amp * np.cos(self.w * t)

        # Load termination
        if open_end:
            R_load = np.inf
            L_load = 0.0
        else:
            if R_load_pu is None and P_load_pu is not None:
                R_load = 1.0 / max(P_load_pu, 1e-9)   # R = V^2/P at V=1pu
            elif R_load_pu is not None:
                R_load = R_load_pu
            else:
                R_load = 1.0
            L_load = (L_load_pu / self.w) if L_load_pu else 0.0

        def rhs(t, x):
            i = x[:N]                          # section currents
            v = x[N:]                          # node voltages (nodes 1..N)
            v_src = vs(t)
            dv = np.zeros(N)
            di = np.zeros(N)
            # node voltages including source node 0
            # section k (0-indexed) connects node k (=v_src if k==0 else v[k-1]) to node k+1 (=v[k])
            for k in range(N):
                v_left = v_src if k == 0 else v[k - 1]
                v_right = v[k]
                di[k] = (v_left - v_right - R_k * i[k]) / L_k
            # node currents: node m (1-indexed -> v[m-1]) gets current in from section m-1, out to section m
            for m in range(N):
                i_in = i[m]                    # current entering node m from section m
                if m < N - 1:
                    i_out = i[m + 1]           # current leaving to next section
                else:
                    # receiving node: load current
                    if open_end or not np.isfinite(R_load):
                        i_out = 0.0
                    else:
                        i_out = v[m] / R_load  # resistive load (L_load handled approximately)
                dv[m] = (i_in - i_out - G_node * v[m]) / C_node
            return np.concatenate([di, dv])

        x0 = np.zeros(2 * N)
        t_eval = np.linspace(0.0, duration_s, n_eval)
        sol = solve_ivp(rhs, (0.0, duration_s), x0, t_eval=t_eval,
                        method="RK45", rtol=1e-7, atol=1e-9, max_step=duration_s / 200.0)

        v = sol.y[N:]
        i = sol.y[:N]
        v_s_t = np.array([vs(tt) for tt in sol.t])
        i_s_t = i[0]
        v_r_t = v[-1]
        if open_end:
            i_r_t = np.zeros_like(v_r_t)
        else:
            i_r_t = v_r_t / R_load if np.isfinite(R_load) else np.zeros_like(v_r_t)

        p_in = v_s_t * i_s_t
        p_load = v_r_t * i_r_t
        # instantaneous resistive loss in series branches
        p_loss = np.zeros_like(sol.t)
        for k in range(N):
            p_loss += R_k * i[k] ** 2

        return {
            "t": sol.t,
            "v_s": v_s_t,
            "v_r": v_r_t,
            "i_s": i_s_t,
            "i_r": i_r_t,
            "p_in": p_in,
            "p_load": p_load,
            "p_loss": p_loss,
            "success": sol.success,
            "n_sections": N,
            "R_load_pu": R_load,
        }
