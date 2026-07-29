"""
EC054 -- Parabolic Trough CSP -- F2a HCE Thermal Model -- Standardized Predict Interface
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from model import ParabolicTroughF2a

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")


def _load_params():
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ComponentModel:
    component_id = "EC054"
    component_name = "Parabolic Trough CSP"
    fidelity = "F2a -- Lumped HCE Thermal Model (Steady-State Energy Balance)"
    version = "1.0.0"

    def __init__(self, params=None):
        self._raw = _load_params()
        if params:
            self._raw["unit"].update(params)
        self._model = ParabolicTroughF2a(self._raw)

    def predict(self, inputs: dict) -> dict:
        """
        Solve the HCE energy balance.

        inputs:
            dni              : Direct Normal Irradiance [W/m2] (scalar or array)
            incidence_angle  : Incidence angle [deg] (scalar or array)
            T_htf_in         : HTF inlet temperature [degC] (scalar or array)
            m_dot            : HTF mass flow rate [kg/s] (scalar or array)
            T_ambient        : Ambient temperature [degC] (default 25)
            h_wind           : Wind convection coeff [W/(m2*K)] (default from params)

        returns:
            Q_useful_W, Q_abs_W, Q_loss_W, T_htf_out_C, T_abs_C, T_glass_C,
            eta_thermal, eta_optical, h_htf, converged
        """
        dni = inputs.get("dni", 900.0)
        theta = inputs.get("incidence_angle", 0.0)
        T_in = inputs.get("T_htf_in", 300.0)
        m_dot = inputs.get("m_dot", 6.0)
        T_amb = inputs.get("T_ambient", 25.0)
        h_wind = inputs.get("h_wind", None)

        # Scalar path
        if np.isscalar(dni) or (isinstance(dni, np.ndarray) and dni.ndim == 0):
            return self._model.solve(
                dni, theta, T_in, m_dot, T_amb, h_wind)

        # Array path
        return self._model.solve_array(
            dni, theta, T_in, m_dot, T_amb, h_wind)

    def get_info(self) -> dict:
        return {
            "name": "Parabolic Trough CSP",
            "ec_id": self.component_id,
            "fidelity": "F2a",
            "description": ("Lumped steady-state HCE thermal model. Iterative energy "
                            "balance on absorber tube and glass envelope to compute "
                            "useful thermal power to HTF (Therminol VP-1)."),
            "inputs": {
                "dni":             {"unit": "W/m2",     "range": [0.0, 1100.0]},
                "incidence_angle": {"unit": "deg",      "range": [0.0, 80.0]},
                "T_htf_in":        {"unit": "degC",     "range": [100.0, 400.0]},
                "m_dot":           {"unit": "kg/s",     "range": [0.5, 12.0]},
                "T_ambient":       {"unit": "degC",     "range": [-10.0, 50.0]},
                "h_wind":          {"unit": "W/(m2*K)", "range": [0.0, 50.0],
                                    "note": "optional, defaults to 10"},
            },
            "outputs": {
                "Q_useful_W":   {"unit": "W",  "note": "Useful thermal power to HTF"},
                "Q_abs_W":      {"unit": "W",  "note": "Solar power absorbed by absorber"},
                "Q_loss_W":     {"unit": "W",  "note": "Total thermal loss from receiver"},
                "T_htf_out_C":  {"unit": "degC", "note": "HTF outlet temperature"},
                "T_abs_C":      {"unit": "degC", "note": "Absorber tube temperature"},
                "T_glass_C":    {"unit": "degC", "note": "Glass envelope temperature"},
                "eta_thermal":  {"unit": "dimensionless"},
                "eta_optical":  {"unit": "dimensionless"},
                "h_htf":        {"unit": "W/(m2*K)", "note": "Internal HTF convection coeff"},
                "converged":    {"unit": "bool"},
            },
            "source": "Forristall (2003), NREL/TP-550-34169",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({
        "dni": 900.0, "incidence_angle": 0.0,
        "T_htf_in": 300.0, "m_dot": 6.0, "T_ambient": 25.0,
    })
    print(f"DNI=900, theta=0, T_in=300C, m_dot=6 kg/s:")
    print(f"  Q_useful = {r['Q_useful_W']/1000:.1f} kW")
    print(f"  Q_abs    = {r['Q_abs_W']/1000:.1f} kW")
    print(f"  Q_loss   = {r['Q_loss_W']/1000:.1f} kW")
    print(f"  T_out    = {r['T_htf_out_C']:.1f} degC")
    print(f"  T_abs    = {r['T_abs_C']:.1f} degC")
    print(f"  T_glass  = {r['T_glass_C']:.1f} degC")
    print(f"  eta_th   = {r['eta_thermal']:.3f}")
    print(f"  eta_opt  = {r['eta_optical']:.3f}")
    print(f"  converged= {r['converged']}")
