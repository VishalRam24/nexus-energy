"""
Phase 11 — ML-guided solving.

Submodules:

- :mod:`features`         — system-level and per-timestep feature
                             extractors used as the input to every ML
                             predictor.
- :mod:`uc_warmstart`     — unit-commitment schedule predictors with a
                             confidence-gated cold-start fallback. A
                             pluggable :class:`UCWarmstartPredictor`
                             abstract interface lets callers swap in a
                             trained GNN without touching the energy
                             model.
- :mod:`varfix`           — learned variable fixing: drive partial MIP
                             fixings from historical solve statistics.
- :mod:`clustering`       — learned representative-period selection
                             on top of the Phase 7 k-medoids pipeline.

The design is **torch-optional**: the default predictors are pure
numpy (merit-order, k-NN over a historical-solve bank). A trained
GNN can be supplied via :class:`uc_warmstart.GNNPredictor` when torch
is available at import time.

See ``docs/planning/PHASE_11_ML.md`` for the broader design note.
"""

from __future__ import annotations

from nexus_energy.ml.features import (
    SystemFeatures,
    TimestepFeatures,
    extract_system_features,
    extract_timestep_features,
)
from nexus_energy.ml.uc_warmstart import (
    UCWarmstartPredictor,
    MeritOrderPredictor,
    HistoricalNeighborPredictor,
    GNNPredictor,
    predict_unit_commitment,
    warm_start_from_prediction,
    solve_with_warm_retry,
    WarmStartOutcome,
)
from nexus_energy.ml.varfix import (
    LearnedVarFixer,
    VarFixingStats,
    apply_varfix,
)
from nexus_energy.ml.clustering import (
    LearnedClusterSelector,
    learned_representative_periods,
)

__all__ = [
    # features
    "SystemFeatures", "TimestepFeatures",
    "extract_system_features", "extract_timestep_features",
    # uc_warmstart
    "UCWarmstartPredictor", "MeritOrderPredictor",
    "HistoricalNeighborPredictor", "GNNPredictor",
    "predict_unit_commitment", "warm_start_from_prediction",
    "solve_with_warm_retry", "WarmStartOutcome",
    # varfix
    "LearnedVarFixer", "VarFixingStats", "apply_varfix",
    # clustering
    "LearnedClusterSelector", "learned_representative_periods",
]
