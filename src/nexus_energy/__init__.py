"""
Nexus-Energy: Energy System Optimisation Library
=================================================

Build, simulate, and optimise energy systems with pre-built components
across 15 sectors, backed by the nexus-opt Rust-core solver.

Quick start:
    >>> import nexus_energy as ne
    >>> sys = ne.EnergySystem("my_system")
    >>> elec = sys.add_bus("elec", carrier="electricity")
    >>> sys.add_generator("solar", bus=elec, capacity=500, marginal_cost=0)
    >>> sys.add_generator("gas", bus=elec, capacity=200, marginal_cost=50)
    >>> sys.add_load("demand", bus=elec, amount=300)
    >>> result = sys.optimise()

Submodules:
    - core: data model and solve pipeline
    - components: registry of 23+ component templates (F0 level)
    - sectors: pre-built sector coupling patterns (P2H, P2G, heat)
    - temporal: time-series aggregation, rolling horizon
    - diagnostics: curtailment, bottleneck, summary reports
    - simulation: forward simulation (merit order, rule-based)
    - pypsa_compat: PyPSA Network import/export
    - decomposition: Benders, temporal decomposition
    - stochastic: scenarios, CVaR, robust optimisation
    - mpc: model predictive control / real-time re-optimisation
    - benchmarks: standard benchmark suite
"""

from nexus_energy.core import (
    EnergySystem,
    Bus,
    Carrier,
    Generator,
    Storage,
    Load,
    Link,
    OptimisationResult,
    annuity,
)

from nexus_energy.components.registry import (
    ComponentTemplate,
    ComponentRegistry,
    registry,
    add_component,
)

from nexus_energy.planning import MultiStageSystem, MultiStageResult

from nexus_energy.temporal import (
    RepresentativePeriods,
    AggregationError,
    aggregate_to_representative_days,
    aggregate_with_feature_embedding,
    ml_feature_embedding,
    apply_representative_days,
    representative_period_error,
    k_medoids,
    rolling_horizon_solve,
    # Phase 7.3 / 7.4 — variable-resolution clock
    ResolutionPlan,
    adaptive_resolution_plan,
    multi_resolution_hierarchy,
    apply_adaptive_resolution,
    # Phase 19 — certified reduced-order bounds
    CertifiedBound,
    certify_reduction,
    certified_reduction_demo,
)

from nexus_energy.decomposition import (
    BendersDecomposer,
    BendersIteration,
    BendersResult,
    solve_with_temporal_benders,
    solve_with_spatial_benders,
    temporal_decomposition,
    recommend_decomposition,
    # Phase 8 depth (8.1–8.4)
    SpatialBendersResult,
    solve_with_nested_benders,
    NestedBendersResult,
    StageProblem as NestedStageProblem,
    solve_with_dantzig_wolfe,
    solve_with_column_generation,
    DantzigWolfeResult,
    LPBlock,
)

from nexus_energy.stochastic import (
    Scenario,
    StochasticResult,
    BudgetUncertaintySet,
    ChanceConstraint,
    apply_scenario,
    solve_stochastic,
    solve_stochastic_ph,
    solve_saa_chance_constrained,
    solve_robust,
    evaluate_plan,
    reduce_scenarios,
    generate_demand_scenarios,
    generate_renewable_scenarios,
    generate_moment_matching_scenarios,
    solve_sddip,
    # Phase 9 depth (9.1–9.4) + 2.5
    StageProblem as SDDiPStageProblem,
    SDDiPResult,
    GeneralChanceConstraint,
    solve_general_chance_constrained,
    solve_wasserstein_dro,
    cvar_change_of_measure,
    solve_risk_averse_benders,
    generate_forced_outage_scenarios,
    # Phase 21 — Wasserstein-robust scenario reduction
    reduce_scenarios_wasserstein,
)

from nexus_energy.network_socp import (
    solve_socp_opf,
    SOCPOpfResult,
    solve_socp_opf_multi,
    MultiSocpOpfResult,
    obbt_tighten,
    OBBTStats,
    # Phase 4.2/10.7 + 4.3/10.8
    add_weymouth_pipe,
    WeymouthVars,
    add_head_dependent_hydro,
    HydroHeadVars,
    # Phase 10.2 — capacity-expansion AC-OPF
    solve_socp_opf_expansion,
    SOCPExpansionResult,
)
from nexus_energy.network_polar import (
    solve_ac_opf_polar,
    ACOpfPolarResult,
)

from nexus_energy.ml import (
    SystemFeatures,
    TimestepFeatures,
    extract_system_features,
    extract_timestep_features,
    UCWarmstartPredictor,
    MeritOrderPredictor,
    HistoricalNeighborPredictor,
    GNNPredictor,
    predict_unit_commitment,
    warm_start_from_prediction,
    LearnedVarFixer,
    VarFixingStats,
    apply_varfix,
    LearnedClusterSelector,
    learned_representative_periods,
)

from nexus_energy.ml.uc_warmstart import (
    AdaptiveThresholdController,
    solve_with_adaptive_warmstart,
)
from nexus_energy.ml.clustering import feature_embedding_periods
from nexus_energy.ml.rl_search import (
    RLVarFixer,
    RLVarFixStats,
    RLSolveOutcome,
    solve_with_rl_search,
)

from nexus_energy.diff import (
    EconomicDispatchLayer,
    TorchDispatchLayer,
    DispatchJacobian,
    solve_dispatch_with_sensitivities,
    # Phase 12.1 / 12.3
    MultiBusDispatchProblem,
    MultiBusDispatchSolution,
    MultiBusDispatchLayer,
    solve_multibus_dispatch_with_sensitivities,
    ElasticityFitResult,
    fit_demand_elasticity,
    # Phase 12.2 — differentiable storage / commitment
    StorageDispatchProblem,
    StorageDispatchSolution,
    StorageDispatchLayer,
    solve_storage_dispatch_with_sensitivities,
    SmoothCommitmentLayer,
    smooth_commitment,
    fit_commitment_threshold,
    CommitmentFitResult,
    # Phase 20 — differentiable capacity expansion
    CapacityExpansionProblem,
    CapacityExpansionSolution,
    CapacityExpansionLayer,
    solve_capacity_expansion_with_sensitivities,
    fit_component_params,
    ComponentFitResult,
)
from nexus_energy.sectors import (
    create_power_to_hydrogen,
    create_heat_system,
    create_power_to_gas,
    create_temperature_heat_network,
    create_multi_carrier_system,
)
from nexus_energy import external_solvers, io_tables
from nexus_energy.components.composition import (
    Subsystem,
    CarrierMismatchError,
)
from nexus_energy.cloud import (
    ParallelResult,
    run_scenarios_parallel,
)
from nexus_energy.browser import (
    WASM_SCHEMA_VERSION,
    export_lp_for_browser,
    import_result_from_browser,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "EnergySystem", "Bus", "Carrier", "Generator", "Storage", "Load", "Link",
    "OptimisationResult", "annuity",
    # Components
    "ComponentTemplate", "ComponentRegistry", "registry", "add_component",
    # Planning
    "MultiStageSystem", "MultiStageResult",
    # Temporal aggregation (Phase 7)
    "RepresentativePeriods", "AggregationError",
    "aggregate_to_representative_days", "aggregate_with_feature_embedding",
    "ml_feature_embedding",
    "apply_representative_days",
    "representative_period_error", "k_medoids", "rolling_horizon_solve",
    # Temporal variable-resolution (Phase 7.3 / 7.4) + certified bounds (Phase 19)
    "ResolutionPlan", "adaptive_resolution_plan", "multi_resolution_hierarchy",
    "apply_adaptive_resolution",
    "CertifiedBound", "certify_reduction", "certified_reduction_demo",
    # Differentiable capacity expansion (Phase 20)
    "CapacityExpansionProblem", "CapacityExpansionSolution",
    "CapacityExpansionLayer", "solve_capacity_expansion_with_sensitivities",
    "fit_component_params", "ComponentFitResult",
    # Wasserstein-robust scenario reduction (Phase 21)
    "reduce_scenarios_wasserstein",
    # Decomposition (Phase 8)
    "BendersDecomposer", "BendersIteration", "BendersResult",
    "solve_with_temporal_benders", "solve_with_spatial_benders",
    "temporal_decomposition", "recommend_decomposition",
    "SpatialBendersResult", "solve_with_nested_benders", "NestedBendersResult",
    "NestedStageProblem", "solve_with_dantzig_wolfe",
    "solve_with_column_generation", "DantzigWolfeResult", "LPBlock",
    # Stochastic / robust (Phase 9)
    "Scenario", "StochasticResult", "BudgetUncertaintySet", "ChanceConstraint",
    "apply_scenario", "solve_stochastic", "solve_stochastic_ph",
    "solve_saa_chance_constrained",
    "solve_robust", "evaluate_plan",
    "reduce_scenarios", "generate_demand_scenarios", "generate_renewable_scenarios",
    "generate_moment_matching_scenarios",
    "solve_sddip", "SDDiPStageProblem", "SDDiPResult",
    "GeneralChanceConstraint", "solve_general_chance_constrained",
    "solve_wasserstein_dro", "cvar_change_of_measure",
    "solve_risk_averse_benders", "generate_forced_outage_scenarios",
    # Conic AC-OPF (Phase 10)
    "solve_socp_opf", "SOCPOpfResult",
    "solve_socp_opf_multi", "MultiSocpOpfResult",
    "obbt_tighten", "OBBTStats",
    "add_weymouth_pipe", "WeymouthVars",
    "add_head_dependent_hydro", "HydroHeadVars",
    "solve_socp_opf_expansion", "SOCPExpansionResult",
    # Polar AC-OPF (N_En_Phase 17.3)
    "solve_ac_opf_polar", "ACOpfPolarResult",
    # ML-guided solving (Phase 11)
    "SystemFeatures", "TimestepFeatures",
    "extract_system_features", "extract_timestep_features",
    "UCWarmstartPredictor", "MeritOrderPredictor",
    "HistoricalNeighborPredictor", "GNNPredictor",
    "predict_unit_commitment", "warm_start_from_prediction",
    "LearnedVarFixer", "VarFixingStats", "apply_varfix",
    "LearnedClusterSelector", "learned_representative_periods",
    # ML depth (Phase 11.6 + 11.2)
    "AdaptiveThresholdController", "solve_with_adaptive_warmstart",
    "feature_embedding_periods",
    "RLVarFixer", "RLVarFixStats", "RLSolveOutcome", "solve_with_rl_search",
    # Differentiable + cloud + browser (Phase 12)
    "EconomicDispatchLayer", "TorchDispatchLayer", "DispatchJacobian",
    "solve_dispatch_with_sensitivities",
    "MultiBusDispatchProblem", "MultiBusDispatchSolution",
    "MultiBusDispatchLayer", "solve_multibus_dispatch_with_sensitivities",
    "ElasticityFitResult", "fit_demand_elasticity",
    "StorageDispatchProblem", "StorageDispatchSolution", "StorageDispatchLayer",
    "solve_storage_dispatch_with_sensitivities", "SmoothCommitmentLayer",
    "smooth_commitment", "fit_commitment_threshold", "CommitmentFitResult",
    "ParallelResult", "run_scenarios_parallel",
    "WASM_SCHEMA_VERSION", "export_lp_for_browser", "import_result_from_browser",
    # Sector-coupling builders
    "create_power_to_hydrogen", "create_heat_system", "create_power_to_gas",
    "create_temperature_heat_network", "create_multi_carrier_system",
    # Solver bridge + IO modules (Phase 10.9 / 16.8)
    "external_solvers", "io_tables",
    # Component composability (Phase 22)
    "Subsystem", "CarrierMismatchError",
]
