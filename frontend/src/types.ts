export type EvidenceClass =
  | "REAL"
  | "SIMULATED"
  | "LITERATURE";

export interface DetectionIncident {
  id: number;
  incident_start: string;
  incident_end: string;
  explanation_timestamp: string;
  timing: "pre_onset" | "post_onset";
  lead_hours: number | null;
  delay_hours: number | null;
  raw_score: number;
  smoothed_score: number;
  top_features: Array<{
    feature: string;
    share: number;
  }>;
  top_groups: Array<{
    group: string;
    share: number;
  }>;
}

export interface VerificationRankedGroup {
  group: string;
  smoothed_contribution: number;
  share: number;
  calibration_percentile: number;
}

export interface CounterfactualRepair {
  repaired_groups: string[];
  window_start: string;
  window_end: string;
  window_rows: number;
  original_raw_score: number;
  repaired_raw_score: number;
  original_smoothed_score: number;
  repaired_smoothed_score: number;
  original_alert: boolean;
  repaired_alert: boolean;
  alert_cleared: boolean;
}

export interface TemporalEvidence {
  window_bins_requested: number;
  window_rows_observed: number;
  window_complete: boolean;
  top_groups: string[];
  rows: Array<{
    timestamp: string;
    percentiles: Record<string, number>;
  }>;
}

export interface VerificationIncident {
  id: number;
  incident_start: string;
  explanation_timestamp: string;
  timing: "pre_onset" | "post_onset";
  lead_hours: number | null;
  delay_hours: number | null;
  original_smoothed_score: number;
  threshold: number;
  ranked_groups: VerificationRankedGroup[];
  temporal_evidence: TemporalEvidence;
  single_group_repairs: CounterfactualRepair[];
  greedy_repair: {
    ranking_basis: string;
    minimum_groups_to_clear: number | null;
    steps: CounterfactualRepair[];
  };
  causal_claim: boolean;
}

export interface DetectionVerification {
  evidence_subtype: string;
  method: Record<string, string>;
  acceptance: Record<string, boolean>;
  incidents: VerificationIncident[];
  limitations: string[];
}

export interface DetectionEvidence {
  evidence_class: EvidenceClass;
  dataset: string;
  detector: Record<string, unknown>;
  metrics: Record<string, number>;
  incidents: DetectionIncident[];
  explanation_basis: string;
  causal_claim: boolean;
  verification: DetectionVerification;
  limitations: string[];
}

export interface EvidenceSummary {
  project: string;
  deployment: {
    mode: string;
    override_equipment_ctrl: boolean;
  };
  evidence_classes: {
    REAL: {
      scope: string;
      detection: DetectionEvidence;
    };
    SIMULATED: {
      scope: string;
      digital_twin: Record<string, any>;
      baseline: Record<string, any>;
      optimizer: Record<string, any>;
      action_explanations: Record<string, any>;
      robustness: Record<string, any>;
    };
    LITERATURE: {
      scope: string;
      runtime_embedded: boolean;
    };
  };
  safety: {
    deployment_mode: string;
    override_equipment_ctrl: boolean;
    recommendation_valid_for_seconds: number;
    requires_reoptimization: boolean;
    open_loop_schedule_approved: boolean;
    robustness_status: string;
  };
}

export interface ProofManifestRealPath {
  source: string;
  incident_id: number;
  detector_evidence_class: "REAL";
  candidate_hypotheses: string[];
  bridge_status: string;
  physical_bridge_available: boolean;
  missing_requirements: string[];
  downstream_physics_executed: boolean;
  robust_control_executed: boolean;
  recommendation: null;
}

export interface ProofManifestSimulatedPath {
  source: string;
  scope: "SIMULATION_ONLY";
  evidence_class: "SIMULATED";
  connected_to_real_incident: false;
  proof_id: string;
  inverse_physics_recovery: {
    passed: boolean;
    truth_total_outflow_kg_s: number;
    inferred_total_outflow_kg_s: number;
    absolute_error_kg_s: number;
  };
  physical_state: {
    total_outflow_kg_s: {
      lower: number;
      center: number;
      upper: number;
    };
    leak_identifiable: boolean;
    leak_kg_s: null;
  };
  robust_action: {
    commands: Array<[string, number]>;
    scenario_count: number;
    robust_safe: boolean;
    valid_for_seconds: number;
    worst_case_min_pressure_bar_g: number;
    worst_case_max_pressure_bar_g: number;
    minimum_reserve_kg_s: number;
    horizon_energy_kwh: number;
    horizon_objective_value: number;
  };
}

export interface ProofManifestResponse {
  schema_version: string;
  manifest_id: string;
  project: "AeroXAI";
  causal_claim: false;
  deployment: {
    mode: "advisory";
    override_equipment_ctrl: false;
  };
  contains_evidence_classes: Array<
    "REAL" | "SIMULATED"
  >;
  real_evidence_path: ProofManifestRealPath;
  simulated_validation_path: ProofManifestSimulatedPath;
  separation_assertions: string[];
  allowed_claims: string[];
  forbidden_claims: string[];
}

export interface TwinRequest {
  initial_pressure_bar_g: number;
  mass_flow_in_kg_s: number;
  demand_mass_flow_kg_s: number;
  leak_mass_flow_kg_s: number;
  duration_seconds: number;
  timestep_seconds: number;
}

export interface TwinResponse {
  evidence_class: EvidenceClass;
  initial_pressure_bar_g: number;
  final_pressure_bar_g: number;
  minimum_pressure_bar_g: number;
  maximum_pressure_bar_g: number;
  duration_seconds: number;
  timestep_seconds: number;
  trajectory: Array<{
    time_seconds: number;
    pressure_bar_g: number;
  }>;
  limitations: string[];
}

export interface OptimizationRequest {
  initial_pressure_bar_g: number;
  demand_profile_kg_s: number[];
  leak_mass_flow_kg_s: number;
}

export interface OptimizationResponse {
  evidence_class: EvidenceClass;
  solver_status: number;
  solver_message: string;
  horizon_intervals: number;
  predicted_horizon_energy_kwh: number;
  current_actions: Array<{
    compressor_id: string;
    kind: "fixed" | "vsd";
    state: "on" | "off";
    load_fraction: number;
  }>;
  explanation: {
    reason: string;
    predicted_pressure_end_bar_g: number;
    safety_margin_bar: number;
    reserve_available_kg_s: number;
    required_reserve_kg_s: number;
    binding_constraints: string[];
    causal_claim: boolean;
  };
  safety: {
    deployment_mode: "advisory";
    override_equipment_ctrl: false;
    valid_for_seconds: number;
    requires_reoptimization: true;
    open_loop_schedule_approved: boolean;
    robustness_status: string;
  };
  limitations: string[];
}
