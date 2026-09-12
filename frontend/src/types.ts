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

export interface DetectionEvidence {
  evidence_class: EvidenceClass;
  dataset: string;
  detector: Record<string, unknown>;
  metrics: Record<string, number>;
  incidents: DetectionIncident[];
  explanation_basis: string;
  causal_claim: boolean;
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
