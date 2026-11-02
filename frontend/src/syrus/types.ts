export type Mode = "asof" | "retro";
export type Depth = "quick" | "investigation" | "technical";

export interface SourceRef {
  file: string;
  path?: string;
  symbol?: string;
  sha256?: string;
}

export interface CaseRecord {
  id: string;
  kind: string;
  visibility: "both" | "retro";
  value: unknown;
  display: string | Record<string, unknown>;
  caveat: string | null;
  source: SourceRef[];
}

export interface CaseFinding {
  id: string;
  rule: string;
  inputs: string[];
  fires: boolean;
  visibility: "both" | "retro";
  text: string;
  detail: string | null;
}

export interface Bilingual {
  en: string;
  vi: string;
}

export interface CaseTest {
  id: string;
  compares: string[][];
  observations: Array<{
    availability: "loaded" | "raw_dataset" | "external";
    text: Bilingual;
    availability_text: Bilingual;
  }>;
  maturity: string;
  maturity_text: Bilingual;
  result: string;
  result_text: Bilingual;
  follow_up: Bilingual;
}

export interface CaseFile {
  schema: string;
  case_id: string;
  alert_time: string;
  asset: Record<string, string>;
  mode: Mode;
  source?: { type: string; incident_id?: number };
  records: CaseRecord[];
  findings: CaseFinding[];
  tests: CaseTest[];
  pair_limits: Array<{
    pair: string[];
    limit: string;
    tests: string[];
    text: Bilingual;
  }>;
  sha256: string;
}

export interface Manifest {
  cases: Array<{ case_id: string; alert_time: string }>;
  sha256: string;
}

export interface Health {
  status: "ok" | "offline";
  reason: string | null;
  model: string;
  mock: boolean;
  cases: number;
}

export interface SessionInfo {
  token: string;
  case_id: string;
  mode: Mode;
  expires_in: number;
  case_sha256: string;
  chat_available: boolean;
  offline_reason: string | null;
}

export interface Qualification {
  id: string;
  severity: "critical" | "context";
  short: boolean;
  text: string;
}

export interface AnswerBlock {
  text: string;
  kind: string;
  records: string[];
  findings: string[];
  qualifications: Qualification[];
  display: string;
}

export interface Answer {
  status: "ok";
  lang: "en" | "vi";
  depth: Depth;
  purposes: string[];
  refusal: string | null;
  refusal_by: "model" | "server" | null;
  blocks: AnswerBlock[];
  also_relevant: Array<{ finding_id: string; text: string }>;
  operator_note: string | null;
  narrowed: string | null;
  next_question: { test_id: string; text: string } | null;
  alternatives: string[];
  gaps: string[];
  answer: string;
  provenance: {
    case_id: string;
    case_sha256: string;
    model: string;
    cited_records: number;
    validator: "passed" | "passed_with_added_findings";
  };
}

export type TurnOutcome =
  | { kind: "answer"; answer: Answer; ms: number }
  | { kind: "failed_checks"; codes: string[] }
  | { kind: "offline"; reason: string; retryAfter: number | null }
  | { kind: "error"; code: string; status: number }
  | { kind: "network" };

export type Turn =
  | { role: "user"; text: string; at: number }
  | { role: "syrus"; outcome: TurnOutcome; at: number };
