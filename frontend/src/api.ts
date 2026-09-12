import type {
  DetectionEvidence,
  EvidenceSummary,
  OptimizationRequest,
  OptimizationResponse,
  TwinRequest,
  TwinResponse,
} from "./types";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;

    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // Keep HTTP status text.
    }

    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function getEvidenceSummary() {
  return request<EvidenceSummary>("/evidence/summary");
}

export function getDetectionEvidence() {
  return request<DetectionEvidence>("/evidence/detection");
}

export function simulateTwin(payload: TwinRequest) {
  return request<TwinResponse>("/twin/simulate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function recommendOptimization(
  payload: OptimizationRequest,
) {
  return request<OptimizationResponse>(
    "/optimization/recommend",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}
