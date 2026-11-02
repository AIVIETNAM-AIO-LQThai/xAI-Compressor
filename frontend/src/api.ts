import type {
  DetectionEvidence,
  EvidenceSummary,
  OptimizationRequest,
  OptimizationResponse,
  ProofBundleResponse,
  ProofBundleVerification,
  ProofManifestResponse,
  TwinRequest,
  TwinResponse,
} from "./types";

// Same origin by default (`uvicorn backend.web:app`, or the Vite proxy in development). VITE_API_BASE_URL is for a separate API host and must never hold a secret.
const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? "";

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

export function getProofManifest() {
  return request<ProofManifestResponse>(
    "/evidence/proof-manifest",
  );
}

export function getProofBundle() {
  return request<ProofBundleResponse>(
    "/evidence/proof-bundle",
  );
}

export function verifyProofBundle(
  bundle: ProofBundleResponse,
) {
  return request<ProofBundleVerification>(
    "/evidence/proof-bundle/verify",
    {
      method: "POST",
      body: JSON.stringify(bundle),
    },
  );
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
