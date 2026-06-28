import type { EpisodeBundle, SessionInfo } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(String(payload.detail ?? response.statusText));
  }
  return response.json() as Promise<T>;
}

export function getSession() {
  return request<SessionInfo>("/api/session");
}

export function openSession(projectRoot: string, outputDir: string) {
  return request<{ status: string; projectRoot: string; outputDir: string }>("/api/session/open", {
    method: "POST",
    body: JSON.stringify({ projectRoot, outputDir }),
  });
}

export function listEpisodes() {
  return request<{ episodes: Array<{ episode_id: string; episode_name: string; episode_date?: string; segment_count?: number; has_reviewed?: boolean; load_error?: string }> }>("/api/episodes");
}

export function loadEpisode(episodeId: string) {
  return request<EpisodeBundle>(`/api/episodes/${encodeURIComponent(episodeId)}`);
}

export function runScan(episodeId: string, force = false) {
  return request<{ findings: unknown[] }>(`/api/episodes/${encodeURIComponent(episodeId)}/scan?force=${force ? "true" : "false"}`, {
    method: "POST",
  });
}

export function previewTextCorrection(episodeId: string, segmentId: number, correctedText: string) {
  return request<{ original_text: string; corrected_text: string; changes: Array<{ field: string; before: string; after: string }> }>(
    `/api/episodes/${encodeURIComponent(episodeId)}/text-corrections/preview`,
    {
      method: "POST",
      body: JSON.stringify({ segmentId, correctedText }),
    },
  );
}

export function applyTextCorrection(episodeId: string, segmentId: number, correctedText: string) {
  return request<{ status: string }>(`/api/episodes/${encodeURIComponent(episodeId)}/text-corrections/apply`, {
    method: "POST",
    body: JSON.stringify({ segmentId, correctedText }),
  });
}

export function previewPreferredTerm(term: string) {
  return request<{ target_path: string; term: string; already_present: boolean; line_will_be_added: string | null }>(
    "/api/glossary/preferred-terms/preview",
    {
      method: "POST",
      body: JSON.stringify({ term }),
    },
  );
}

export function applyPreferredTerm(term: string) {
  return request<{ status: string }>("/api/glossary/preferred-terms/apply", {
    method: "POST",
    body: JSON.stringify({ term }),
  });
}

export function previewReplacement(preferredTerm: string, alias: string) {
  return request<{ target_path: string; preferred_term: string; alias: string; already_present: boolean; updated_aliases: string[] }>(
    "/api/glossary/replacements/preview",
    {
      method: "POST",
      body: JSON.stringify({ preferredTerm, alias }),
    },
  );
}

export function applyReplacement(preferredTerm: string, alias: string) {
  return request<{ status: string }>("/api/glossary/replacements/apply", {
    method: "POST",
    body: JSON.stringify({ preferredTerm, alias }),
  });
}

export function loadAudit() {
  return request<{ entries: Array<Record<string, unknown>> }>("/api/audit");
}
