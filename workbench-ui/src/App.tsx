import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  applyPreferredTerm,
  applyReplacement,
  applyTextCorrection,
  getSession,
  listEpisodes,
  loadAudit,
  loadEpisode,
  openSession,
  previewPreferredTerm,
  previewReplacement,
  previewTextCorrection,
  runScan,
} from "./api";
import type { EpisodeBundle, Finding, TranscriptSegment } from "./types";

const LOCAL_STORAGE_PROJECT_KEY = "podcast-workbench-project-root";
const LOCAL_STORAGE_OUTPUT_KEY = "podcast-workbench-output-dir";
const LOCAL_STORAGE_VIEW_KEY = "podcast-workbench-view-mode";

function formatTimestamp(value: number | undefined) {
  if (value === undefined || value === null) {
    return "";
  }
  const whole = Math.max(0, Math.floor(value));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const seconds = whole % 60;
  return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function transcriptRows(bundle: EpisodeBundle | undefined, mode: "cleaned" | "compare") {
  if (!bundle) {
    return [];
  }
  const reviewedById = new Map(bundle.reviewed.segments.map((segment) => [segment.id, segment]));
  return bundle.cleaned.segments.map((segment) => ({
    ...segment,
    reviewedText: reviewedById.get(segment.id)?.text ?? "",
    changed: bundle.reviewed.present && reviewedById.get(segment.id)?.text && reviewedById.get(segment.id)?.text !== segment.text,
    compareMode: mode === "compare",
  }));
}

export default function App() {
  const queryClient = useQueryClient();
  const [projectRoot, setProjectRoot] = useState(localStorage.getItem(LOCAL_STORAGE_PROJECT_KEY) ?? "");
  const [outputDir, setOutputDir] = useState(localStorage.getItem(LOCAL_STORAGE_OUTPUT_KEY) ?? "");
  const [selectedEpisodeId, setSelectedEpisodeId] = useState("");
  const [viewMode, setViewMode] = useState<"cleaned" | "compare">(
    (localStorage.getItem(LOCAL_STORAGE_VIEW_KEY) as "cleaned" | "compare") || "compare",
  );
  const [activeFinding, setActiveFinding] = useState<Finding | null>(null);
  const [correctionText, setCorrectionText] = useState("");
  const [preferredTermInput, setPreferredTermInput] = useState("");
  const [replacementPreferred, setReplacementPreferred] = useState("");
  const [replacementAlias, setReplacementAlias] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  const sessionQuery = useQuery({
    queryKey: ["session"],
    queryFn: getSession,
  });

  useEffect(() => {
    if (!sessionQuery.data?.sessionOpen) {
      return;
    }
    if (sessionQuery.data.projectRoot) {
      setProjectRoot(sessionQuery.data.projectRoot);
      localStorage.setItem(LOCAL_STORAGE_PROJECT_KEY, sessionQuery.data.projectRoot);
    }
    if (sessionQuery.data.outputDir) {
      setOutputDir(sessionQuery.data.outputDir);
      localStorage.setItem(LOCAL_STORAGE_OUTPUT_KEY, sessionQuery.data.outputDir);
    }
    setStatusMessage((current) => current || "Workbench session opened from launcher defaults.");
  }, [sessionQuery.data]);

  const openSessionMutation = useMutation({
    mutationFn: () => openSession(projectRoot, outputDir),
    onSuccess: async () => {
      localStorage.setItem(LOCAL_STORAGE_PROJECT_KEY, projectRoot);
      localStorage.setItem(LOCAL_STORAGE_OUTPUT_KEY, outputDir);
      await queryClient.invalidateQueries({ queryKey: ["session"] });
      await queryClient.invalidateQueries({ queryKey: ["episodes"] });
      setStatusMessage("Workbench session opened.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const episodesQuery = useQuery({
    queryKey: ["episodes"],
    queryFn: listEpisodes,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  useEffect(() => {
    const firstEpisode = episodesQuery.data?.episodes?.[0]?.episode_id;
    if (!selectedEpisodeId && firstEpisode) {
      setSelectedEpisodeId(firstEpisode);
    }
  }, [episodesQuery.data, selectedEpisodeId]);

  const episodeQuery = useQuery({
    queryKey: ["episode", selectedEpisodeId],
    queryFn: () => loadEpisode(selectedEpisodeId),
    enabled: Boolean(selectedEpisodeId),
  });

  const auditQuery = useQuery({
    queryKey: ["audit"],
    queryFn: loadAudit,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const scanMutation = useMutation({
    mutationFn: (force: boolean) => runScan(selectedEpisodeId, force),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["episode", selectedEpisodeId] });
      setStatusMessage("Semantic scan complete.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const previewCorrectionMutation = useMutation({
    mutationFn: (payload: { segmentId: number; correctedText: string }) => previewTextCorrection(selectedEpisodeId, payload.segmentId, payload.correctedText),
    onSuccess: (payload) => {
      setStatusMessage(`Preview ready: ${payload.original_text} -> ${payload.corrected_text}`);
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const applyCorrectionMutation = useMutation({
    mutationFn: (payload: { segmentId: number; correctedText: string }) => applyTextCorrection(selectedEpisodeId, payload.segmentId, payload.correctedText),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
      setStatusMessage("Text correction applied.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const previewPreferredMutation = useMutation({
    mutationFn: () => previewPreferredTerm(preferredTermInput),
    onSuccess: (payload) => {
      setStatusMessage(
        payload.already_present
          ? "Preferred term is already present."
          : `Preview ready: add '${payload.term}' to preferred terms.`,
      );
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const applyPreferredMutation = useMutation({
    mutationFn: () => applyPreferredTerm(preferredTermInput),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["audit"] });
      setStatusMessage("Preferred term applied.");
      setPreferredTermInput("");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const previewReplacementMutation = useMutation({
    mutationFn: () => previewReplacement(replacementPreferred, replacementAlias),
    onSuccess: (payload) => {
      setStatusMessage(
        payload.already_present
          ? "Replacement alias is already present."
          : `Preview ready: add alias '${payload.alias}' for '${payload.preferred_term}'.`,
      );
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const applyReplacementMutation = useMutation({
    mutationFn: () => applyReplacement(replacementPreferred, replacementAlias),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["audit"] });
      setStatusMessage("Replacement map updated.");
      setReplacementAlias("");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const rows = useMemo(() => transcriptRows(episodeQuery.data, viewMode), [episodeQuery.data, viewMode]);
  const semanticFindings = (episodeQuery.data?.semantic_scan?.findings as Finding[] | undefined) ?? [];
  const allFindings = [...(episodeQuery.data?.deterministic_findings ?? []), ...semanticFindings];

  useEffect(() => {
    localStorage.setItem(LOCAL_STORAGE_VIEW_KEY, viewMode);
  }, [viewMode]);

  const activeSegment = useMemo<TranscriptSegment | undefined>(() => {
    const segmentId = activeFinding?.segment_ids?.[0];
    return episodeQuery.data?.cleaned.segments.find((segment) => segment.id === segmentId);
  }, [activeFinding, episodeQuery.data]);

  useEffect(() => {
    setCorrectionText(activeFinding?.suggested_text ?? "");
  }, [activeFinding]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Transcript Review Workbench</h1>
          <p>Episode-first review of cleaned and reviewed transcript artifacts.</p>
        </div>
        <div className="topbar-actions">
          <button onClick={() => scanMutation.mutate(false)} disabled={!selectedEpisodeId || scanMutation.isPending}>
            Run semantic scan
          </button>
          <button onClick={() => scanMutation.mutate(true)} disabled={!selectedEpisodeId || scanMutation.isPending}>
            Refresh scan
          </button>
        </div>
      </header>

      <section className="setup-panel">
        <div className="field-row">
          <label>
            Project root
            <input value={projectRoot} onChange={(event) => setProjectRoot(event.target.value)} placeholder="C:\\path\\to\\podcast-host-transcription-pipeline" />
          </label>
          <label>
            Output folder
            <input value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="C:\\path\\to\\output" />
          </label>
          <button onClick={() => openSessionMutation.mutate()} disabled={openSessionMutation.isPending}>
            Open session
          </button>
        </div>
        <div className="status-line">{statusMessage || (sessionQuery.data?.sessionOpen ? "Session ready." : "Open a project root and output folder to begin.")}</div>
      </section>

      <main className="main-grid">
        <aside className="episode-list-panel panel">
          <div className="panel-header">
            <h2>Episodes</h2>
            <span>{episodesQuery.data?.episodes?.length ?? 0}</span>
          </div>
          <div className="episode-list">
            {(episodesQuery.data?.episodes ?? []).map((episode) => (
              <button
                key={episode.episode_id}
                className={`episode-list-item ${selectedEpisodeId === episode.episode_id ? "selected" : ""}`}
                onClick={() => {
                  setSelectedEpisodeId(episode.episode_id);
                  setActiveFinding(null);
                }}
              >
                <div className="episode-name">{episode.episode_name}</div>
                <div className="episode-meta">
                  <span>{episode.episode_date || "No date"}</span>
                  <span>{episode.segment_count ?? 0} seg</span>
                  <span>{episode.has_reviewed ? "Reviewed" : "Cleaned only"}</span>
                </div>
                {episode.load_error ? <div className="error-text">{episode.load_error}</div> : null}
              </button>
            ))}
          </div>
        </aside>

        <section className="transcript-panel panel">
          <div className="panel-header">
            <h2>Transcript</h2>
            <div className="segmented-control">
              <button className={viewMode === "cleaned" ? "active" : ""} onClick={() => setViewMode("cleaned")}>Cleaned</button>
              <button className={viewMode === "compare" ? "active" : ""} onClick={() => setViewMode("compare")}>Compare</button>
            </div>
          </div>
          <div className="transcript-table-wrap">
            <table className="transcript-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Speaker</th>
                  <th>Cleaned</th>
                  {viewMode === "compare" ? <th>Reviewed</th> : null}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id} className={row.changed ? "changed-row" : ""}>
                    <td>{formatTimestamp(row.start)} - {formatTimestamp(row.end)}</td>
                    <td>{row.speaker || "UNKNOWN"}</td>
                    <td>{row.text}</td>
                    {viewMode === "compare" ? <td>{row.reviewedText || "—"}</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="inspector-panel">
          <section className="panel metadata-panel">
            <div className="panel-header"><h2>Metadata</h2></div>
            <div className="meta-grid">
              <div><strong>Episode</strong><span>{episodeQuery.data?.episode_id ?? "—"}</span></div>
              <div><strong>Cleaned path</strong><span>{episodeQuery.data?.cleaned.path ?? "—"}</span></div>
              <div><strong>Reviewed</strong><span>{episodeQuery.data?.reviewed.present ? "Present" : "Absent"}</span></div>
              <div><strong>Review status</strong><span>{String(episodeQuery.data?.summary_row?.review_status ?? "—")}</span></div>
              <div><strong>Review stages</strong><span>{String(episodeQuery.data?.summary_row?.review_completed_stages ?? "—")}</span></div>
              <div><strong>Host detected</strong><span>{episodeQuery.data?.cleaned.host_detected ? "Yes" : "No"}</span></div>
            </div>
          </section>

          <section className="panel findings-panel">
            <div className="panel-header"><h2>Findings</h2><span>{allFindings.length}</span></div>
            <div className="findings-list">
              {allFindings.map((finding) => (
                <button
                  key={finding.finding_id}
                  className={`finding-item ${activeFinding?.finding_id === finding.finding_id ? "selected" : ""}`}
                  onClick={() => setActiveFinding(finding)}
                >
                  <div className="finding-row">
                    <strong>{finding.issue_type}</strong>
                    <span>{finding.severity}</span>
                  </div>
                  <div className="finding-reason">{finding.reason}</div>
                  <div className="finding-meta">{finding.source} | segments {finding.segment_ids.join(", ")}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="panel actions-panel">
            <div className="panel-header"><h2>Actions</h2></div>
            <div className="action-block">
              <h3>Text correction</h3>
              <div className="hint-text">Anchor approved transcript fixes to cleaned text and write them into episode correction CSVs.</div>
              <div className="action-meta">
                Segment: {activeSegment?.id ?? "—"} {activeSegment ? `(${activeSegment.speaker})` : ""}
              </div>
              <textarea
                value={correctionText}
                onChange={(event) => setCorrectionText(event.target.value)}
                placeholder="Suggested corrected text"
                rows={4}
              />
              <div className="button-row">
                <button
                  onClick={() => activeSegment && previewCorrectionMutation.mutate({ segmentId: activeSegment.id, correctedText: correctionText })}
                  disabled={!activeSegment || !correctionText.trim()}
                >
                  Preview
                </button>
                <button
                  onClick={() => activeSegment && applyCorrectionMutation.mutate({ segmentId: activeSegment.id, correctedText: correctionText })}
                  disabled={!activeSegment || !correctionText.trim()}
                >
                  Apply
                </button>
              </div>
            </div>

            <div className="action-block">
              <h3>Preferred term</h3>
              <input value={preferredTermInput} onChange={(event) => setPreferredTermInput(event.target.value)} placeholder="Preferred term" />
              <div className="button-row">
                <button onClick={() => previewPreferredMutation.mutate()} disabled={!preferredTermInput.trim()}>Preview</button>
                <button onClick={() => applyPreferredMutation.mutate()} disabled={!preferredTermInput.trim()}>Apply</button>
              </div>
            </div>

            <div className="action-block">
              <h3>Replacement map</h3>
              <input value={replacementPreferred} onChange={(event) => setReplacementPreferred(event.target.value)} placeholder="Preferred term" />
              <input value={replacementAlias} onChange={(event) => setReplacementAlias(event.target.value)} placeholder="Alias to replace" />
              <div className="button-row">
                <button onClick={() => previewReplacementMutation.mutate()} disabled={!replacementPreferred.trim() || !replacementAlias.trim()}>Preview</button>
                <button onClick={() => applyReplacementMutation.mutate()} disabled={!replacementPreferred.trim() || !replacementAlias.trim()}>Apply</button>
              </div>
            </div>

            <div className="action-block">
              <h3>Recent audit entries</h3>
              <div className="audit-list">
                {(auditQuery.data?.entries ?? []).slice().reverse().slice(0, 8).map((entry, index) => (
                  <div className="audit-item" key={index}>
                    <strong>{String(entry.action ?? "action")}</strong>
                    <span>{String(entry.episode_id ?? entry.target_path ?? "")}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
