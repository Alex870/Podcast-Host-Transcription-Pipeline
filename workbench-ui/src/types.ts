export type EpisodeSummary = {
  episode_id: string;
  episode_name: string;
  episode_date?: string;
  host_detected?: boolean;
  segment_count?: number;
  has_reviewed?: boolean;
  load_error?: string;
};

export type TranscriptSegment = {
  id: number;
  start?: number;
  end?: number;
  speaker: string;
  text: string;
  original_text?: string | null;
  llm_reviewed_text?: string | null;
};

export type Finding = {
  finding_id: string;
  source: string;
  issue_type: string;
  severity: string;
  reason: string;
  segment_ids: number[];
  suggested_text?: string | null;
};

export type EpisodeBundle = {
  episode_id: string;
  cleaned: {
    path: string;
    metadata: Record<string, unknown>;
    segments: TranscriptSegment[];
    host_detected: boolean;
    host_original_speaker_id?: string | null;
    speaker_mapping: Record<string, string>;
    speaker_durations_seconds: Record<string, number>;
  };
  reviewed: {
    present: boolean;
    path: string;
    metadata: Record<string, unknown>;
    segments: TranscriptSegment[];
  };
  manifest: Record<string, unknown>;
  summary_row: Record<string, unknown>;
  review_run_report: Record<string, unknown>;
  speaker_workflow_report: Record<string, unknown>;
  deterministic_findings: Finding[];
  semantic_scan?: {
    findings?: Finding[];
  };
};

export type SessionInfo = {
  sessionOpen: boolean;
  projectRoot?: string;
  outputDir?: string;
};

export type LearnedRule = {
  rule_id: string;
  status: string;
  activation_status: string;
  rule_family: string;
  stage_target: string;
  summary: string;
  explanation: string;
  confidence: number;
  ambiguity_notes: string[];
  validation: Record<string, unknown>;
  source_examples: Array<Record<string, unknown>>;
};

export type TeachMeProposal = {
  status: string;
  session_id: string;
  episode_id: string;
  segment_id: number;
  source_example: Record<string, unknown>;
  rule_candidate: LearnedRule;
};
