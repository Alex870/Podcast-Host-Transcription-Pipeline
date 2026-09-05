import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  applyPreferredTerm,
  applyReplacement,
  applyTextCorrection,
  acceptEvaluationBaseline,
  approveReviewRule,
  backfillReviewRule,
  disableReviewRule,
  getSession,
  getPartition,
  initializeEvaluationCampaign,
  listEpisodes,
  listEpisodeCorrections,
  listReviewRules,
  loadAudit,
  loadEpisode,
  loadEvaluationCampaignProposal,
  loadEvaluationQueues,
  loadSpeakerIdentities,
  loadSpeakerWorkflow,
  listPartitions,
  openSession,
  createPartition,
  scanPartition,
  archivePartition,
  updatePartition,
  validatePartition,
  mergeSpeakerIdentities,
  previewPreferredTerm,
  previewReplacement,
  previewTextCorrection,
  promoteSpeakerCandidate,
  proposeTeachMeRule,
  rejectReviewRule,
  rerunReviewRule,
  rollbackTextCorrection,
  rollbackSpeakerLibrary,
  runScan,
  saveGoldSegmentAnnotation,
  speakerEvidenceAudioUrl,
  splitSpeakerIdentity,
} from "./api";
import type { EpisodeBundle, Finding, LearnedRule, PartitionRecord, TeachMeProposal, TranscriptSegment } from "./types";

const LOCAL_STORAGE_PROJECT_KEY = "podcast-workbench-project-root";
const LOCAL_STORAGE_OUTPUT_KEY = "podcast-workbench-output-dir";
const LOCAL_STORAGE_PARTITION_KEY = "podcast-workbench-partition-id";
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

function transcriptRows(bundle: EpisodeBundle | undefined, mode: "cleaned" | "compare" | "changed" | "speaker") {
  if (!bundle) {
    return [];
  }
  const reviewedById = new Map(bundle.reviewed.segments.map((segment) => [segment.id, segment]));
  const rows = bundle.cleaned.segments.map((segment) => ({
    ...segment,
    reviewedText: reviewedById.get(segment.id)?.text ?? "",
    changed: bundle.reviewed.present && reviewedById.get(segment.id)?.text && reviewedById.get(segment.id)?.text !== segment.text,
    compareMode: mode === "compare",
  }));
  if (mode === "changed") return rows.filter((row) => row.changed);
  if (mode === "speaker") return rows.filter((row) => !["", "HOST"].includes(String(row.speaker || "").toUpperCase()));
  return rows;
}

export default function App() {
  const queryClient = useQueryClient();
  const [projectRoot, setProjectRoot] = useState(localStorage.getItem(LOCAL_STORAGE_PROJECT_KEY) ?? "");
  const [outputDir, setOutputDir] = useState(localStorage.getItem(LOCAL_STORAGE_OUTPUT_KEY) ?? "");
  const [selectedPartitionId, setSelectedPartitionId] = useState(localStorage.getItem(LOCAL_STORAGE_PARTITION_KEY) ?? "");
  const [newPartitionName, setNewPartitionName] = useState("");
  const [newPartitionContext, setNewPartitionContext] = useState("podcast");
  const [newPartitionIntake, setNewPartitionIntake] = useState("");
  const [newPartitionOutput, setNewPartitionOutput] = useState("");
  const [partitionIntakeForEdit, setPartitionIntakeForEdit] = useState("");
  const [partitionOutputForEdit, setPartitionOutputForEdit] = useState("");
  const [partitionSpeakerReferences, setPartitionSpeakerReferences] = useState("");
  const [partitionCorrections, setPartitionCorrections] = useState("");
  const [partitionReviewBackend, setPartitionReviewBackend] = useState("");
  const [partitionReviewModel, setPartitionReviewModel] = useState("");
  const [partitionReviewReasoning, setPartitionReviewReasoning] = useState("");
  const [partitionPreferredTerms, setPartitionPreferredTerms] = useState("");
  const [partitionReplacementMap, setPartitionReplacementMap] = useState("");
  const [partitionCorpusId, setPartitionCorpusId] = useState("");
  const [partitionDownstreamProject, setPartitionDownstreamProject] = useState("");
  const [partitionValidationMessage, setPartitionValidationMessage] = useState("");
  const [selectedEpisodeId, setSelectedEpisodeId] = useState("");
  const [viewMode, setViewMode] = useState<"cleaned" | "compare" | "changed" | "speaker">(
    (localStorage.getItem(LOCAL_STORAGE_VIEW_KEY) as "cleaned" | "compare" | "changed" | "speaker") || "compare",
  );
  const [activeFinding, setActiveFinding] = useState<Finding | null>(null);
  const [selectedTranscriptSegmentId, setSelectedTranscriptSegmentId] = useState<number | null>(null);
  const [correctionText, setCorrectionText] = useState("");
  const [teachMeText, setTeachMeText] = useState("");
  const [teachMeProposal, setTeachMeProposal] = useState<TeachMeProposal | null>(null);
  const [preferredTermInput, setPreferredTermInput] = useState("");
  const [replacementPreferred, setReplacementPreferred] = useState("");
  const [replacementAlias, setReplacementAlias] = useState("");
  const [selectedRuleId, setSelectedRuleId] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [goldReferenceText, setGoldReferenceText] = useState("");
  const [goldReferenceSpeaker, setGoldReferenceSpeaker] = useState("");
  const [goldTags, setGoldTags] = useState("");
  const [goldNotes, setGoldNotes] = useState("");
  const [goldReviewerId, setGoldReviewerId] = useState("");
  const [goldApprovalStatus, setGoldApprovalStatus] = useState("pending_review");
  const [speakerCandidateName, setSpeakerCandidateName] = useState("");
  const [speakerCandidateRole, setSpeakerCandidateRole] = useState("guest");
  const [speakerMergeIds, setSpeakerMergeIds] = useState("");
  const [speakerSplitId, setSpeakerSplitId] = useState("");
  const [speakerSplitEvidenceIds, setSpeakerSplitEvidenceIds] = useState("");
  const [speakerSplitName, setSpeakerSplitName] = useState("");

  const sessionQuery = useQuery({
    queryKey: ["session"],
    queryFn: getSession,
  });

  const partitionsQuery = useQuery({
    queryKey: ["partitions", projectRoot],
    queryFn: () => listPartitions(projectRoot),
    enabled: Boolean(projectRoot),
  });

  const selectedPartition = useMemo(
    () => partitionsQuery.data?.partitions.find((partition) => partition.partition_id === selectedPartitionId),
    [partitionsQuery.data, selectedPartitionId],
  );

  const partitionDetailQuery = useQuery({
    queryKey: ["partition", selectedPartitionId],
    queryFn: () => getPartition(selectedPartitionId),
    enabled: Boolean(selectedPartitionId) && !selectedPartition?.archived,
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
    if (sessionQuery.data.partitionId) {
      setSelectedPartitionId(sessionQuery.data.partitionId);
      localStorage.setItem(LOCAL_STORAGE_PARTITION_KEY, sessionQuery.data.partitionId);
    }
    setStatusMessage((current) => current || "Workbench session opened from launcher defaults.");
  }, [sessionQuery.data]);

  const openSessionMutation = useMutation({
    mutationFn: () => openSession(
      projectRoot,
      selectedPartitionId
        ? (partitionsQuery.data?.partitions.find((partition) => partition.partition_id === selectedPartitionId)?.output_dir ?? outputDir)
        : outputDir,
      selectedPartitionId || undefined,
    ),
    onSuccess: async () => {
      localStorage.setItem(LOCAL_STORAGE_PROJECT_KEY, projectRoot);
      localStorage.setItem(LOCAL_STORAGE_OUTPUT_KEY, outputDir);
      if (selectedPartitionId) {
        localStorage.setItem(LOCAL_STORAGE_PARTITION_KEY, selectedPartitionId);
      }
      await queryClient.invalidateQueries({ queryKey: ["session"] });
      await queryClient.invalidateQueries({ queryKey: ["episodes"] });
      setStatusMessage("Workbench session opened.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const createPartitionMutation = useMutation({
    mutationFn: () => createPartition({
      projectRoot,
      displayName: newPartitionName,
      contextType: newPartitionContext,
      intakeDir: newPartitionIntake || undefined,
      outputDir: newPartitionOutput || undefined,
    }),
    onSuccess: async (payload) => {
      setNewPartitionName("");
      setNewPartitionIntake("");
      setNewPartitionOutput("");
      setSelectedPartitionId(payload.partition.partition_id);
      await queryClient.invalidateQueries({ queryKey: ["partitions", projectRoot] });
      setStatusMessage(`Created processing space: ${payload.partition.display_name}`);
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const scanPartitionMutation = useMutation({
    mutationFn: () => scanPartition(selectedPartitionId),
    onSuccess: async (payload) => {
      await queryClient.invalidateQueries({ queryKey: ["partitions", projectRoot] });
      setStatusMessage(`Scanned intake: ${Object.entries(payload.counts).map(([key, value]) => `${key}=${value}`).join(", ") || "no audio files"}.`);
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const archivePartitionMutation = useMutation({
    mutationFn: () => archivePartition(selectedPartitionId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["partitions", projectRoot] });
      setSelectedPartitionId("");
      setStatusMessage("Processing space archived. Existing data was not moved or deleted.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const validatePartitionMutation = useMutation({
    mutationFn: () => validatePartition(selectedPartitionId),
    onSuccess: (payload) => {
      setPartitionValidationMessage(payload.valid ? "Space is valid." : `Missing paths: ${payload.missingPaths.join(", ")}`);
      setStatusMessage(payload.valid ? "Processing-space validation passed." : "Processing-space validation found missing paths.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const reactivatePartitionMutation = useMutation({
    mutationFn: () => updatePartition(selectedPartitionId, { archived: false }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["partitions", projectRoot] });
      setStatusMessage("Processing space reactivated.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const updatePartitionMutation = useMutation({
    mutationFn: () => updatePartition(selectedPartitionId, {
      intakeDir: partitionIntakeForEdit || undefined,
      outputDir: partitionOutputForEdit || undefined,
      speakerReferenceDir: partitionSpeakerReferences || undefined,
      correctionsDir: partitionCorrections || undefined,
      configOverrides: {
        ...(selectedPartition?.config_overrides ?? {}),
        backend: partitionReviewBackend || undefined,
        review_model_name: partitionReviewModel || undefined,
        review_reasoning_effort: partitionReviewReasoning || undefined,
        preferred_terms_file: partitionPreferredTerms || undefined,
        replacement_map_json: partitionReplacementMap || undefined,
      },
      downstreamConfig: {
        ...(selectedPartition?.downstream_config ?? {}),
        corpus_id: partitionCorpusId || undefined,
        project: partitionDownstreamProject || undefined,
      },
    }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["partitions", projectRoot] });
      setStatusMessage("Processing-space settings saved.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  useEffect(() => {
    if (selectedPartitionId || !partitionsQuery.data?.partitions?.length) {
      return;
    }
    const first = partitionsQuery.data.partitions.find((partition) => !partition.archived);
    if (first) {
      setSelectedPartitionId(first.partition_id);
    }
  }, [partitionsQuery.data, selectedPartitionId]);

  useEffect(() => {
    if (!selectedPartition) {
      return;
    }
    setPartitionIntakeForEdit(selectedPartition.intake_dir);
    setPartitionOutputForEdit(selectedPartition.output_dir);
    setPartitionSpeakerReferences(selectedPartition.speaker_reference_dir ?? "");
    setPartitionCorrections(selectedPartition.corrections_dir ?? "");
    const overrides = selectedPartition.config_overrides ?? {};
    setPartitionReviewBackend(String(overrides.backend ?? ""));
    setPartitionReviewModel(String(overrides.review_model_name ?? ""));
    setPartitionReviewReasoning(String(overrides.review_reasoning_effort ?? ""));
    setPartitionPreferredTerms(String(overrides.preferred_terms_file ?? ""));
    setPartitionReplacementMap(String(overrides.replacement_map_json ?? ""));
    const downstream = selectedPartition.downstream_config ?? {};
    setPartitionCorpusId(String(downstream.corpus_id ?? ""));
    setPartitionDownstreamProject(String(downstream.project ?? ""));
  }, [selectedPartition]);

  const episodesQuery = useQuery({
    queryKey: ["episodes"],
    queryFn: listEpisodes,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const evaluationQueuesQuery = useQuery({
    queryKey: ["evaluation-queues"],
    queryFn: loadEvaluationQueues,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const evaluationCampaignQuery = useQuery({
    queryKey: ["evaluation-campaign"],
    queryFn: loadEvaluationCampaignProposal,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const initializeCampaignMutation = useMutation({
    mutationFn: initializeEvaluationCampaign,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["evaluation-queues"] }),
        queryClient.invalidateQueries({ queryKey: ["evaluation-campaign"] }),
      ]);
      setStatusMessage("Guided 12-episode evaluation campaign initialized.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const acceptBaselineMutation = useMutation({
    mutationFn: () => acceptEvaluationBaseline(goldReviewerId),
    onSuccess: (payload) => setStatusMessage(`Evaluation baseline accepted: ${payload.path}`),
    onError: (error: Error) => setStatusMessage(error.message),
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

  const correctionHistoryQuery = useQuery({
    queryKey: ["correction-history", selectedEpisodeId],
    queryFn: () => listEpisodeCorrections(selectedEpisodeId),
    enabled: Boolean(selectedEpisodeId),
  });

  const speakerIdentitiesQuery = useQuery({
    queryKey: ["speaker-identities"],
    queryFn: loadSpeakerIdentities,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const auditQuery = useQuery({
    queryKey: ["audit"],
    queryFn: loadAudit,
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const speakerWorkflowQuery = useQuery({
    queryKey: ["speakerWorkflow"],
    queryFn: () => loadSpeakerWorkflow("all"),
    enabled: sessionQuery.data?.sessionOpen === true,
  });

  const rulesQuery = useQuery({
    queryKey: ["reviewRules"],
    queryFn: listReviewRules,
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
    mutationFn: (payload: { segmentId: number; correctedText: string }) =>
      applyTextCorrection(selectedEpisodeId, payload.segmentId, payload.correctedText, episodeQuery.data?.cleaned.source_revision),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["correction-history", selectedEpisodeId] }),
      ]);
      setStatusMessage("Text correction applied.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const rollbackCorrectionMutation = useMutation({
    mutationFn: (correctionId: string) => rollbackTextCorrection(selectedEpisodeId, correctionId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["correction-history", selectedEpisodeId] }),
      ]);
      setStatusMessage("Correction rolled back.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const promoteSpeakerMutation = useMutation({
    mutationFn: (candidateId: string) =>
      promoteSpeakerCandidate(candidateId, speakerCandidateName, [speakerCandidateRole], [], goldReviewerId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["speaker-identities"] }),
        queryClient.invalidateQueries({ queryKey: ["speakerWorkflow"] }),
      ]);
      setSpeakerCandidateName("");
      setStatusMessage("Speaker identity promoted.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const mergeSpeakerMutation = useMutation({
    mutationFn: () => mergeSpeakerIdentities(
      speakerMergeIds.split(",").map((item) => item.trim()).filter(Boolean),
      goldReviewerId,
    ),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["speaker-identities"] });
      setStatusMessage("Speaker identities merged.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const splitSpeakerMutation = useMutation({
    mutationFn: () => splitSpeakerIdentity(
      speakerSplitId.trim(),
      speakerSplitEvidenceIds.split(",").map((item) => item.trim()).filter(Boolean),
      speakerSplitName.trim(),
      goldReviewerId,
    ),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["speaker-identities"] });
      setStatusMessage("Speaker identity split.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const rollbackSpeakerMutation = useMutation({
    mutationFn: () => rollbackSpeakerLibrary(goldReviewerId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["speaker-identities"] });
      setStatusMessage("Speaker identity library rolled back.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const goldAnnotationMutation = useMutation({
    mutationFn: () => {
      if (!activeSegment) {
        throw new Error("Select a transcript segment first.");
      }
      return saveGoldSegmentAnnotation(
        selectedEpisodeId,
        activeSegment.id,
        goldReferenceText,
        goldReferenceSpeaker,
        goldTags.split(",").map((item) => item.trim()).filter(Boolean),
        goldNotes,
        goldReviewerId,
        goldApprovalStatus,
      );
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["episode", selectedEpisodeId] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["evaluation-queues"] }),
      ]);
      setStatusMessage("Gold-set reference annotation saved.");
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

  const teachMeMutation = useMutation({
    mutationFn: (payload: { segmentId: number; desiredReviewedText: string; supersedesRuleId?: string }) =>
      proposeTeachMeRule(selectedEpisodeId, payload.segmentId, payload.desiredReviewedText, payload.supersedesRuleId ?? ""),
    onSuccess: async (payload) => {
      setTeachMeProposal(payload);
      setSelectedRuleId(payload.rule_candidate.rule_id);
      await queryClient.invalidateQueries({ queryKey: ["reviewRules"] });
      setStatusMessage("Generalized rule proposal ready for review.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const approveRuleMutation = useMutation({
    mutationFn: (ruleId: string) => approveReviewRule(ruleId, selectedEpisodeId),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["reviewRules"] }),
        queryClient.invalidateQueries({ queryKey: ["episode", selectedEpisodeId] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["episodes"] }),
      ]);
      setTeachMeProposal(null);
      setStatusMessage(`Rule approved and current episode rerun: ${payload.rule.rule_id}`);
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const rejectRuleMutation = useMutation({
    mutationFn: (ruleId: string) => rejectReviewRule(ruleId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["reviewRules"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
      setTeachMeProposal(null);
      setStatusMessage("Rule proposal rejected.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const disableRuleMutation = useMutation({
    mutationFn: (ruleId: string) => disableReviewRule(ruleId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["reviewRules"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
      setStatusMessage("Rule disabled.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const rerunRuleMutation = useMutation({
    mutationFn: (ruleId: string) => rerunReviewRule(ruleId, selectedEpisodeId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["episode", selectedEpisodeId] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["episodes"] }),
      ]);
      setStatusMessage("Current episode rerun with approved learned rules.");
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const backfillRuleMutation = useMutation({
    mutationFn: (ruleId: string) => backfillReviewRule(ruleId),
    onSuccess: async (payload) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["episode", selectedEpisodeId] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["episodes"] }),
        queryClient.invalidateQueries({ queryKey: ["reviewRules"] }),
      ]);
      setStatusMessage(`Backfill complete across ${payload.episode_count} episode(s).`);
    },
    onError: (error: Error) => setStatusMessage(error.message),
  });

  const rows = useMemo(() => transcriptRows(episodeQuery.data, viewMode), [episodeQuery.data, viewMode]);
  const semanticFindings = (episodeQuery.data?.semantic_scan?.findings as Finding[] | undefined) ?? [];
  const allFindings = [...(episodeQuery.data?.deterministic_findings ?? []), ...semanticFindings];
  const selectedRule: LearnedRule | undefined = rulesQuery.data?.rules.find((rule) => rule.rule_id === selectedRuleId);

  useEffect(() => {
    localStorage.setItem(LOCAL_STORAGE_VIEW_KEY, viewMode);
  }, [viewMode]);

  const activeSegment = useMemo<TranscriptSegment | undefined>(() => {
    const segmentId = selectedTranscriptSegmentId ?? activeFinding?.segment_ids?.[0];
    return episodeQuery.data?.cleaned.segments.find((segment) => segment.id === segmentId);
  }, [activeFinding, episodeQuery.data, selectedTranscriptSegmentId]);

  useEffect(() => {
    setCorrectionText(activeFinding?.suggested_text ?? "");
  }, [activeFinding]);

  useEffect(() => {
    if (!activeSegment) {
      setTeachMeText("");
      return;
    }
    const reviewedSegment = episodeQuery.data?.reviewed.segments.find((segment) => segment.id === activeSegment.id);
    setTeachMeText(reviewedSegment?.text ?? activeSegment.text);
  }, [activeSegment, episodeQuery.data]);

  useEffect(() => {
    if (!activeSegment) {
      setGoldReferenceText("");
      setGoldReferenceSpeaker("");
      return;
    }
    const annotated = episodeQuery.data?.gold_annotation?.segments.find((segment) => segment.id === activeSegment.id);
    setGoldReferenceText(annotated?.text ?? activeSegment.text);
    setGoldReferenceSpeaker(annotated?.speaker ?? activeSegment.speaker);
    setGoldReviewerId(String(episodeQuery.data?.gold_annotation?.annotation_metadata?.reviewer_id ?? ""));
    setGoldApprovalStatus(String(episodeQuery.data?.gold_annotation?.annotation_metadata?.approval_status ?? "pending_review"));
  }, [activeSegment, episodeQuery.data]);

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
            Processing space
            <select value={selectedPartitionId} onChange={(event) => setSelectedPartitionId(event.target.value)}>
              <option value="">Legacy output folder</option>
              {(partitionsQuery.data?.partitions ?? []).map((partition: PartitionRecord) => (
                <option key={partition.partition_id} value={partition.partition_id}>
                  {partition.display_name}{partition.archived ? " (archived)" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Output folder
            <input value={selectedPartitionId ? (partitionsQuery.data?.partitions.find((partition) => partition.partition_id === selectedPartitionId)?.output_dir ?? outputDir) : outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="C:\\path\\to\\output" disabled={Boolean(selectedPartitionId)} />
          </label>
          <button onClick={() => openSessionMutation.mutate()} disabled={openSessionMutation.isPending || Boolean(selectedPartition?.archived)}>
            Open session
          </button>
          <button onClick={() => scanPartitionMutation.mutate()} disabled={!selectedPartitionId || Boolean(selectedPartition?.archived) || !sessionQuery.data?.sessionOpen || scanPartitionMutation.isPending}>
            Scan intake
          </button>
        </div>
        <div className="field-row">
          <label>
            New space name
            <input value={newPartitionName} onChange={(event) => setNewPartitionName(event.target.value)} placeholder="Work meetings" />
          </label>
          <label>
            Context
            <select value={newPartitionContext} onChange={(event) => setNewPartitionContext(event.target.value)}>
              <option value="podcast">Podcast</option>
              <option value="meeting">Work meeting</option>
              <option value="custom">Custom</option>
            </select>
          </label>
          <label>
            Intake folder (optional)
            <input value={newPartitionIntake} onChange={(event) => setNewPartitionIntake(event.target.value)} placeholder="Managed default" />
          </label>
          <label>
            Output folder (optional)
            <input value={newPartitionOutput} onChange={(event) => setNewPartitionOutput(event.target.value)} placeholder="Managed default" />
          </label>
          <button onClick={() => createPartitionMutation.mutate()} disabled={!projectRoot || !newPartitionName.trim() || createPartitionMutation.isPending}>
            Create space
          </button>
          <button onClick={() => validatePartitionMutation.mutate()} disabled={!selectedPartitionId || validatePartitionMutation.isPending}>
            Validate selected
          </button>
          {selectedPartition?.archived ? (
            <button onClick={() => reactivatePartitionMutation.mutate()} disabled={reactivatePartitionMutation.isPending}>
              Reactivate selected
            </button>
          ) : null}
          <button onClick={() => archivePartitionMutation.mutate()} disabled={!selectedPartitionId || Boolean(selectedPartition?.archived) || archivePartitionMutation.isPending}>
            Archive selected
          </button>
        </div>
        {selectedPartition ? (
          <div className="field-row">
            <label>
              Selected intake
              <input value={partitionIntakeForEdit} onChange={(event) => setPartitionIntakeForEdit(event.target.value)} />
            </label>
            <label>
              Selected output
              <input value={partitionOutputForEdit} onChange={(event) => setPartitionOutputForEdit(event.target.value)} />
            </label>
            <label>
              Speaker references
              <input value={partitionSpeakerReferences} onChange={(event) => setPartitionSpeakerReferences(event.target.value)} placeholder="Optional" />
            </label>
            <label>
              Corrections folder
              <input value={partitionCorrections} onChange={(event) => setPartitionCorrections(event.target.value)} />
            </label>
            <label>
              Review backend
              <input value={partitionReviewBackend} onChange={(event) => setPartitionReviewBackend(event.target.value)} placeholder="Inherited" />
            </label>
            <label>
              Review model
              <input value={partitionReviewModel} onChange={(event) => setPartitionReviewModel(event.target.value)} placeholder="Inherited" />
            </label>
            <label>
              Reasoning effort
              <select value={partitionReviewReasoning} onChange={(event) => setPartitionReviewReasoning(event.target.value)}>
                <option value="">Inherited</option>
                <option value="none">None</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="xhigh">Xhigh</option>
              </select>
            </label>
            <label>
              Preferred terms file
              <input value={partitionPreferredTerms} onChange={(event) => setPartitionPreferredTerms(event.target.value)} placeholder="Inherited" />
            </label>
            <label>
              Replacement map
              <input value={partitionReplacementMap} onChange={(event) => setPartitionReplacementMap(event.target.value)} placeholder="Inherited" />
            </label>
            <label>
              Corpus ID
              <input value={partitionCorpusId} onChange={(event) => setPartitionCorpusId(event.target.value)} placeholder="Partition default" />
            </label>
            <label>
              Downstream project
              <input value={partitionDownstreamProject} onChange={(event) => setPartitionDownstreamProject(event.target.value)} placeholder="Optional" />
            </label>
            <button onClick={() => updatePartitionMutation.mutate()} disabled={updatePartitionMutation.isPending}>
              Save selected settings
            </button>
            <span className="status-line">
              Status: {Object.entries(selectedPartition.status_counts ?? {}).map(([key, value]) => `${key}=${value}`).join(", ") || "not scanned"}
              {partitionValidationMessage ? ` — ${partitionValidationMessage}` : ""}
            </span>
            {partitionDetailQuery.data?.effectiveConfig ? (
              <details>
                <summary>Effective inherited settings</summary>
                <pre>{JSON.stringify(partitionDetailQuery.data.effectiveConfig, null, 2)}</pre>
              </details>
            ) : null}
          </div>
        ) : null}
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
              <button className={viewMode === "changed" ? "active" : ""} onClick={() => setViewMode("changed")}>Changed</button>
              <button className={viewMode === "speaker" ? "active" : ""} onClick={() => setViewMode("speaker")}>Speakers</button>
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
                  <tr
                    key={row.id}
                    className={`${row.changed ? "changed-row" : ""} ${selectedTranscriptSegmentId === row.id ? "selected-row" : ""}`}
                    onClick={() => {
                      setSelectedTranscriptSegmentId(row.id);
                      setActiveFinding(null);
                    }}
                  >
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

          <section className="panel metadata-panel">
            <div className="panel-header"><h2>Speaker workflow</h2></div>
            <div className="meta-grid">
              <div><strong>Evidence rows</strong><span>{speakerWorkflowQuery.data?.row_count ?? 0}</span></div>
              <div><strong>Changed rows</strong><span>{speakerWorkflowQuery.data?.changed_count ?? 0}</span></div>
              <div><strong>Recurring unknowns</strong><span>{speakerWorkflowQuery.data?.recurring_unknown_speakers?.length ?? 0}</span></div>
            </div>
            <div className="hint-text">
              Candidates are grouped from compatible voice embeddings, never from reused SPEAKER_ labels.
            </div>
          </section>

          <section className="panel metadata-panel">
            <div className="panel-header"><h2>Evaluation campaign</h2></div>
            <div className="meta-grid">
              <div><strong>Unlabelled</strong><span>{evaluationQueuesQuery.data?.counts?.unlabelled ?? 0}</span></div>
              <div><strong>Pending</strong><span>{evaluationQueuesQuery.data?.counts?.pending_review ?? 0}</span></div>
              <div><strong>Adjudication</strong><span>{evaluationQueuesQuery.data?.counts?.adjudication_required ?? 0}</span></div>
              <div><strong>Approved</strong><span>{evaluationQueuesQuery.data?.counts?.human_approved ?? 0}</span></div>
              <div><strong>Campaign sample</strong><span>{evaluationCampaignQuery.data?.selected?.length ?? 0}/12</span></div>
            </div>
            <div className="hint-text">{evaluationQueuesQuery.data?.evaluation_pack_path ?? "Evaluation pack not resolved."}</div>
            <div className="button-row">
              <button
                disabled={(evaluationCampaignQuery.data?.selected?.length ?? 0) < 12}
                onClick={() => initializeCampaignMutation.mutate()}
              >
                Initialize campaign
              </button>
              <button
                disabled={(evaluationQueuesQuery.data?.counts?.human_approved ?? 0) < 12}
                onClick={() => acceptBaselineMutation.mutate()}
              >
                Accept benchmark baseline
              </button>
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
              <div className="audit-list">
                {(correctionHistoryQuery.data?.corrections ?? []).slice().reverse().map((item) => (
                  <div className="audit-item" key={String(item.correction_id)}>
                    <strong>{String(item.status ?? "unknown")} | {String(item.correction_kind ?? "text")}</strong>
                    <span>{String(item.after ?? "")}</span>
                    {item.status === "approved" ? (
                      <button onClick={() => rollbackCorrectionMutation.mutate(String(item.correction_id))}>Rollback</button>
                    ) : null}
                  </div>
                ))}
              </div>
              <input
                value={speakerMergeIds}
                onChange={(event) => setSpeakerMergeIds(event.target.value)}
                placeholder="Speaker IDs to merge, comma separated"
              />
              <button
                disabled={speakerMergeIds.split(",").filter((item) => item.trim()).length < 2}
                onClick={() => mergeSpeakerMutation.mutate()}
              >
                Merge identities
              </button>
              <input value={speakerSplitId} onChange={(event) => setSpeakerSplitId(event.target.value)} placeholder="Speaker ID to split" />
              <input
                value={speakerSplitEvidenceIds}
                onChange={(event) => setSpeakerSplitEvidenceIds(event.target.value)}
                placeholder="Evidence IDs to move, comma separated"
              />
              <input value={speakerSplitName} onChange={(event) => setSpeakerSplitName(event.target.value)} placeholder="New identity name" />
              <div className="button-row">
                <button
                  disabled={!speakerSplitId.trim() || !speakerSplitEvidenceIds.trim() || !speakerSplitName.trim()}
                  onClick={() => splitSpeakerMutation.mutate()}
                >
                  Split identity
                </button>
                <button onClick={() => rollbackSpeakerMutation.mutate()}>Rollback last identity change</button>
              </div>
            </div>

            <div className="action-block">
              <h3>Speaker identities</h3>
              <div className="hint-text">
                Promote only candidates that meet the cross-episode evidence threshold. Host and co-host roles may both be assigned.
              </div>
              <input
                value={speakerCandidateName}
                onChange={(event) => setSpeakerCandidateName(event.target.value)}
                placeholder="Speaker display name"
              />
              <select value={speakerCandidateRole} onChange={(event) => setSpeakerCandidateRole(event.target.value)}>
                <option value="guest">Guest</option>
                <option value="co-host">Co-host</option>
                <option value="host">Host</option>
              </select>
              <div className="audit-list">
                {(speakerIdentitiesQuery.data?.workflow?.recurring_unknown_speakers ?? []).map((candidate) => (
                  <div className="audit-item" key={String(candidate.candidate_id)}>
                    <strong>{String(candidate.episode_count)} episodes | {String(candidate.total_duration_seconds)} seconds</strong>
                    <span>{String(candidate.embedding_family)}</span>
                    {(() => {
                      const clips = Array.isArray(candidate.evidence_clips)
                        ? candidate.evidence_clips as Array<Record<string, unknown>>
                        : [];
                      const clip = clips[0];
                      const spans = Array.isArray(clip?.spans)
                        ? clip.spans as Array<Record<string, unknown>>
                        : [];
                      const evidenceId = String(clip?.evidence_id ?? "");
                      const start = Number(spans[0]?.start ?? 0);
                      return evidenceId ? (
                        <audio
                          controls
                          preload="none"
                          src={speakerEvidenceAudioUrl(evidenceId)}
                          onLoadedMetadata={(event) => {
                            event.currentTarget.currentTime = start;
                          }}
                        />
                      ) : null;
                    })()}
                    <button
                      disabled={!speakerCandidateName.trim() || candidate.promotion_eligible !== true}
                      onClick={() => promoteSpeakerMutation.mutate(String(candidate.candidate_id))}
                    >
                      Promote
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="action-block">
              <h3>Teach me</h3>
              <div className="hint-text">Teach the review layer how to make this kind of reviewed-text revision in future runs.</div>
              <div className="action-meta">
                Segment: {activeSegment?.id ?? "—"} {activeSegment ? `(${activeSegment.speaker})` : ""}
              </div>
              <textarea
                value={teachMeText}
                onChange={(event) => setTeachMeText(event.target.value)}
                placeholder="Desired reviewed text for this segment"
                rows={4}
              />
              <div className="button-row">
                <button
                  onClick={() => activeSegment && teachMeMutation.mutate({ segmentId: activeSegment.id, desiredReviewedText: teachMeText })}
                  disabled={!activeSegment || !teachMeText.trim()}
                >
                  Teach me from this edit
                </button>
                <button
                  onClick={() =>
                    activeSegment &&
                    selectedRule &&
                    teachMeMutation.mutate({
                      segmentId: activeSegment.id,
                      desiredReviewedText: teachMeText,
                      supersedesRuleId: selectedRule.rule_id,
                    })
                  }
                  disabled={!activeSegment || !teachMeText.trim() || !selectedRule}
                >
                  Refine rule from this edit
                </button>
              </div>
              {teachMeProposal ? (
                <div className="teach-me-proposal">
                  <strong>{teachMeProposal.rule_candidate.summary || "Untitled learned rule"}</strong>
                  <div>{teachMeProposal.rule_candidate.rule_family} | {teachMeProposal.rule_candidate.stage_target}</div>
                  <div>Confidence: {teachMeProposal.rule_candidate.confidence.toFixed(2)}</div>
                  <div>{teachMeProposal.rule_candidate.explanation || "No explanation provided."}</div>
                  <div className="hint-text">
                    Validation: {String((teachMeProposal.rule_candidate.validation as Record<string, unknown>)?.pass ?? false)} |{" "}
                    Warnings: {((teachMeProposal.rule_candidate.validation as Record<string, unknown>)?.warnings as string[] | undefined)?.join(", ") || "none"}
                  </div>
                  <div className="button-row">
                    <button onClick={() => approveRuleMutation.mutate(teachMeProposal.rule_candidate.rule_id)}>
                      Approve rule
                    </button>
                    <button onClick={() => rejectRuleMutation.mutate(teachMeProposal.rule_candidate.rule_id)}>
                      Reject rule
                    </button>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="action-block">
              <h3>Gold-set reference</h3>
              <div className="hint-text">
                Save a human-approved text and speaker reference for full-pipeline quality benchmarking.
              </div>
              <div className="action-meta">
                Segment: {activeSegment?.id ?? "—"} | {episodeQuery.data?.gold_annotation?.present ? "Reference exists" : "Not annotated"}
              </div>
              <textarea
                value={goldReferenceText}
                onChange={(event) => setGoldReferenceText(event.target.value)}
                placeholder="Human-approved reference text"
                rows={4}
              />
              <input
                value={goldReferenceSpeaker}
                onChange={(event) => setGoldReferenceSpeaker(event.target.value)}
                placeholder="Reference speaker label"
              />
              <input
                value={goldReviewerId}
                onChange={(event) => setGoldReviewerId(event.target.value)}
                placeholder="Reviewer pseudonym"
              />
              <select value={goldApprovalStatus} onChange={(event) => setGoldApprovalStatus(event.target.value)}>
                <option value="pending_review">Pending review</option>
                <option value="human_approved">Human approved</option>
                <option value="adjudication_required">Adjudication required</option>
              </select>
              <input
                value={goldTags}
                onChange={(event) => setGoldTags(event.target.value)}
                placeholder="Tags, comma separated (crosstalk, noise, short-turn)"
              />
              <textarea
                value={goldNotes}
                onChange={(event) => setGoldNotes(event.target.value)}
                placeholder="Annotation notes"
                rows={2}
              />
              <div className="button-row">
                <button
                  onClick={() => goldAnnotationMutation.mutate()}
                  disabled={!activeSegment || !goldReferenceText.trim() || !goldReferenceSpeaker.trim() || goldAnnotationMutation.isPending}
                >
                  Save reference
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

            <div className="action-block">
              <h3>Learned rules</h3>
              <div className="audit-list">
                {(rulesQuery.data?.rules ?? []).map((rule) => (
                  <button
                    key={rule.rule_id}
                    className={`finding-item ${selectedRuleId === rule.rule_id ? "selected" : ""}`}
                    onClick={() => setSelectedRuleId(rule.rule_id)}
                  >
                    <div className="finding-row">
                      <strong>{rule.summary || rule.rule_id}</strong>
                      <span>{rule.status}</span>
                    </div>
                    <div className="finding-meta">{rule.rule_family} | {rule.stage_target}</div>
                  </button>
                ))}
              </div>
              {selectedRule ? (
                <div className="teach-me-proposal">
                  <strong>{selectedRule.summary || selectedRule.rule_id}</strong>
                  <div>{selectedRule.rule_family} | {selectedRule.stage_target}</div>
                  <div>{selectedRule.explanation || "No explanation provided."}</div>
                  <div className="button-row">
                    <button
                      onClick={() => selectedRule && approveRuleMutation.mutate(selectedRule.rule_id)}
                      disabled={selectedRule.status === "approved" || !selectedEpisodeId}
                    >
                      Approve rule
                    </button>
                    <button onClick={() => selectedRule && disableRuleMutation.mutate(selectedRule.rule_id)}>
                      Disable rule
                    </button>
                    <button
                      onClick={() => selectedRule && rerunRuleMutation.mutate(selectedRule.rule_id)}
                      disabled={selectedRule.status !== "approved" || !selectedEpisodeId}
                    >
                      Re-run current episode
                    </button>
                    <button
                      onClick={() => selectedRule && backfillRuleMutation.mutate(selectedRule.rule_id)}
                      disabled={selectedRule.status !== "approved"}
                    >
                      Backfill other episodes
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
