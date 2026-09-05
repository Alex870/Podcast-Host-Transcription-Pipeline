"""FastAPI app for the transcript review workbench."""

from __future__ import annotations

import argparse
import json
import os
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from podcast_transcribe.workbench_core import (
    approve_review_rule,
    accept_evaluation_baseline,
    apply_preferred_term_addition,
    apply_replacement_map_update,
    apply_text_correction,
    backfill_review_rule,
    disable_review_rule,
    discover_episode_bundles,
    evaluation_queues,
    initialize_quality_campaign,
    get_review_rule,
    list_review_rules,
    list_episode_corrections,
    load_audit_log,
    load_episode_bundle,
    propose_teach_me_rule,
    propose_quality_campaign,
    reject_review_rule,
    rerun_review_with_approved_rules,
    rollback_text_correction,
    resolve_workbench_paths,
    run_semantic_scan,
    save_gold_segment_annotation,
)
from podcast_transcribe.speaker_workflow import (
    build_cross_episode_speaker_view,
    collect_speaker_evidence,
    load_speaker_library,
    merge_speaker_identities,
    promote_speaker_candidate,
    rollback_speaker_library,
    split_speaker_identity,
)
from podcast_transcribe.speakers import (
    approve_speaker_profile_promotion,
    rollback_speaker_profile_promotion,
    stage_speaker_profile_promotion,
)
from podcast_transcribe.operations import (
    apply_retention,
    campaign_preflight,
    downstream_delivery_status,
    retry_downstream_delivery,
)
from podcast_transcribe.partitions import (
    CONTEXT_TYPES,
    PartitionError,
    PartitionRegistry,
    resolve_partition_context,
)


class SessionOpenRequest(BaseModel):
    project_root: str = Field(..., alias="projectRoot")
    output_dir: str = Field("", alias="outputDir")
    partition_id: Optional[str] = Field(default=None, alias="partitionId")


class PartitionCreateRequest(BaseModel):
    project_root: str = Field(..., alias="projectRoot")
    display_name: str = Field(..., alias="displayName")
    context_type: str = Field("podcast", alias="contextType")
    workflow_profile: Optional[str] = Field(default=None, alias="workflowProfile")
    intake_dir: Optional[str] = Field(default=None, alias="intakeDir")
    output_dir: Optional[str] = Field(default=None, alias="outputDir")
    state_dir: Optional[str] = Field(default=None, alias="stateDir")
    speaker_reference_dir: Optional[str] = Field(default=None, alias="speakerReferenceDir")
    corrections_dir: Optional[str] = Field(default=None, alias="correctionsDir")
    config_overrides: Dict[str, object] = Field(default_factory=dict, alias="configOverrides")
    downstream_config: Dict[str, object] = Field(default_factory=dict, alias="downstreamConfig")


class PartitionUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, alias="displayName")
    context_type: Optional[str] = Field(default=None, alias="contextType")
    workflow_profile: Optional[str] = Field(default=None, alias="workflowProfile")
    intake_dir: Optional[str] = Field(default=None, alias="intakeDir")
    output_dir: Optional[str] = Field(default=None, alias="outputDir")
    state_dir: Optional[str] = Field(default=None, alias="stateDir")
    speaker_reference_dir: Optional[str] = Field(default=None, alias="speakerReferenceDir")
    corrections_dir: Optional[str] = Field(default=None, alias="correctionsDir")
    config_overrides: Optional[Dict[str, object]] = Field(default=None, alias="configOverrides")
    downstream_config: Optional[Dict[str, object]] = Field(default=None, alias="downstreamConfig")
    archived: Optional[bool] = None


class TextCorrectionRequest(BaseModel):
    segment_id: int = Field(..., alias="segmentId")
    corrected_text: str = Field(..., alias="correctedText")
    expected_revision: Optional[Dict[str, object]] = Field(default=None, alias="expectedRevision")


class CorrectionRollbackRequest(BaseModel):
    correction_id: str = Field(..., alias="correctionId")


class TeachMeRequest(BaseModel):
    segment_id: int = Field(..., alias="segmentId")
    desired_reviewed_text: str = Field(..., alias="desiredReviewedText")
    supersedes_rule_id: str = Field("", alias="supersedesRuleId")


class RuleEpisodeRequest(BaseModel):
    episode_id: str = Field(..., alias="episodeId")


class PreferredTermRequest(BaseModel):
    term: str


class ReplacementMapRequest(BaseModel):
    preferred_term: str = Field(..., alias="preferredTerm")
    alias: str


class GoldSegmentAnnotationRequest(BaseModel):
    segment_id: int = Field(..., alias="segmentId")
    reference_text: str = Field(..., alias="referenceText")
    reference_speaker: str = Field(..., alias="referenceSpeaker")
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    reviewer_id: str = Field("", alias="reviewerId")
    approval_status: str = Field("pending_review", alias="approvalStatus")


class SpeakerProfilePromotionRequest(BaseModel):
    profile_path: str = Field(..., alias="profilePath")
    candidate_profile: Dict[str, object] = Field(default_factory=dict, alias="candidateProfile")
    evaluation_report: Dict[str, object] = Field(default_factory=dict, alias="evaluationReport")
    reviewer_id: str = Field("", alias="reviewerId")


class SpeakerCandidatePromotionRequest(BaseModel):
    candidate_id: str = Field(..., alias="candidateId")
    display_name: str = Field(..., alias="displayName")
    roles: List[str] = Field(default_factory=lambda: ["guest"])
    aliases: List[str] = Field(default_factory=list)
    reviewer_id: str = Field("", alias="reviewerId")


class SpeakerMergeRequest(BaseModel):
    speaker_ids: List[str] = Field(..., alias="speakerIds")
    reviewer_id: str = Field("", alias="reviewerId")


class SpeakerSplitRequest(BaseModel):
    speaker_id: str = Field(..., alias="speakerId")
    evidence_ids: List[str] = Field(..., alias="evidenceIds")
    display_name: str = Field(..., alias="displayName")
    reviewer_id: str = Field("", alias="reviewerId")


class ReviewerApprovalRequest(BaseModel):
    reviewer_id: str = Field("", alias="reviewerId")


class RetentionRequest(BaseModel):
    categories: List[str] = Field(default_factory=list)
    older_than_days: int = Field(30, alias="olderThanDays", ge=0)
    dry_run: bool = Field(True, alias="dryRun")


def _frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "workbench-ui" / "dist"


class WorkbenchSession:
    def __init__(self):
        self.project_root: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.partition_id: Optional[str] = None
        self.partition_context = None

    def open(self, project_root: str, output_dir: str = "", partition_id: Optional[str] = None):
        project = Path(project_root).resolve()
        output = Path(output_dir).resolve()
        if not project.exists():
            raise RuntimeError(f"Project root does not exist: {project}")
        if partition_id:
            context = resolve_partition_context(project, partition_id)
            output = context.output_dir
            self.partition_context = context
            self.partition_id = context.partition_id
        else:
            self.partition_context = None
            self.partition_id = None
        if not output.exists():
            raise RuntimeError(f"Output directory does not exist: {output}")
        self.project_root = project
        self.output_dir = output
        resolve_workbench_paths(project, output)

    def require(self) -> tuple[Path, Path]:
        if self.project_root is None or self.output_dir is None:
            raise RuntimeError("Workbench session is not open yet.")
        return self.project_root, self.output_dir

    def require_registry(self) -> tuple[Path, PartitionRegistry]:
        if self.project_root is None:
            raise RuntimeError("Workbench project is not open yet.")
        return self.project_root, PartitionRegistry(self.project_root)


SESSION = WorkbenchSession()

app = FastAPI(title="Podcast Transcript Review Workbench", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _redact_config(payload: Dict[str, object]) -> Dict[str, object]:
    """Keep credentials out of the browser's effective-configuration view."""

    result: Dict[str, object] = {}
    for key, value in payload.items():
        folded = str(key).lower()
        if any(fragment in folded for fragment in ("token", "secret", "password", "api_key", "apikey")):
            result[key] = "[configured]" if value not in (None, "") else ""
        elif isinstance(value, dict):
            result[key] = _redact_config(value)
        else:
            result[key] = value
    return result


@app.get("/api/health")
def health():
    dist_dir = _frontend_dist_dir()
    return {
        "status": "ok",
        "session_open": SESSION.project_root is not None and SESSION.output_dir is not None,
        "partition_id": SESSION.partition_id,
        "frontend_dist_present": dist_dir.exists(),
    }


@app.post("/api/session/open")
def open_session(payload: SessionOpenRequest):
    try:
        SESSION.open(payload.project_root, payload.output_dir, payload.partition_id)
        project_root, output_dir = SESSION.require()
        return {
            "status": "ok",
            "projectRoot": str(project_root),
            "outputDir": str(output_dir),
            "partitionId": SESSION.partition_id,
        }
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/session")
def get_session():
    if SESSION.project_root is None or SESSION.output_dir is None:
        return {"sessionOpen": False}
    return {
        "sessionOpen": True,
        "projectRoot": str(SESSION.project_root),
        "outputDir": str(SESSION.output_dir),
        "partitionId": SESSION.partition_id,
        "partition": SESSION.partition_context.record.to_dict() if SESSION.partition_context else None,
    }


@app.get("/api/partitions")
def list_partitions(project_root: Optional[str] = None, include_archived: bool = True):
    try:
        root = Path(project_root).resolve() if project_root else SESSION.project_root
        if root is None:
            raise RuntimeError("Project root is required before listing processing spaces.")
        registry = PartitionRegistry(root)
        rows = []
        for record in registry.list(include_archived=include_archived):
            summary = registry.summary(record.partition_id, include_archived=True)
            rows.append({**record.to_dict(), "status_counts": summary.get("counts", {})})
        return {"partitions": rows, "projectRoot": str(root), "contextTypes": sorted(CONTEXT_TYPES)}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/partitions")
def create_partition(payload: PartitionCreateRequest):
    try:
        registry = PartitionRegistry(payload.project_root)
        record = registry.create(
            payload.display_name,
            context_type=payload.context_type,
            workflow_profile=payload.workflow_profile,
            intake_dir=payload.intake_dir,
            output_dir=payload.output_dir,
            state_dir=payload.state_dir,
            speaker_reference_dir=payload.speaker_reference_dir,
            corrections_dir=payload.corrections_dir,
            config_overrides=payload.config_overrides,
            downstream_config=payload.downstream_config,
        )
        return {"status": "created", "partition": record.to_dict()}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/partitions/{partition_id}")
def get_partition(partition_id: str):
    try:
        project_root, registry = SESSION.require_registry()
        record = registry.get(partition_id, include_archived=True)
        summary = registry.summary(partition_id, include_archived=True)
        context = resolve_partition_context(project_root, partition_id)
        return {"partition": record.to_dict(), "summary": summary, "effectiveConfig": _redact_config(context.effective_config)}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.put("/api/partitions/{partition_id}")
def update_partition(partition_id: str, payload: PartitionUpdateRequest):
    try:
        _project_root, registry = SESSION.require_registry()
        changes = payload.model_dump(exclude_none=True, by_alias=False)
        record = registry.update(partition_id, **changes)
        return {"status": "updated", "partition": record.to_dict()}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/partitions/{partition_id}/validate")
def validate_partition(partition_id: str):
    try:
        _project_root, registry = SESSION.require_registry()
        record = registry.get(partition_id, include_archived=True)
        summary = registry.summary(partition_id, include_archived=True)
        missing = [str(path) for path in (record.intake_dir, record.output_dir, record.state_dir) if not path.exists()]
        return {"valid": not missing and not record.archived, "archived": record.archived, "missingPaths": missing, "summary": summary}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/partitions/{partition_id}/scan")
def scan_partition(partition_id: str):
    try:
        _project_root, registry = SESSION.require_registry()
        return registry.scan(partition_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/partitions/{partition_id}/adopt")
def adopt_partition(partition_id: str, payload: PartitionUpdateRequest):
    """Adopt existing folders into a registered processing space without moving data."""
    try:
        _project_root, registry = SESSION.require_registry()
        changes = payload.model_dump(exclude_none=True, by_alias=False)
        if not any(changes.get(key) for key in ("intake_dir", "output_dir", "state_dir")):
            raise PartitionError("Adoption requires at least an existing intake, output, or state path.")
        record = registry.update(partition_id, **changes)
        summary = registry.scan(partition_id)
        return {"status": "adopted", "partition": record.to_dict(), "summary": summary}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/partitions/{partition_id}/archive")
def archive_partition(partition_id: str):
    try:
        _project_root, registry = SESSION.require_registry()
        record = registry.update(partition_id, archived=True)
        return {"status": "archived", "partition": record.to_dict()}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/episodes")
def list_episodes():
    try:
        project_root, output_dir = SESSION.require()
        return {
            "episodes": discover_episode_bundles(output_dir),
            "projectRoot": str(project_root),
            "outputDir": str(output_dir),
        }
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/evaluation/queues")
def evaluation_queue_endpoint():
    try:
        project_root, output_dir = SESSION.require()
        return evaluation_queues(project_root, output_dir)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/operations/preflight")
def operations_preflight_endpoint():
    try:
        project_root, output_dir = SESSION.require()
        return campaign_preflight(project_root, output_dir)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/operations/downstream")
def downstream_status_endpoint():
    try:
        _, output_dir = SESSION.require()
        return downstream_delivery_status(output_dir)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/operations/downstream/{correction_set_id}/retry")
def downstream_retry_endpoint(correction_set_id: str):
    try:
        project_root, output_dir = SESSION.require()
        return retry_downstream_delivery(project_root, output_dir, correction_set_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/operations/retention")
def retention_endpoint(payload: RetentionRequest):
    try:
        _, output_dir = SESSION.require()
        return apply_retention(output_dir, payload.model_dump(by_alias=False), dry_run=payload.dry_run)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/evaluation/campaign/proposal")
def evaluation_campaign_proposal_endpoint():
    try:
        project_root, output_dir = SESSION.require()
        return propose_quality_campaign(project_root, output_dir)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/evaluation/campaign/initialize")
def evaluation_campaign_initialize_endpoint():
    try:
        project_root, output_dir = SESSION.require()
        return initialize_quality_campaign(project_root, output_dir)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/evaluation/baseline/accept")
def accept_evaluation_baseline_endpoint(payload: ReviewerApprovalRequest):
    try:
        project_root, output_dir = SESSION.require()
        return accept_evaluation_baseline(project_root, output_dir, payload.reviewer_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/episodes/{episode_id}")
def get_episode(episode_id: str):
    try:
        project_root, output_dir = SESSION.require()
        return load_episode_bundle(project_root, output_dir, episode_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/episodes/{episode_id}/gold-annotation")
def save_gold_annotation_endpoint(episode_id: str, payload: GoldSegmentAnnotationRequest):
    try:
        project_root, output_dir = SESSION.require()
        return save_gold_segment_annotation(
            project_root,
            output_dir,
            episode_id,
            payload.segment_id,
            payload.reference_text,
            payload.reference_speaker,
            tags=payload.tags,
            notes=payload.notes,
            reviewer_id=payload.reviewer_id,
            approval_status=payload.approval_status,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/episodes/{episode_id}/scan")
def scan_episode(episode_id: str, force: bool = False):
    try:
        project_root, output_dir = SESSION.require()
        return run_semantic_scan(project_root, output_dir, episode_id, force=force)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/episodes/{episode_id}/scan")
def get_scan(episode_id: str):
    try:
        project_root, output_dir = SESSION.require()
        bundle = load_episode_bundle(project_root, output_dir, episode_id)
        return bundle.get("semantic_scan") or {"episode_id": episode_id, "findings": []}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/episodes/{episode_id}/text-corrections/preview")
def preview_text_correction_endpoint(episode_id: str, payload: TextCorrectionRequest):
    try:
        from podcast_transcribe.workbench_core import preview_text_correction

        project_root, output_dir = SESSION.require()
        return preview_text_correction(project_root, output_dir, episode_id, payload.segment_id, payload.corrected_text)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/episodes/{episode_id}/text-corrections/apply")
def apply_text_correction_endpoint(episode_id: str, payload: TextCorrectionRequest):
    try:
        project_root, output_dir = SESSION.require()
        return apply_text_correction(
            project_root,
            output_dir,
            episode_id,
            payload.segment_id,
            payload.corrected_text,
            expected_revision=payload.expected_revision,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/episodes/{episode_id}/text-corrections")
def list_text_corrections_endpoint(episode_id: str):
    try:
        _project_root, output_dir = SESSION.require()
        return list_episode_corrections(output_dir, episode_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/episodes/{episode_id}/text-corrections/rollback")
def rollback_text_correction_endpoint(episode_id: str, payload: CorrectionRollbackRequest):
    try:
        project_root, output_dir = SESSION.require()
        return rollback_text_correction(project_root, output_dir, episode_id, payload.correction_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/glossary/preferred-terms/preview")
def preview_preferred_term_endpoint(payload: PreferredTermRequest):
    try:
        from podcast_transcribe.workbench_core import preview_preferred_term_addition

        project_root, output_dir = SESSION.require()
        return preview_preferred_term_addition(project_root, output_dir, payload.term)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/glossary/preferred-terms/apply")
def apply_preferred_term_endpoint(payload: PreferredTermRequest):
    try:
        project_root, output_dir = SESSION.require()
        return apply_preferred_term_addition(project_root, output_dir, payload.term)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/glossary/replacements/preview")
def preview_replacement_endpoint(payload: ReplacementMapRequest):
    try:
        from podcast_transcribe.workbench_core import preview_replacement_map_update

        project_root, output_dir = SESSION.require()
        return preview_replacement_map_update(project_root, output_dir, payload.preferred_term, payload.alias)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/glossary/replacements/apply")
def apply_replacement_endpoint(payload: ReplacementMapRequest):
    try:
        project_root, output_dir = SESSION.require()
        return apply_replacement_map_update(project_root, output_dir, payload.preferred_term, payload.alias)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/audit")
def get_audit(limit: int = 200):
    try:
        project_root, output_dir = SESSION.require()
        return {"entries": load_audit_log(project_root, output_dir, limit=limit)}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/speaker-workflow")
def speaker_workflow(view: str = "all"):
    try:
        _project_root, output_dir = SESSION.require()
        return build_cross_episode_speaker_view(output_dir, view=view)
    except Exception as exc:
        raise _json_error(exc) from exc


def _speaker_library_path(project_root: Path) -> Path:
    from podcast_transcribe.workbench_core import load_project_config

    if SESSION.partition_context is not None and SESSION.partition_context.record.speaker_reference_dir:
        return SESSION.partition_context.record.speaker_reference_dir / "speakers.json"

    config = load_project_config(project_root)
    configured = str(config.get("known_speakers_dir") or "speaker_reference_samples")
    root = Path(configured)
    if not root.is_absolute():
        root = project_root / root
    resolved = root.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise RuntimeError("Known-speakers directory must remain inside the project root.") from exc
    return resolved / "speakers.json"


@app.get("/api/speaker-identities")
def speaker_identities():
    try:
        project_root, output_dir = SESSION.require()
        return {
            "library": load_speaker_library(_speaker_library_path(project_root)),
            "workflow": build_cross_episode_speaker_view(output_dir),
        }
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/speaker-evidence/{evidence_id}/audio")
def speaker_evidence_audio(evidence_id: str):
    try:
        _project_root, output_dir = SESSION.require()
        evidence = next(
            (
                row.get("identity_evidence")
                for row in collect_speaker_evidence(output_dir)
                if isinstance(row.get("identity_evidence"), dict)
                and str(row["identity_evidence"].get("evidence_id") or "") == evidence_id
            ),
            None,
        )
        if not isinstance(evidence, dict):
            raise RuntimeError(f"Speaker evidence not found: {evidence_id}")
        source_audio = Path(str(evidence.get("source_audio") or "")).resolve()
        if not source_audio.exists() or not source_audio.is_file():
            raise RuntimeError(f"Source audio is unavailable for speaker evidence: {evidence_id}")
        return FileResponse(source_audio)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/speaker-identities/promote")
def promote_speaker_identity(payload: SpeakerCandidatePromotionRequest):
    try:
        project_root, output_dir = SESSION.require()
        workflow = build_cross_episode_speaker_view(output_dir)
        candidate = next(
            (
                item for item in workflow.get("recurring_unknown_speakers") or []
                if str(item.get("candidate_id") or "") == payload.candidate_id
            ),
            None,
        )
        if candidate is None:
            raise RuntimeError(f"Speaker candidate not found: {payload.candidate_id}")
        return promote_speaker_candidate(
            _speaker_library_path(project_root),
            candidate,
            display_name=payload.display_name,
            roles=payload.roles,
            aliases=payload.aliases,
            reviewer_id=payload.reviewer_id,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/speaker-identities/merge")
def merge_speaker_identity_endpoint(payload: SpeakerMergeRequest):
    try:
        project_root, _output_dir = SESSION.require()
        return merge_speaker_identities(
            _speaker_library_path(project_root),
            payload.speaker_ids,
            reviewer_id=payload.reviewer_id,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/speaker-identities/split")
def split_speaker_identity_endpoint(payload: SpeakerSplitRequest):
    try:
        project_root, _output_dir = SESSION.require()
        return split_speaker_identity(
            _speaker_library_path(project_root),
            payload.speaker_id,
            payload.evidence_ids,
            display_name=payload.display_name,
            reviewer_id=payload.reviewer_id,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/speaker-identities/rollback")
def rollback_speaker_identity_endpoint(payload: ReviewerApprovalRequest):
    try:
        project_root, _output_dir = SESSION.require()
        return rollback_speaker_library(
            _speaker_library_path(project_root),
            reviewer_id=payload.reviewer_id,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


def _profile_path(profile_path: str, project_root: Path) -> Path:
    candidate = Path(profile_path).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise RuntimeError("Speaker profile path must remain inside the opened project root.") from exc
    return candidate


@app.post("/api/speaker-profile/stage")
def stage_speaker_profile(payload: SpeakerProfilePromotionRequest):
    try:
        project_root, _output_dir = SESSION.require()
        path = _profile_path(payload.profile_path, project_root)
        staged = stage_speaker_profile_promotion(path, payload.candidate_profile, payload.evaluation_report)
        return {"status": "pending_review", "candidate_path": str(staged)}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/speaker-profile/approve")
def approve_speaker_profile(payload: SpeakerProfilePromotionRequest):
    try:
        project_root, _output_dir = SESSION.require()
        path = _profile_path(payload.profile_path, project_root)
        return approve_speaker_profile_promotion(path, payload.reviewer_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/speaker-profile/rollback")
def rollback_speaker_profile(payload: SpeakerProfilePromotionRequest):
    try:
        project_root, _output_dir = SESSION.require()
        path = _profile_path(payload.profile_path, project_root)
        return rollback_speaker_profile_promotion(path, payload.reviewer_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/episodes/{episode_id}/teach-me/propose")
def propose_teach_me_rule_endpoint(episode_id: str, payload: TeachMeRequest):
    try:
        project_root, output_dir = SESSION.require()
        return propose_teach_me_rule(
            project_root,
            output_dir,
            episode_id,
            payload.segment_id,
            payload.desired_reviewed_text,
            supersedes_rule_id=payload.supersedes_rule_id,
        )
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/review-rules")
def review_rules():
    try:
        project_root, _output_dir = SESSION.require()
        return {"rules": list_review_rules(project_root)}
    except Exception as exc:
        raise _json_error(exc) from exc


@app.get("/api/review-rules/{rule_id}")
def review_rule_detail(rule_id: str):
    try:
        project_root, _output_dir = SESSION.require()
        rule = get_review_rule(project_root, rule_id)
        if rule is None:
            raise RuntimeError(f"Learned rule not found: {rule_id}")
        return rule
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/review-rules/{rule_id}/approve")
def approve_review_rule_endpoint(rule_id: str, payload: RuleEpisodeRequest):
    try:
        project_root, output_dir = SESSION.require()
        return approve_review_rule(project_root, output_dir, rule_id, payload.episode_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/review-rules/{rule_id}/reject")
def reject_review_rule_endpoint(rule_id: str):
    try:
        project_root, output_dir = SESSION.require()
        return reject_review_rule(project_root, output_dir, rule_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/review-rules/{rule_id}/disable")
def disable_review_rule_endpoint(rule_id: str):
    try:
        project_root, output_dir = SESSION.require()
        return disable_review_rule(project_root, output_dir, rule_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/review-rules/{rule_id}/rerun-current-episode")
def rerun_review_rule_endpoint(rule_id: str, payload: RuleEpisodeRequest):
    try:
        project_root, output_dir = SESSION.require()
        return rerun_review_with_approved_rules(project_root, output_dir, payload.episode_id, focus_rule_id=rule_id)
    except Exception as exc:
        raise _json_error(exc) from exc


@app.post("/api/review-rules/{rule_id}/backfill")
def backfill_review_rule_endpoint(rule_id: str):
    try:
        project_root, output_dir = SESSION.require()
        return backfill_review_rule(project_root, output_dir, rule_id)
    except Exception as exc:
        raise _json_error(exc) from exc


frontend_dist = _frontend_dist_dir()


@app.get("/assets/{asset_path:path}")
def frontend_asset(asset_path: str):
    asset_root = frontend_dist / "assets"
    candidate = (asset_root / asset_path).resolve()
    try:
        candidate.relative_to(asset_root.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Static asset path is outside the frontend assets directory.")
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    raise HTTPException(status_code=404, detail="Frontend asset not found.")


@app.get("/")
def root():
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        return FileResponse(frontend_dist / "index.html")
    return JSONResponse(
        {
            "message": "Frontend build not found. Start the Vite dev server or build workbench-ui/dist.",
            "viteDevUrl": "http://127.0.0.1:5173",
        }
    )


@app.get("/{path_name:path}")
def spa_fallback(path_name: str):
    index_path = frontend_dist / "index.html"
    candidate = (frontend_dist / path_name).resolve()
    if frontend_dist.exists() and candidate.exists() and candidate.is_file():
        try:
            candidate.relative_to(frontend_dist.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Static asset path is outside the frontend build directory.")
        return FileResponse(candidate)
    if frontend_dist.exists() and index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend build not found.")


def main():
    parser = argparse.ArgumentParser(description="Run the transcript review workbench API server.")
    parser.add_argument("--host", default=os.getenv("PODCAST_TRANSCRIBE_WORKBENCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PODCAST_TRANSCRIBE_WORKBENCH_PORT", "8765")))
    parser.add_argument("--project-root", default=os.getenv("PODCAST_TRANSCRIBE_WORKBENCH_PROJECT_ROOT", ""))
    parser.add_argument("--output-dir", default=os.getenv("PODCAST_TRANSCRIBE_WORKBENCH_OUTPUT_DIR", ""))
    parser.add_argument("--partition", default=os.getenv("PODCAST_TRANSCRIBE_WORKBENCH_PARTITION", ""))
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    if args.project_root and (args.output_dir or args.partition):
        SESSION.open(args.project_root, args.output_dir, args.partition or None)

    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
