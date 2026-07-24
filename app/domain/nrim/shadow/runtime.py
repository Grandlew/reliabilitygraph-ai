from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Sequence

from app.domain.nrim.baselines.dual_path_episode_gate import (
    DualPathEpisodeGate,
    confirm_observable_impact,
)
from app.domain.nrim.baselines.learned_fusion import learned_fusion_scores
from app.domain.nrim.baselines.shift_sentinels import assess_support
from app.domain.nrim.baselines.temporal_episode_gate import (
    EpisodeRecord,
    GateState,
)

from .bundle import FrozenInferenceBundle
from .contracts import (
    DataQualityState,
    DecisionState,
    PredictionEnvelope,
    RankedCause,
    ServingSnapshot,
    SnapshotMode,
)
from .hashing import canonical_hash
from .replay import ServingFeatureBuilder
from .store import AppendOnlyEvidenceStore


INFRASTRUCTURE_VERSION = "0.7.0"


def _decision(value: GateState) -> DecisionState:
    return DecisionState(value.value)


class ShadowInferenceOrchestrator:
    """Read-only F/S/D shadow inference with immutable evidence writes."""

    source_write_capability = False
    exposes_predictions_to_frontline = False

    def __init__(
        self,
        *,
        bundle: FrozenInferenceBundle,
        store: AppendOnlyEvidenceStore,
    ) -> None:
        self.bundle = bundle
        self.store = store
        self.feature_builder = ServingFeatureBuilder(
            schema=bundle.feature_schema
        )
        if (
            self.feature_builder.schema_hash
            != canonical_hash(
                bundle.feature_schema.model_dump(mode="json")
            )
        ):
            raise RuntimeError("Serving feature schema differs from bundle")

    @staticmethod
    def _metadata(snapshot: ServingSnapshot) -> dict[str, Any]:
        return snapshot.profile.inference_metadata(
            1.0 - snapshot.availability_fraction
        )

    @staticmethod
    def _record(
        *,
        deployment_pseudonym: str,
        windows: Sequence[dict[str, Any]],
    ) -> EpisodeRecord:
        # Truth is deliberately absent from serving inputs. The frozen batch
        # gate's result container expects a truth field, so the adapter adds a
        # constant placeholder only after feature hashing and inference input
        # audits. The placeholder is never persisted as model evidence.
        gate_windows = tuple(
            {
                **window,
                "targets": {"current_incident": 0},
            }
            for window in windows
        )
        return EpisodeRecord(
            scenario_id="shadow_sequence",
            split="prospective_shadow",
            topology_group=deployment_pseudonym,
            ordered_window_ids=tuple(
                str(item["window_id"]) for item in windows
            ),
            windows=gate_windows,
            impact_onset=None,
            recovery_timestamp=None,
            confounder_family="not_available",
            healthy_control=False,
            failure_family="not_available",
        )

    def _prediction_id(
        self,
        snapshot: ServingSnapshot,
    ) -> str:
        return "prediction_" + canonical_hash(
            {
                "snapshot_hash": snapshot.content_hash(),
                "bundle_hash": self.bundle.bundle_hash,
                "mode": snapshot.mode,
            }
        )[:32]

    def _previous_state(
        self,
        snapshot: ServingSnapshot,
    ) -> DecisionState:
        candidates = [
            item
            for item in self.store.list_predictions()
            if item.deployment_pseudonym == snapshot.deployment_pseudonym
            and item.snapshot_mode is SnapshotMode.PROSPECTIVE
            and item.decision_cutoff_utc < snapshot.decision_cutoff_utc
        ]
        if not candidates:
            return DecisionState.HEALTHY
        return max(
            candidates,
            key=lambda item: item.decision_cutoff_utc,
        ).episode_state_after

    def _sequence(
        self,
        snapshot: ServingSnapshot,
    ) -> list[ServingSnapshot]:
        prior = self.store.list_snapshots(
            deployment_pseudonym=snapshot.deployment_pseudonym,
            mode=SnapshotMode.PROSPECTIVE,
            through_cutoff_utc=snapshot.decision_cutoff_utc,
        )
        if snapshot.mode is not SnapshotMode.PROSPECTIVE:
            prior = [
                item
                for item in prior
                if item.decision_cutoff_utc < snapshot.decision_cutoff_utc
            ]
            prior.append(snapshot)
        # A topology/profile transition starts a new temporal state epoch. It
        # is observable and monitored; silently carrying CUSUM/state across a
        # changed graph would violate the frozen feature semantics.
        return [
            item
            for item in prior
            if item.topology_version == snapshot.topology_version
            and item.profile.profile_version
            == snapshot.profile.profile_version
            and item.data_quality_state is not DataQualityState.BLOCKED
        ]

    def _blocked_envelope(
        self,
        *,
        snapshot: ServingSnapshot,
        prediction_id: str,
        started: float,
    ) -> PredictionEnvelope:
        return PredictionEnvelope(
            prediction_id=prediction_id,
            decision_cutoff_utc=snapshot.decision_cutoff_utc,
            deployment_pseudonym=snapshot.deployment_pseudonym,
            snapshot_mode=snapshot.mode,
            telemetry_snapshot_hash=snapshot.content_hash(),
            topology_snapshot_hash=snapshot.topology_hash(),
            operational_profile_hash=snapshot.profile_hash(),
            feature_schema_hash=self.feature_builder.schema_hash,
            model_bundle_hash=self.bundle.bundle_hash,
            policy_hash=self.bundle.policy_hash,
            episode_state_before=self._previous_state(snapshot),
            episode_state_after=DecisionState.DATA_QUALITY_ESCALATION,
            final_decision=DecisionState.DATA_QUALITY_ESCALATION,
            activation_path="data_quality",
            data_quality_state=DataQualityState.BLOCKED,
            data_quality_warnings=snapshot.data_quality_warnings,
            inference_latency_ms=(time.perf_counter() - started) * 1000.0,
            infrastructure_version=INFRASTRUCTURE_VERSION,
            created_at_utc=datetime.now(timezone.utc),
            supersedes_prediction_id=snapshot.supersedes_prediction_id,
        )

    def process(self, snapshot: ServingSnapshot) -> PredictionEnvelope:
        """Persist one immutable prospective or replay prediction."""

        started = time.perf_counter()
        snapshot_hash, _ = self.store.append_snapshot(snapshot)
        prediction_id = self._prediction_id(snapshot)
        try:
            return self.store.get_prediction(prediction_id)
        except KeyError:
            pass

        if snapshot.data_quality_state is DataQualityState.BLOCKED:
            envelope = self._blocked_envelope(
                snapshot=snapshot,
                prediction_id=prediction_id,
                started=started,
            )
            self.store.append_prediction(
                envelope=envelope,
                snapshot_hash=snapshot_hash,
            )
            if snapshot.mode is SnapshotMode.PROSPECTIVE:
                self.store.append_state_transition(
                    prediction_id=prediction_id,
                    deployment_pseudonym=snapshot.deployment_pseudonym,
                    decision_cutoff_utc=snapshot.decision_cutoff_utc,
                    payload={
                        "before": envelope.episode_state_before.value,
                        "after": envelope.episode_state_after.value,
                        "reason": "data_quality_blocked",
                    },
                )
            return envelope

        snapshots = self._sequence(snapshot)
        windows = self.feature_builder.build_sequence(snapshots)
        if not windows:
            raise RuntimeError("No inferable snapshot exists in state epoch")
        metadata_rows = [self._metadata(item) for item in snapshots]
        probabilities = [
            self.bundle.incident_model.predict_score(window)
            for window in windows
        ]
        support = [
            assess_support(
                model=self.bundle.support_model,
                window=window,
                metadata=metadata,
            )
            for window, metadata in zip(
                windows,
                metadata_rows,
                strict=True,
            )
        ]
        record = self._record(
            deployment_pseudonym=snapshot.deployment_pseudonym,
            windows=windows,
        )
        full_config = self.bundle.policy
        fast_config = replace(
            full_config,
            enable_fast_path=True,
            enable_slow_path=False,
        )
        slow_config = replace(
            full_config,
            enable_fast_path=False,
            enable_slow_path=True,
        )

        def run(config):
            return DualPathEpisodeGate(
                config=config,
                residual_model=self.bundle.residual_model,
            ).run(
                record=record,
                probabilities=probabilities,
                metadata=metadata_rows[-1],
                metadata_sequence=metadata_rows,
                support=support,
            )

        fast_result = run(fast_config)
        slow_result = run(slow_config)
        dual_result = run(full_config)
        fast_decision = fast_result.decisions[-1]
        slow_decision = slow_result.decisions[-1]
        dual_decision = dual_result.decisions[-1]
        current_window = windows[-1]
        impact = confirm_observable_impact(
            window=current_window,
            score_threshold=full_config.impact_score_threshold,
            minimum_families=full_config.impact_minimum_families,
            minimum_node_fraction=(
                full_config.impact_minimum_node_fraction
            ),
            require_causal_consistency=(
                full_config.require_causal_consistency
            ),
        )
        final_decision = _decision(dual_decision.state)
        ranking: tuple[RankedCause, ...] = ()
        if final_decision is DecisionState.INCIDENT:
            scores = sorted(
                learned_fusion_scores(
                    window=current_window,
                    model=self.bundle.fusion_model,
                ),
                key=lambda item: (-item.score, item.node_id),
            )[:3]
            ranking = tuple(
                RankedCause(
                    component_pseudonym=item.node_id,
                    score=item.score,
                    evidence={
                        key: float(value)
                        for key, value in item.evidence.items()
                    },
                )
                for item in scores
            )
        envelope = PredictionEnvelope(
            prediction_id=prediction_id,
            decision_cutoff_utc=snapshot.decision_cutoff_utc,
            deployment_pseudonym=snapshot.deployment_pseudonym,
            snapshot_mode=snapshot.mode,
            telemetry_snapshot_hash=snapshot.content_hash(),
            topology_snapshot_hash=snapshot.topology_hash(),
            operational_profile_hash=snapshot.profile_hash(),
            feature_schema_hash=self.feature_builder.schema_hash,
            feature_vector_hash=self.feature_builder.feature_hash(
                current_window
            ),
            model_bundle_hash=self.bundle.bundle_hash,
            policy_hash=self.bundle.policy_hash,
            stage1_probability=dual_decision.probability,
            healthy_residual=dual_decision.residual,
            support_score=dual_decision.support_score,
            support_axes=dict(support[-1].axis_scores),
            impact_score=impact.score,
            impact_families=impact.families,
            causal_consistency=impact.causal_consistent,
            fast_path_state=_decision(fast_decision.state),
            slow_path_state=_decision(slow_decision.state),
            dual_path_state=final_decision,
            slow_path_statistic=dual_decision.slow_statistic,
            episode_state_before=self._previous_state(snapshot),
            episode_state_after=final_decision,
            final_decision=final_decision,
            activation_path=dual_decision.activation_path,
            stage2_top_k=ranking,
            counterfactual_policy_states={
                "F": _decision(fast_decision.state),
                "S": _decision(slow_decision.state),
                "D": final_decision,
            },
            data_quality_state=snapshot.data_quality_state,
            data_quality_warnings=snapshot.data_quality_warnings,
            inference_latency_ms=(time.perf_counter() - started) * 1000.0,
            infrastructure_version=INFRASTRUCTURE_VERSION,
            created_at_utc=datetime.now(timezone.utc),
            supersedes_prediction_id=snapshot.supersedes_prediction_id,
        )
        self.store.append_prediction(
            envelope=envelope,
            snapshot_hash=snapshot_hash,
        )
        if snapshot.mode is SnapshotMode.PROSPECTIVE:
            self.store.append_state_transition(
                prediction_id=prediction_id,
                deployment_pseudonym=snapshot.deployment_pseudonym,
                decision_cutoff_utc=snapshot.decision_cutoff_utc,
                payload={
                    "before": envelope.episode_state_before.value,
                    "after": envelope.episode_state_after.value,
                    "fast": envelope.fast_path_state.value,
                    "slow": envelope.slow_path_state.value,
                    "dual": envelope.dual_path_state.value,
                    "activation_path": envelope.activation_path,
                },
            )
        return envelope
