from app.domain.nrim.shadow.readiness import historical_replay_gate


def test_synthetic_evidence_has_explicit_real_data_required_terminal_state():
    result = historical_replay_gate(
        {
            "real_data": False,
            "analysis_plan_preregistered": True,
            "required_feature_fraction": 1.0,
            "collector_mapping_fraction": 1.0,
            "semantic_mutation_rejection_fraction": 1.0,
            "incident_alignment_fraction": 1.0,
            "label_availability_reported": True,
            "future_leakage_count": 0,
            "replay_determinism_fraction": 1.0,
            "root_cause_mapping_fraction": 1.0,
            "topology_reconstruction_fraction": 1.0,
            "model_tuning_event_count": 0,
            "threshold_tuning_event_count": 0,
            "feature_tuning_event_count": 0,
            "support_rule_tuning_event_count": 0,
            "watermark_tuning_event_count": 0,
            "episode_grouping_tuning_event_count": 0,
            "semantic_deviation_count": 0,
            "semantic_lineage_record_count": 0,
            "data_gap_report_complete": True,
        }
    )

    assert result["terminal_decision"] == "REAL_DATA_REQUIRED"
