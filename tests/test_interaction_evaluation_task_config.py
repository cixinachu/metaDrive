import pytest

from studio.config import clean_interaction_evaluation_request


def test_closed_loop_evaluation_defaults_to_meta_drive_physics_and_candidate_replanning():
    config = clean_interaction_evaluation_request({})
    assert config["task"] == "evaluation"
    assert config["scene_source"] == "procedural"
    assert config["dynamics_mode"] == "metadrive_physics"
    assert config["candidate_count"] == 16
    assert config["replan_steps"] == 5
    assert config["initialization_protocol"] == "canonical"


def test_closed_loop_evaluation_rejects_invalid_candidate_count_and_merge_distribution():
    with pytest.raises(ValueError, match="Planner 候选数"):
        clean_interaction_evaluation_request({"pipeline_candidates": 0})
    with pytest.raises(ValueError, match="NGSIM 初始分布"):
        clean_interaction_evaluation_request({
            "pipeline_scenario": "merge",
            "pipeline_initialization": "ngsim_distribution",
        })
