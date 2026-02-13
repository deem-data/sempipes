import json
import tempfile
from pathlib import Path

from sempipes.config import LLM, Config
from sempipes.optimisers.search_policy import Outcome, SearchNode
from sempipes.optimisers.trajectory import Trajectory, load_trajectory_from_json, save_trajectory_as_json


def test_trajectory_save_and_load():
    """Test that trajectories can be saved and loaded correctly."""
    # Create a test trajectory
    config = Config(
        llm_for_code_generation=LLM(name="test-llm", parameters={"temperature": 0.5}),
        llm_for_batch_processing=LLM(name="test-batch-llm", parameters={"temperature": 0.0}),
        batch_size_for_batch_processing=10,
    )

    search_node = SearchNode(
        trial=0,
        parent_trial=None,
        memory=[],
        predefined_state={"generated_code": []},
        parent_score=None,
        inspirations=[],
    )

    outcome = Outcome(
        search_node=search_node,
        state={"generated_code": ["test_code"]},
        score=0.95,
        memory_update="test update",
    )

    trajectory = Trajectory(
        sempipes_config=config,
        optimizer_args={"operator_name": "test_operator", "num_trials": 5, "scoring": "accuracy"},
        outcomes=[outcome],
    )

    # Test saving and loading with new format using save_trajectory_as_json
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        output_path = save_trajectory_as_json(trajectory, run_name="test", output_folder=output_folder)

        # Load and verify
        loaded_trajectory = load_trajectory_from_json(output_path)

        assert loaded_trajectory.sempipes_config.llm_for_code_generation.name == "test-llm"
        assert loaded_trajectory.sempipes_config.llm_for_code_generation.parameters["temperature"] == 0.5
        assert loaded_trajectory.optimizer_args["operator_name"] == "test_operator"
        assert len(loaded_trajectory.outcomes) == 1
        assert loaded_trajectory.outcomes[0].score == 0.95
        assert loaded_trajectory.outcomes[0].state["generated_code"] == ["test_code"]


def test_trajectory_load_old_format():
    """Test that old double-serialized trajectory files can still be loaded."""
    # Create a test trajectory
    config = Config(
        llm_for_code_generation=LLM(name="test-llm", parameters={"temperature": 0.5}),
        llm_for_batch_processing=LLM(name="test-batch-llm", parameters={"temperature": 0.0}),
        batch_size_for_batch_processing=10,
    )

    search_node = SearchNode(
        trial=0,
        parent_trial=None,
        memory=[],
        predefined_state={"generated_code": []},
        parent_score=None,
        inspirations=[],
    )

    outcome = Outcome(
        search_node=search_node,
        state={"generated_code": ["test_code"]},
        score=0.95,
        memory_update="test update",
    )

    trajectory = Trajectory(
        sempipes_config=config,
        optimizer_args={"operator_name": "test_operator", "num_trials": 5, "scoring": "accuracy"},
        outcomes=[outcome],
    )

    # Test loading old format (double-serialized JSON string)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_trajectory_old.json"

        # Save in old format: to_json() returns a string, which gets serialized again
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectory.to_json(), f, indent=2, ensure_ascii=False)  # pylint: disable=no-member

        # Load and verify - should work with old format
        loaded_trajectory = load_trajectory_from_json(output_path)

        assert loaded_trajectory.sempipes_config.llm_for_code_generation.name == "test-llm"
        assert loaded_trajectory.sempipes_config.llm_for_code_generation.parameters["temperature"] == 0.5
        assert loaded_trajectory.optimizer_args["operator_name"] == "test_operator"
        assert len(loaded_trajectory.outcomes) == 1
        assert loaded_trajectory.outcomes[0].score == 0.95
        assert loaded_trajectory.outcomes[0].state["generated_code"] == ["test_code"]


def test_trajectory_both_formats():
    """Test that both old and new formats produce equivalent loaded trajectories."""
    # Create a test trajectory
    config = Config(
        llm_for_code_generation=LLM(name="test-llm", parameters={"temperature": 0.5}),
        llm_for_batch_processing=LLM(name="test-batch-llm", parameters={"temperature": 0.0}),
        batch_size_for_batch_processing=10,
    )

    search_node = SearchNode(
        trial=0,
        parent_trial=None,
        memory=[],
        predefined_state={"generated_code": []},
        parent_score=None,
        inspirations=[],
    )

    outcome = Outcome(
        search_node=search_node,
        state={"generated_code": ["test_code"]},
        score=0.95,
        memory_update="test update",
    )

    trajectory = Trajectory(
        sempipes_config=config,
        optimizer_args={"operator_name": "test_operator", "num_trials": 5, "scoring": "accuracy"},
        outcomes=[outcome],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        # Save in new format using save_trajectory_as_json
        new_format_path = save_trajectory_as_json(trajectory, run_name="test_new", output_folder=output_folder)

        # Save in old format (double-serialized JSON string) - manually create to test backward compatibility
        old_format_path = Path(tmpdir) / "test_trajectory_old.json"
        with open(old_format_path, "w", encoding="utf-8") as f:
            json.dump(trajectory.to_json(), f, indent=2, ensure_ascii=False)  # pylint: disable=no-member

        # Load both
        loaded_new = load_trajectory_from_json(new_format_path)
        loaded_old = load_trajectory_from_json(old_format_path)

        # Verify both produce equivalent results
        assert (
            loaded_new.sempipes_config.llm_for_code_generation.name
            == loaded_old.sempipes_config.llm_for_code_generation.name
        )
        assert loaded_new.optimizer_args == loaded_old.optimizer_args
        assert len(loaded_new.outcomes) == len(loaded_old.outcomes)
        assert loaded_new.outcomes[0].score == loaded_old.outcomes[0].score
        assert loaded_new.outcomes[0].state == loaded_old.outcomes[0].state
