import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dataclasses_json import dataclass_json

from sempipes.config import Config
from sempipes.optimisers.search_policy import Outcome


@dataclass_json
@dataclass
class Trajectory:
    sempipes_config: Config
    optimizer_args: dict[str, Any]
    outcomes: list[Outcome]


def save_trajectory_as_json(trajectory: Trajectory) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"colopro_{timestamp}_{unique_id}.json"

    output_path = Path(".sempipes_trajectories") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectory.to_json(), f, indent=2, ensure_ascii=False)  # type: ignore

    return output_path


def load_trajectory_from_json(json_path: Path) -> Trajectory:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # pylint: disable=no-member
    return Trajectory.from_json(data)  # type: ignore
