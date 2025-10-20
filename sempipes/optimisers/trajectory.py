import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sempipes.config import Config
from sempipes.optimisers.search_policy import Outcome


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
        json.dump(asdict(trajectory), f, indent=2, ensure_ascii=False)

    return output_path
