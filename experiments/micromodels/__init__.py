import json
import pandas as pd


def as_dataframe(empathy_data_path: str) -> pd.DataFrame:
    with open(empathy_data_path, "r", encoding="utf-8") as file_p:
        data = json.load(file_p)

    posts = []
    responses = []
    emotional_reaction_levels = []
    interpretation_levels = []
    explorations_levels = []
    emotional_reaction_level_rationales = []
    interpretation_level_rationales = []
    explorations_level_rationales = []

    for _, conversation in data.items():
        posts.append(conversation["seeker_post"])
        responses.append(conversation["response_post"])
        emotional_reaction_levels.append(conversation["emotional_reactions"]["level"])
        interpretation_levels.append(conversation["interpretations"]["level"])
        explorations_levels.append(conversation["explorations"]["level"])
        emotional_reaction_level_rationales.append(conversation["emotional_reactions"]["rationales"])
        interpretation_level_rationales.append(conversation["interpretations"]["rationales"])
        explorations_level_rationales.append(conversation["explorations"]["rationales"])

    df = pd.DataFrame.from_dict(
        {
            "post": posts,
            "response": responses,
            "emotional_reaction_level": emotional_reaction_levels,
            "emotional_reaction_level_rationale": emotional_reaction_level_rationales,
            "interpretation_level": interpretation_levels,
            "interpretation_level_rationale": interpretation_level_rationales,
            "explorations_level": explorations_levels,
            "explorations_level_rationale": explorations_level_rationales,
        }
    )

    df["emotional_reaction_level"] = df["emotional_reaction_level"].astype(int)
    df["interpretation_level"] = df["interpretation_level"].astype(int)
    df["explorations_level"] = df["explorations_level"].astype(int)

    return df


__all__ = ["as_dataframe"]
