import csv
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skrub import DataOp, TableVectorizer

import sempipes
from experiments.colopro import Setup, TestPipeline
from experiments.colopro._midwest import MidwestSurveyPipeline
from experiments.prompt_levels.common import PromptLevelTestPipeline
from experiments.prompt_levels.midwest.prompts import PROMPTS
from sempipes.optimisers import EvolutionarySearch

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_FIELDNAMES = [
    "timestamp",
    "tier",
    "mode",
    "seed_start",
    "nreps",
    "record_type",
    "rep",
    "rep_seed",
    "test_accuracy",
    "train_score",
    "max_trial",
    "chosen_trial",
    "num_trials",
    "cv",
]


def _default_results_path(tier: str, mode: str, seed: int, nreps: int) -> Path:
    return RESULTS_DIR / f"midwest_{tier}_{mode}_seed{seed}_nreps{nreps}.csv"


def _write_results_csv(
    path: Path,
    *,
    tier: str,
    mode: str,
    seed: int,
    nreps: int,
    rep_rows: list[dict],
    mean_score: float,
    std_score: float,
    num_trials: int | None = None,
    cv: int | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    def _base_row(record_type: str, **kwargs) -> dict:
        row = {
            "timestamp": timestamp,
            "tier": tier,
            "mode": mode,
            "seed_start": seed,
            "nreps": nreps,
            "record_type": record_type,
            "rep": "",
            "rep_seed": "",
            "test_accuracy": "",
            "train_score": "",
            "max_trial": "",
            "chosen_trial": "",
            "num_trials": num_trials if num_trials is not None else "",
            "cv": cv if cv is not None else "",
        }
        row.update(kwargs)
        return row

    rows = [
        _base_row(
            "rep",
            rep=row["rep"],
            rep_seed=row["rep_seed"],
            test_accuracy=row["test_accuracy"],
            train_score=row.get("train_score", ""),
            max_trial=row.get("max_trial", ""),
            chosen_trial=row.get("chosen_trial", ""),
        )
        for row in rep_rows
    ]
    rows.append(_base_row("mean", test_accuracy=mean_score))
    rows.append(_base_row("std", test_accuracy=std_score))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{tier}] wrote results to {path}")
    return path


def _load_midwest_data(seed: int, train_only: bool):
    dataset = skrub.datasets.fetch_midwest_survey()

    X = dataset.X.iloc[:2000]
    mask = ~X["In_what_ZIP_code_is_your_home_located"].str.contains(r"[^0-9.]")
    X = X[mask]
    y = dataset.y.iloc[:2000][mask]

    if train_only:
        X, _, y, _ = train_test_split(X, y, train_size=TestPipeline.TEST_SIZE, random_state=seed)

    env_variables = {
        "data": X,
        "labels": y,
    }

    return dataset.metadata["description"], dataset.metadata["target"], env_variables


def _build_pipeline(X_description: str, y_description: str, nl_prompt: str) -> DataOp:
    responses = skrub.var("data")
    responses = responses.skb.set_description(X_description)

    labels = skrub.var("labels")
    labels = labels.skb.set_description(y_description)

    responses = responses.skb.mark_as_X()
    labels = labels.skb.mark_as_y()

    responses_with_additional_features = responses.sem_gen_features(
        nl_prompt=nl_prompt,
        name=TestPipeline.OPERATOR_NAME,
        how_many=5,
    )

    encoded_responses = responses_with_additional_features.skb.apply(TableVectorizer())
    return encoded_responses.skb.apply(RandomForestClassifier(random_state=0), y=labels)


class MidwestPromptPipeline(PromptLevelTestPipeline):
    def __init__(self, prompt_level: str):
        if prompt_level not in PROMPTS:
            raise ValueError(f"Unknown prompt level: {prompt_level}. Choose from {list(PROMPTS)}")
        self.prompt_level = prompt_level
        self._nl_prompt = PROMPTS[prompt_level]

    @property
    def name(self) -> str:
        return f"prompt_levels_midwest_{self.prompt_level}"

    @property
    def scoring(self) -> str:
        return "accuracy"

    def score(self, y_true, y_pred) -> float:
        from sklearn.metrics import accuracy_score

        return accuracy_score(y_true, y_pred)

    def pipeline_with_all_data(self, seed: int) -> tuple[DataOp, dict]:
        X_description, y_description, env_variables = _load_midwest_data(seed, train_only=False)
        return _build_pipeline(X_description, y_description, self._nl_prompt), env_variables

    def pipeline_with_train_data(self, seed: int) -> tuple[DataOp, dict]:
        X_description, y_description, env_variables = _load_midwest_data(seed, train_only=True)
        return _build_pipeline(X_description, y_description, self._nl_prompt), env_variables


class MidwestMediumPipeline(PromptLevelTestPipeline, MidwestSurveyPipeline):
    pass


def configure_llm():
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
        prefer_empty_state_in_preview=True,
    )


def get_pipeline(prompt_level: str) -> PromptLevelTestPipeline:
    if prompt_level == "medium":
        return MidwestMediumPipeline()
    return MidwestPromptPipeline(prompt_level)


def run_baseline(
    prompt_level: str,
    seed: int = 42,
    nreps: int = 5,
    results_csv: Path | None = None,
):
    configure_llm()
    pipeline = get_pipeline(prompt_level)
    test_scores = []
    rep_rows = []

    for rep in range(nreps):
        rep_seed = seed + rep
        np.random.seed(rep_seed)
        data_pipeline, env_variables = pipeline.pipeline_with_all_data(rep_seed)
        score = pipeline._evaluate_with_prompt_generation(rep_seed, data_pipeline, env_variables)
        test_scores.append(score)
        rep_rows.append({"rep": rep + 1, "rep_seed": rep_seed, "test_accuracy": score})
        print(f"[{prompt_level}] rep {rep + 1}/{nreps}: test accuracy = {score}")

    mean_score = float(np.mean(test_scores))
    std_score = float(np.std(test_scores))
    print(f"[{prompt_level}] mean test accuracy = {mean_score:.4f}")
    print(f"[{prompt_level}] std test accuracy = {std_score:.4f}")

    csv_path = results_csv or _default_results_path(prompt_level, "baseline", seed, nreps)
    _write_results_csv(
        csv_path,
        tier=prompt_level,
        mode="baseline",
        seed=seed,
        nreps=nreps,
        rep_rows=rep_rows,
        mean_score=mean_score,
        std_score=std_score,
    )
    return test_scores


def run_optimized(
    prompt_level: str,
    seed: int = 42,
    nreps: int = 5,
    num_trials: int = 36,
    cv: int = 5,
    results_csv: Path | None = None,
):
    configure_llm()
    pipeline = get_pipeline(prompt_level)
    setup = Setup(
        search=EvolutionarySearch(population_size=6),
        num_trials=num_trials,
        cv=cv,
        llm_for_code_generation=sempipes.get_config().llm_for_code_generation,
    )

    all_final_test_scores = []
    rep_rows = []

    for rep in range(nreps):
        rep_seed = seed + rep
        np.random.seed(rep_seed)
        scores_over_time = pipeline.optimize(rep_seed, setup)
        final = scores_over_time[-1]
        max_trial, chosen_trial, train_score, test_score = final
        all_final_test_scores.append(test_score)
        rep_rows.append(
            {
                "rep": rep + 1,
                "rep_seed": rep_seed,
                "test_accuracy": test_score,
                "train_score": train_score,
                "max_trial": max_trial,
                "chosen_trial": chosen_trial,
            }
        )
        print(
            f"[{prompt_level}] rep {rep + 1}/{nreps}: "
            f"trial={max_trial}, chosen={chosen_trial}, "
            f"train={train_score}, test={test_score}"
        )

    mean_score = float(np.nanmean([s for s in all_final_test_scores if s is not None] or [np.nan]))
    std_score = float(np.nanstd([s for s in all_final_test_scores if s is not None] or [np.nan]))
    print(f"[{prompt_level}] mean final test accuracy = {mean_score:.4f}")
    print(f"[{prompt_level}] std final test accuracy = {std_score:.4f}")

    csv_path = results_csv or _default_results_path(prompt_level, "optimized", seed, nreps)
    _write_results_csv(
        csv_path,
        tier=prompt_level,
        mode="optimized",
        seed=seed,
        nreps=nreps,
        rep_rows=rep_rows,
        mean_score=mean_score,
        std_score=std_score,
        num_trials=num_trials,
        cv=cv,
    )
    return all_final_test_scores


def run_from_trajectory(
    prompt_level: str,
    trajectory_paths: list[Path],
    seed: int = 42,
    num_trials: int | None = None,
    results_csv: Path | None = None,
):
    pipeline = get_pipeline(prompt_level)
    nreps = len(trajectory_paths)
    all_final_test_scores = []
    rep_rows = []

    for rep, trajectory_path in enumerate(trajectory_paths):
        rep_seed = seed + rep
        np.random.seed(rep_seed)
        print(f"[{prompt_level}] rep {rep + 1}/{nreps}: loading {trajectory_path}")
        scores_over_time = pipeline.evaluate_trajectory_scores_over_time(
            rep_seed, trajectory_path, num_trials=num_trials
        )
        final = scores_over_time[-1]
        max_trial, chosen_trial, train_score, test_score = final
        all_final_test_scores.append(test_score)
        rep_rows.append(
            {
                "rep": rep + 1,
                "rep_seed": rep_seed,
                "test_accuracy": test_score,
                "train_score": train_score,
                "max_trial": max_trial,
                "chosen_trial": chosen_trial,
            }
        )
        print(
            f"[{prompt_level}] rep {rep + 1}/{nreps}: "
            f"trial={max_trial}, chosen={chosen_trial}, "
            f"train={train_score}, test={test_score}"
        )

    mean_score = float(np.nanmean([s for s in all_final_test_scores if s is not None] or [np.nan]))
    std_score = float(np.nanstd([s for s in all_final_test_scores if s is not None] or [np.nan]))
    print(f"[{prompt_level}] mean final test accuracy = {mean_score:.4f}")
    print(f"[{prompt_level}] std final test accuracy = {std_score:.4f}")

    csv_path = results_csv or _default_results_path(prompt_level, "from_trajectory", seed, nreps)
    _write_results_csv(
        csv_path,
        tier=prompt_level,
        mode="from_trajectory",
        seed=seed,
        nreps=nreps,
        rep_rows=rep_rows,
        mean_score=mean_score,
        std_score=std_score,
        num_trials=num_trials,
    )
    return all_final_test_scores


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Midwest prompt-level experiment")
    parser.add_argument("tier", choices=["lightweight", "medium", "elaborate"])
    parser.add_argument("mode", choices=["baseline", "optimized", "from_trajectory"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nreps", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=36)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        help="Trajectory JSON (repeat for multiple reps). Required for from_trajectory mode.",
    )
    parser.add_argument(
        "--trajectory-glob",
        type=str,
        default=None,
        help="Glob under .sempipes_trajectories/ when --trajectory is not set (from_trajectory mode)",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="CSV path for per-rep scores and summary (default: experiments/prompt_levels/midwest/results/...)",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline(args.tier, seed=args.seed, nreps=args.nreps, results_csv=args.results_csv)
    elif args.mode == "optimized":
        run_optimized(
            args.tier,
            seed=args.seed,
            nreps=args.nreps,
            num_trials=args.num_trials,
            cv=args.cv,
            results_csv=args.results_csv,
        )
    else:
        trajectory_paths = list(args.trajectory or [])
        if not trajectory_paths and args.trajectory_glob:
            trajectory_paths = sorted(Path(".sempipes_trajectories").glob(args.trajectory_glob))
        if not trajectory_paths:
            parser.error("from_trajectory mode requires --trajectory and/or --trajectory-glob")
        run_from_trajectory(
            args.tier,
            trajectory_paths=trajectory_paths,
            seed=args.seed,
            num_trials=args.num_trials,
            results_csv=args.results_csv,
        )


if __name__ == "__main__":
    main()
