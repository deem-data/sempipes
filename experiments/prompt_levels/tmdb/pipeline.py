import csv
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from skrub import DataOp
from skrub._data_ops._evaluation import find_node_by_name

import sempipes
from experiments.colopro import Setup
from experiments.prompt_levels.common import PromptLevelTestPipeline
from experiments.prompt_levels.tmdb._pipeline import LearnerKind, sempipes_pipeline
from experiments.prompt_levels.tmdb.prompts import PROMPTS
from sempipes.logging import get_logger
from sempipes.optimisers import EvolutionarySearch, optimise_colopro

warnings.filterwarnings("ignore")

logger = get_logger()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_PATH = Path("experiments/tmdb_box_office_prediction/data.csv")
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
    return RESULTS_DIR / f"tmdb_{tier}_{mode}_seed{seed}_nreps{nreps}.csv"


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


def _load_tmdb_data(seed: int, train_only: bool) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH)
    movie_stats = data.drop(columns=["revenue"])
    revenue = data["revenue"]
    if train_only:
        movie_stats, _, revenue, _ = train_test_split(
            movie_stats,
            revenue,
            train_size=PromptLevelTestPipeline.TEST_SIZE,
            random_state=seed,
        )
    return movie_stats, revenue


def _env_for_seed(seed: int, train_only: bool) -> dict:
    movie_stats, revenue = _load_tmdb_data(seed, train_only=train_only)
    return {"movie_stats": movie_stats, "revenue": revenue}


class TmdbTestPipeline(PromptLevelTestPipeline):
    OPERATOR_NAME = "additional_movie_features"

    @property
    def scoring(self) -> str:
        return "neg_root_mean_squared_log_error"

    def score(self, y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))

    def _env_for(self, pipeline: DataOp, additional_env_variables: dict) -> dict:
        env = pipeline.skb.get_data()
        env.update(additional_env_variables)
        return env

    def _apply_outcome_states(self, env: dict, outcome) -> None:
        for operator_name in outcome.states.operators():
            env[f"sempipes_prefitted_state__{operator_name}"] = outcome.states.get(operator_name)

    def _evaluate_with_outcome(self, seed, pipeline, additional_env_variables, outcome):
        logger.info(
            ("#" * 80)
            + "\n"
            + f"Operator states: { {name: outcome.states.get(name) for name in outcome.states.operators()} }"
            + "\n"
            + ("#" * 80)
        )

        np.random.seed(seed)
        movie_stats = additional_env_variables["movie_stats"]
        revenue = additional_env_variables["revenue"]
        train_ms, test_ms, train_rev, test_rev = train_test_split(
            movie_stats, revenue, test_size=self.TEST_SIZE, random_state=seed
        )

        train_env = self._env_for(pipeline, additional_env_variables)
        test_env = self._env_for(pipeline, additional_env_variables)
        train_env["movie_stats"] = train_ms
        train_env["revenue"] = train_rev
        test_env["movie_stats"] = test_ms
        test_env["revenue"] = test_rev
        self._apply_outcome_states(train_env, outcome)
        self._apply_outcome_states(test_env, outcome)

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        learner.fit(train_env)
        y_pred = learner.predict(test_env)
        score = self.score(test_rev.values, y_pred)

        logger.info(("%" * 80) + "\n" + f"Score: {score}" + "\n" + ("%" * 80))
        return score

    def _evaluate(self, seed, pipeline, additional_env_variables, operator_state=None):
        if operator_state is None:
            data_op = find_node_by_name(pipeline, self.OPERATOR_NAME)
            operator_state = data_op._skrub_impl.estimator.empty_state()

        logger.info(("#" * 80) + "\n" + f"Operator state: {operator_state}" + "\n" + ("#" * 80))

        np.random.seed(seed)
        movie_stats = additional_env_variables["movie_stats"]
        revenue = additional_env_variables["revenue"]
        train_ms, test_ms, train_rev, test_rev = train_test_split(
            movie_stats, revenue, test_size=self.TEST_SIZE, random_state=seed
        )

        train_env = self._env_for(pipeline, additional_env_variables)
        test_env = self._env_for(pipeline, additional_env_variables)
        train_env["movie_stats"] = train_ms
        train_env["revenue"] = train_rev
        test_env["movie_stats"] = test_ms
        test_env["revenue"] = test_rev
        train_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = operator_state
        test_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = operator_state

        learner = pipeline.skb.make_learner(fitted=False, keep_subsampling=False)
        learner.fit(train_env)
        y_pred = learner.predict(test_env)
        score = self.score(test_rev.values, y_pred)

        logger.info(("%" * 80) + "\n" + f"Score: {score}" + "\n" + ("%" * 80))
        return score

    def _evaluate_with_prompt_generation(self, seed, pipeline, additional_env_variables):
        np.random.seed(seed)
        movie_stats = additional_env_variables["movie_stats"]
        revenue = additional_env_variables["revenue"]
        train_ms, _, train_rev, _ = train_test_split(
            movie_stats, revenue, test_size=self.TEST_SIZE, random_state=seed
        )

        train_env = self._env_for(pipeline, additional_env_variables)
        train_env["movie_stats"] = train_ms
        train_env["revenue"] = train_rev
        train_env[f"sempipes_prefitted_state__{self.OPERATOR_NAME}"] = None

        pipeline_clone = pipeline.skb.clone()
        operator_op = find_node_by_name(pipeline_clone, self.OPERATOR_NAME)
        operator_op.skb.eval(train_env)
        operator_state = operator_op._skrub_impl.estimator_.state_after_fit()

        return self._evaluate(seed, pipeline, additional_env_variables, operator_state=operator_state)

    def _optimize_pipeline(self, pipeline, setup, additional_env_variables):
        search_policy = setup.search.clone_empty()
        return optimise_colopro(
            pipeline,
            num_trials=setup.num_trials,
            scoring=self.scoring,
            search=search_policy,
            cv=setup.cv,
            run_name=self.name,
            n_jobs_for_evaluation=-1,
            additional_env_variables=additional_env_variables,
            only_optimize_operator=self.OPERATOR_NAME,
        )


class TmdbPromptPipeline(TmdbTestPipeline):
    def __init__(self, prompt_level: str, learner: LearnerKind = "default"):
        if prompt_level not in PROMPTS:
            raise ValueError(f"Unknown prompt level: {prompt_level}. Choose from {list(PROMPTS)}")
        self.prompt_level = prompt_level
        self._nl_prompt = PROMPTS[prompt_level]
        self.learner = learner

    @property
    def name(self) -> str:
        return f"prompt_levels_tmdb_{self.prompt_level}"

    def pipeline_with_all_data(self, seed: int) -> tuple[DataOp, dict]:
        return (
            sempipes_pipeline(
                movie_features_prompt=self._nl_prompt,
                pipeline_seed=seed,
                learner=self.learner,
            ),
            _env_for_seed(seed, train_only=False),
        )

    def pipeline_with_train_data(self, seed: int) -> tuple[DataOp, dict]:
        return (
            sempipes_pipeline(
                movie_features_prompt=self._nl_prompt,
                pipeline_seed=seed,
                learner=self.learner,
            ),
            _env_for_seed(seed, train_only=True),
        )


class TmdbOriginalPipeline(TmdbTestPipeline):
    def __init__(self, learner: LearnerKind = "default"):
        self.learner = learner

    @property
    def name(self) -> str:
        return "prompt_levels_tmdb_medium"

    def pipeline_with_all_data(self, seed: int) -> tuple[DataOp, dict]:
        return (
            sempipes_pipeline(pipeline_seed=seed, learner=self.learner),
            _env_for_seed(seed, train_only=False),
        )

    def pipeline_with_train_data(self, seed: int) -> tuple[DataOp, dict]:
        return (
            sempipes_pipeline(pipeline_seed=seed, learner=self.learner),
            _env_for_seed(seed, train_only=True),
        )


def configure_llm():
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 2.0},
        ),
        prefer_empty_state_in_preview=True,
    )


def get_pipeline(prompt_level: str, learner: LearnerKind = "default") -> PromptLevelTestPipeline:
    if prompt_level == "medium":
        return TmdbOriginalPipeline(learner=learner)
    return TmdbPromptPipeline(prompt_level, learner=learner)


def run_baseline(
    prompt_level: str,
    seed: int = 42,
    nreps: int = 5,
    results_csv: Path | None = None,
    learner: LearnerKind = "default",
):
    configure_llm()
    pipeline = get_pipeline(prompt_level, learner=learner)
    test_scores = []
    rep_rows = []

    for rep in range(nreps):
        rep_seed = seed + rep
        np.random.seed(rep_seed)
        data_pipeline, env_variables = pipeline.pipeline_with_all_data(rep_seed)
        score = pipeline._evaluate_with_prompt_generation(rep_seed, data_pipeline, env_variables)
        test_scores.append(score)
        rep_rows.append({"rep": rep + 1, "rep_seed": rep_seed, "test_accuracy": score})
        print(f"[{prompt_level}] rep {rep + 1}/{nreps}: test RMSLE = {score}")

    mean_score = float(np.mean(test_scores))
    std_score = float(np.std(test_scores))
    print(f"[{prompt_level}] mean test RMSLE = {mean_score:.4f}")
    print(f"[{prompt_level}] std test RMSLE = {std_score:.4f}")

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
    learner: LearnerKind = "default",
):
    configure_llm()
    pipeline = get_pipeline(prompt_level, learner=learner)
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
    print(f"[{prompt_level}] mean final test RMSLE = {mean_score:.4f}")
    print(f"[{prompt_level}] std final test RMSLE = {std_score:.4f}")

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
    learner: LearnerKind = "default",
):
    pipeline = get_pipeline(prompt_level, learner=learner)
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
    print(f"[{prompt_level}] mean final test RMSLE = {mean_score:.4f}")
    print(f"[{prompt_level}] std final test RMSLE = {std_score:.4f}")

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

    parser = argparse.ArgumentParser(description="TMDB box office prompt-level experiment")
    parser.add_argument("tier", choices=["lightweight", "medium", "elaborate"])
    parser.add_argument("mode", choices=["baseline", "optimized", "from_trajectory"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nreps", type=int, default=5)
    parser.add_argument("--num-trials", type=int, default=36)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--learner",
        choices=["default", "fast"],
        default="default",
        help="Downstream model: default (full RF pipeline from impl2) or fast (smaller RF, for COLOPRO search)",
    )
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
        help="CSV path for per-rep scores and summary (default: experiments/prompt_levels/tmdb/results/...)",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline(
            args.tier,
            seed=args.seed,
            nreps=args.nreps,
            results_csv=args.results_csv,
            learner=args.learner,
        )
    elif args.mode == "optimized":
        run_optimized(
            args.tier,
            seed=args.seed,
            nreps=args.nreps,
            num_trials=args.num_trials,
            cv=args.cv,
            results_csv=args.results_csv,
            learner=args.learner,
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
            learner=args.learner,
        )


if __name__ == "__main__":
    main()
