from abc import abstractmethod
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from skrub import DataOp

from sempipes.config import get_config
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.llm.llm import generate_python_code_from_messages
from sempipes.logging import get_logger
from sempipes.operators.operators import ContextAwareMixin, OptimisableMixin

logger = get_logger()


class IterativeCodeGenEstimator(BaseEstimator, TransformerMixin, ContextAwareMixin, OptimisableMixin):  # pylint: disable=too-many-ancestors
    """Base class for semantic operators that iteratively generate Python code via an LLM with retry logic."""

    _SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        nl_prompt: str,
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
    ) -> None:
        self.nl_prompt = nl_prompt
        self._pipeline_summary = _pipeline_summary
        self._prefitted_state: dict[str, Any] | DataOp | None = _prefitted_state
        self._memory: list[dict[str, Any]] | DataOp | None = _memory
        self._inspirations: list[dict[str, Any]] | DataOp | None = _inspirations
        self.generated_code_: str | None = None

    @abstractmethod
    def _try_to_execute(self, code: str, *args, **kwargs) -> None:
        """Validate generated code; raise on failure."""

    @abstractmethod
    def empty_state(self) -> dict[str, Any]:
        """Return fallback state dict."""

    def _resolve_target_metric(self) -> str:
        if self._pipeline_summary is not None and self._pipeline_summary.target_metric is not None:
            return self._pipeline_summary.target_metric
        return "accuracy"

    def _memory_preamble_messages(self) -> list[dict[str, str]]:
        """Hook returning extra messages before memory history (default: empty).

        Override in subclasses to inject domain-specific guidance before the memory replay.
        """
        return []

    def _add_memorized_history(self, messages: list[dict[str, str]], target_metric: str) -> None:
        if self._memory is not None and len(self._memory) > 0:
            preamble = self._memory_preamble_messages()
            if preamble:
                messages += preamble

            current_score = None

            for memory_line in self._memory:
                memorized_code = memory_line["update"]
                memorized_score = memory_line["score"]

                if current_score is None:
                    improvement = abs(memorized_score)
                else:
                    improvement = memorized_score - current_score

                if improvement > 0.0:
                    add_feature_sentence = (
                        "The code was executed and improved the downstream performance. "
                        "You may choose to copy from this previous version of the code for the next version of the code."
                    )
                    current_score = memorized_score
                else:
                    add_feature_sentence = (
                        f"The last code changes did not improve performance. " f"(Improvement: {improvement})"
                    )

                messages += [
                    {"role": "assistant", "content": memorized_code},
                    {
                        "role": "user",
                        "content": f"Performance for last code block: {target_metric}={memorized_score:.5f}. "
                        f".{add_feature_sentence}\nNext codeblock:\n",
                    },
                ]

    def _build_error_feedback_message(self, code: str, error: Exception) -> list[dict[str, str]]:
        return [
            {"role": "assistant", "content": code},
            {
                "role": "user",
                "content": f"Code execution failed with error: {type(error)} {error}.\n "
                + f"Code: ```python{code}```\n Retry and fix the errors!\n```python\n",
            },
        ]

    def _load_prefitted_state(self) -> bool:
        if self._prefitted_state is not None:
            self.generated_code_ = self._prefitted_state["generated_code"]
            return True
        return False

    def _iterative_code_generation(self, *exec_args, prompt: str, **exec_kwargs) -> None:
        """Core retry loop: builds messages, injects memory, calls LLM, validates code."""
        target_metric = self._resolve_target_metric()
        messages: list[dict[str, str]] = []

        for attempt in range(1, get_config().max_retries_for_code_gen + 1):
            if attempt == 1:
                messages += [{"role": "system", "content": self._SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                self._add_memorized_history(messages, target_metric)

            code = generate_python_code_from_messages(messages)
            try:
                self._try_to_execute(code, *exec_args, **exec_kwargs)
                self.generated_code_ = code
                break
            except Exception as e:  # pylint: disable=broad-except
                logger.info(f"An error occurred in attempt {attempt}:", e)
                logger.debug(f"{e}", exc_info=True)
                messages += self._build_error_feedback_message(code, e)

        if self.generated_code_ is None:
            logger.error(f"No code generated after {get_config().max_retries_for_code_gen} retries. Falling back to empty state.")
            self.generated_code_ = self.empty_state()["generated_code"]

    def state_after_fit(self) -> dict[str, Any]:
        return {"generated_code": self.generated_code_}

    def memory_update_from_latest_fit(self) -> str:
        if self.generated_code_ is not None:
            return self.generated_code_
        return OptimisableMixin.EMPTY_MEMORY_UPDATE
