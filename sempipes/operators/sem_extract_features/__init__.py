from typing import Any
import skrub
from skrub import DataOp
from sempipes.inspection.pipeline_summary import PipelineSummary
from sempipes.operators.operators import EstimatorTransformer, SemExtractFeaturesOperator
from sempipes.operators.sem_extract_features.with_code import CodeBasedFeatureExtractor
from sempipes.operators.sem_extract_features.with_llm import LLMFeatureExtractor


class SemExtractFeaturesLLM(SemExtractFeaturesOperator):
    def generate_features_extractor(
        self,
        nl_prompt: str,
        input_columns: list[str],
        output_columns: dict[str, str] | None = None,
        _pipeline_summary: PipelineSummary | None | DataOp = None,
        _prefitted_state: dict[str, Any] | DataOp | None = None,
        _memory: list[dict[str, Any]] | DataOp | None = None,
        _inspirations: list[dict[str, Any]] | DataOp | None = None,
        **kwargs,
    ) -> EstimatorTransformer:
        if kwargs.get("generate_via_code", True):
            return CodeBasedFeatureExtractor(
                nl_prompt=nl_prompt,
                input_columns=input_columns,
                output_columns=output_columns,
                _pipeline_summary=_pipeline_summary,
                _prefitted_state=_prefitted_state,
                _memory=_memory,
                _inspirations=_inspirations,
            )

        # TODO: Should be made optimizable as well...
        return LLMFeatureExtractor(
            nl_prompt=nl_prompt,
            input_columns=input_columns,
            output_columns=output_columns,
            #                _pipeline_summary=_pipeline_summary,
            #                _prefitted_state=_prefitted_state,
            #                _memory=_memory,
            #                _inspirations=_inspirations
        )


def sem_extract_features(
    self: DataOp,
    nl_prompt: str,
    input_columns: list[str],
    name: str,
    output_columns: dict[str, str] | None = None,
    **kwargs,
) -> DataOp:
    _pipeline_summary = skrub.var(f"sempipes_pipeline_summary__{name}", None)
    _prefitted_state = skrub.var(f"sempipes_prefitted_state__{name}", None)
    _memory = skrub.var(f"sempipes_memory__{name}", [])
    _inspirations = skrub.var(f"sempipes_inspirations__{name}", [])

    feature_extractor = SemExtractFeaturesLLM().generate_features_extractor(
        nl_prompt, input_columns, output_columns, _pipeline_summary, _prefitted_state, _memory, _inspirations, **kwargs
    )

    return self.skb.apply(feature_extractor, how="no_wrap").skb.set_name(name)
