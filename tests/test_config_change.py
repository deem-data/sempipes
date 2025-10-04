import sempipes


def test_updates():
    # Get default config and check its values
    default_config = sempipes.get_config()

    assert default_config.llm_for_code_generation is not None
    assert default_config.llm_for_batch_processing is not None
    assert isinstance(default_config.llm_for_code_generation, sempipes.LLM)
    assert isinstance(default_config.llm_for_batch_processing, sempipes.LLM)

    assert default_config.llm_for_code_generation.name == "openai/gpt-4.1"
    assert default_config.llm_for_code_generation.parameters["temperature"] == 0.0

    assert default_config.llm_for_batch_processing.name == "ollama/gpt-oss:20b"
    assert default_config.llm_for_batch_processing.parameters["api_base"] == "http://localhost:11434"
    assert default_config.llm_for_batch_processing.parameters["temperature"] == 0.0

    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="ollama/gpt-oss:20b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.5},
        )
    )

    current_config = sempipes.get_config()
    assert current_config.llm_for_code_generation.name == "ollama/gpt-oss:20b"
    assert current_config.llm_for_code_generation.parameters["api_base"] == "http://localhost:11434"
    assert current_config.llm_for_code_generation.parameters["temperature"] == 0.5

    sempipes.update_config(
        llm_for_batch_processing=sempipes.LLM(
            name="ollama/gemma3:1b",
            parameters={"api_base": "http://localhost:11434", "temperature": 0.0},
        ),
    )

    current_config = sempipes.get_config()
    assert current_config.llm_for_batch_processing.name == "ollama/gemma3:1b"
    assert current_config.llm_for_batch_processing.parameters["api_base"] == "http://localhost:11434"
    assert current_config.llm_for_batch_processing.parameters["temperature"] == 0.0
