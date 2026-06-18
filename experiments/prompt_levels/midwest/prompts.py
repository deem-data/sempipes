PROMPT_LIGHTWEIGHT = """
Compute features that help predict US census region from survey demographics.
"""

# One-to-one copy of experiments/colopro/_midwest.py.
# Medium runs via MidwestSurveyPipeline (original pipeline), not via PROMPTS.
PROMPT_MEDIUM = """
Compute additional features which help predict the census region of a respondent based on their demographics. Use your intrinsic knowledge about the US to come up with the features. Pay special attention to the zip code of the person.
"""

PROMPT_ELABORATE = """
Compute additional features which help predict the census region of a respondent based on their demographics.
Use your intrinsic knowledge about the US to come up with the features.

Extract the first 3 digits of the ZIP code (ZIP3 prefix) from In_what_ZIP_code_is_your_home_located.
Map each respondent's ZIP3 prefix to its census division. Handle missing or invalid ZIP codes gracefully.
"""

PROMPTS = {
    "lightweight": PROMPT_LIGHTWEIGHT,
    "medium": PROMPT_MEDIUM,
    "elaborate": PROMPT_ELABORATE,
}
