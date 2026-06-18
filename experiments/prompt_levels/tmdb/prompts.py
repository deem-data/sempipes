PROMPT_LIGHTWEIGHT = """
Create a small set of simple numeric features for movie box office prediction.
Focus on release date parts (year and month) and basic counts (number of cast members,
number of crew members).
Do not extract individual crew roles, parse JSON metadata, or engineer creative derived features.
"""

# Kept in sync with DEFAULT_TMDB_FEATURES_PROMPT in _pipeline.py.
# Medium runs via TmdbOriginalPipeline (uses _pipeline.py default), not PROMPTS directly.
PROMPT_MEDIUM = """
Create additional features that could help predict the box office revenue of a movie.
Consider aspects like genre, production details, cast, crew, and any other relevant information
that could influence a movie's financial success. Some of the attributes are in JSON format,
so you might need to parse them to extract useful information.
"""

PROMPT_ELABORATE = """
Create additional features that could help predict the box office revenue of a movie. Here are detailed instructions:

Compute the year, month, day, day of week, day of year of the movie release data, and the elapsed time since then

Compute the following features:
- number of cast members
- number of crew members
- whether the movie has a tagline
- length of the tagline (in words)
- whether the movie has an overview
- length of the overview (in words)
- whether the movie has a budget greater than zero
- whether the movie has a homepage

Next, extract the following metadata:
- director of the movie
- screenplay writer
- director of photography
- original music composer
- art director

Besides that, think creatively about other features that could be useful for predicting box office revenue,
and create them as well."""

PROMPTS = {
    "lightweight": PROMPT_LIGHTWEIGHT,
    "medium": PROMPT_MEDIUM,
    "elaborate": PROMPT_ELABORATE,
}
