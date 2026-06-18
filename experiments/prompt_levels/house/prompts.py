PROMPT_LIGHTWEIGHT = """
Compute a small set of simple numeric features for house sale price prediction.
Focus on basic aggregates only: total bathrooms, house age, and total living square footage.
Do not create interaction terms, quality scores, or ordinal encodings of categorical columns.
"""

# Kept in sync with DEFAULT_HOUSE_FEATURES_PROMPT in _pipeline.py.
# Medium runs via HousePricesOriginalPipeline (uses _pipeline.py default), not PROMPTS directly.
PROMPT_MEDIUM = """
Compute a small set of simple numeric features for house sale price prediction — slightly beyond
the bare minimum, but keep the total to about five features.

Include total bathrooms, house age, total living area (GrLivArea plus TotalBsmtSF), and years
since remodel. You may add one simple binary flag such as HasGarage (GarageArea > 0).

Do not create interaction terms, quality scores, outdoor-area sums, extra amenity flags, or
ordinal encodings of categorical columns.
"""

PROMPT_ELABORATE = """
Compute numeric engineered features from house attributes to predict sale price.
Create only numeric features from existing numeric columns — do not ordinal-encode or map string
categorical columns (a separate pipeline step handles those).

Include:
- Total bathrooms (full baths plus 0.5 × half baths, including basement)
- House age and years since remodel (YrSold minus YearBuilt / YearRemodAdd)
- Total living area (GrLivArea + TotalBsmtSF)
- Quality–size interactions: OverallQual × total living area, OverallQual × GarageArea
- OverallQual × OverallCond combined score
- Binary indicators such as HasGarage (GarageArea > 0)
- Total outdoor living area (WoodDeckSF plus all porch square footages)
- Room/bath aggregates (e.g. TotRmsAbvGrd + bedrooms + baths)
"""

PROMPTS = {
    "lightweight": PROMPT_LIGHTWEIGHT,
    "medium": PROMPT_MEDIUM,
    "elaborate": PROMPT_ELABORATE,
}
