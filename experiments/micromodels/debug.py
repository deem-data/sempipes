import numpy as np
import skrub
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import f1_score

import sempipes
from experiments.micromodels import as_dataframe


def create_pipeline():
    posts_and_responses = skrub.var("posts_and_responses")

    y = posts_and_responses["emotional_reaction_level"].skb.mark_as_y()
    posts_and_responses = posts_and_responses[["post", "response"]].skb.mark_as_X()

    extracted = posts_and_responses.sem_extract_features(
        nl_prompt="""
        Given posts and responses from a social media forum, extract certain linguistic features on a scale from 0.0 to 2.0.        
        """,
        input_columns=["response"],
        output_columns={
            "sp_emotional_reaction_level": """The emotional reaction level of the response on a scale from 0.0 to 2.0.
            Most responses will have level 0.
            A response containing the text 'I'm with you all the way man.' has level 1.
            A response containing the text 'I agree with him' has level 1.
            A response containing the text 'Just keep practicing and never give up.' has level 1.
            A response containing the text 'I'm so sorry you feel that way.' has level 2.
            A response containing the text 'Holy shit I can relate so well to this.' has level 2.
            A response containing the text 'really sorry going through this.' has level 2.""",
            "sp_interpretation_level": """The interpretation level of the response on a scale from 0.0 to 2.0.
            Most responses will have level 0.
            A response containing the text 'People ask me why I'm zoned out most of the time. I'm not zoned out, I'm just in bot mode so I don't have to be attached to anything. It helps me supress the depressive thoughts but every night, it all just flows back in. I hate it.' has level 1.
            A response containing the text 'i skipped one class today because i don't really get much out of the class, but i forgot that there was 5% of the grade for in-class activities, so i'll probably not skip any more classes' has level 1.
            A response containing the text 'Actually I find myself taking much longer showers when I'm depressed. Sometimes twice in a day. It's the only time I feel relaxed.' has level 1.
            A response containing the text 'No, that's what I'm doing and it isn't working.' has level 2.
            A response containing the text 'I stopped catering to my problems... they just compile. I stopped obsessing over them or handling them in anyway. I have no control over my life. I simply look at my problems from my couch...' has level 2.
            A response containing the text 'I understand how you feel.' has level 2.        
            """,
            "sp_explorations_level": """The explorations level of the response on a scale from 0.0 to 2.0.
            Most responses will have level 0.
            A response containing the text 'What can we do today that will help?' has level 1.
            A response containing the text 'What happened?' has level 1.
            A response containing the text 'What makes you think you're a shitty human being?' has level 1.
            A response containing the text 'What makes you say these things?' has level 2.
            A response containing the text 'What do you feel is bringing on said troubling thoughts?' has level 2.
            A response containing the text 'Do you have any friends that aren't always going to blow sunshine up your ass or a therapist?' has level 2.        
            """,
        },
        name="response_features",
        generate_via_code=True,
    )

    X = extracted[["sp_emotional_reaction_level", "sp_interpretation_level", "sp_explorations_level"]]

    emo_ebm = ExplainableBoostingClassifier(
        feature_names=["sp_emotional_reaction_level", "sp_interpretation_level", "sp_explorations_level"]
    )
    return X.skb.apply(emo_ebm, y=y)

all_data = as_dataframe("experiments/micromodels/empathy.json").tail(123)
pipeline = create_pipeline()

env = pipeline.skb.get_data()
env["posts_and_responses"] = all_data

split = pipeline.skb.train_test_split(environment=env, test_size=0.5, random_state=42)
learner = pipeline.skb.make_learner(fitted=False)
print("###DEBUG -> TRAINING")
learner.fit(split["train"])
print("###DEBUG -> PREDICTING")
predictions = learner.predict(split["test"])
score = f1_score(split["y_test"], predictions, average="micro")
print(f"F1 score {score}")