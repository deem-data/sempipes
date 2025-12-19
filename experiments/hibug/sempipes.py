import pandas as pd
import skrub

import sempipes

sempipes.update_config(
    llm_for_code_generation=sempipes.LLM(
        name="gemini/gemini-2.5-flash",
        parameters={"temperature": 0.0},
    ),
)


def annotation_pipeline():
    samples_to_annotate = skrub.var("samples_to_annotate")

    annotated = samples_to_annotate.sem_extract_features(
        nl_prompt="""
        Annotate a set of celebrity images with the specified attributes. The attributes will be used to debug a model predicting whether a celebrity is wearing lipstick in an image. Make sure that your attributes are correlated with potential failures of this prediction task.

        IMPORTANT: Each attribute value should consist of a single word or phrase only from the list of potential answers!. 
        """,
        input_columns=["image"],
        name="extract_features",
        output_columns={
            "beard": "Does the person have a beard?",
            "makeup": "Does the person wear makeup?",
            "gender": "Is the person in the photo a male or a female?",
            "hair_color": "Which of the following hair colors does the person in the photo have: blonde, brown, black, gray, white or red?",
            "skin_color": "Does the person in the photo have white, brown or black skin?",
            "emotion": "Which of the following emotions is the person in the photo showing: sad, serious, calm, happy, surprised, neutral, angry, excited, pensive?",
            "age": "Which of the following age ranges is the person in the photo in: young, middle-aged, old?",
        },
        generate_via_code=True,
        print_code_to_console=True,
    )

    def inspect_afterwards(df):
        print(df.head(n=30))
        return df

    return annotated.skb.apply_func(inspect_afterwards)


df = pd.read_csv("experiments/hibug/hibug_attributes.csv")
df = df.iloc[2000:]
df = df[["idx", "image", "label", "prediction"]]


pipeline = annotation_pipeline()
env = pipeline.skb.get_data()
env["samples_to_annotate"] = df
result = pipeline.skb.eval(env)
result.to_csv("experiments/hibug/sempipes.csv", index=False)
