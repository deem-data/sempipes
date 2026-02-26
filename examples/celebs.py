import numpy as np
import pandas as pd
import skrub

import sempipes


def annotation_pipeline():
    samples_to_annotate = skrub.var("samples_to_annotate")

    annotated = samples_to_annotate.sem_extract_features(
        nl_prompt="""
        Annotate a set of celebrity images with the specified attributes.

        IMPORTANT: Each attribute value should consist of a single word or phrase only from the list of potential answers!. 
        """,
        input_columns=["image"],
        name="extract_features",
        output_columns={
            "beard": "Does the person have a beard?",
            "gender": "Is the person in the photo a male or a female?",
            "young": "Which of the following age ranges is the person in the photo in: young, middle-aged, old?",
        },
        generate_via_code=True,
        print_code_to_console=True,
    )

    annotated = annotated.assign(
        beard=annotated["beard"].str.lower(),
        gender=annotated["gender"].str.lower(),
        young=annotated["young"].str.lower(),
    )

    return annotated


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        ),
    )

    attributes = pd.read_csv("examples/data/celebs.csv")

    to_annotate = attributes[["idx", "image"]]

    pipeline = annotation_pipeline()
    env = pipeline.skb.get_data()
    env["samples_to_annotate"] = to_annotate
    annotated_images = pipeline.skb.eval(env)

    annotated_images = annotated_images.merge(attributes, on="idx", how="left", suffixes=("", "_prediction"))

    print("Accuracy for beard: ", np.mean(annotated_images["beard"] == annotated_images["beard_prediction"]))
    print("Accuracy for gender: ", np.mean(annotated_images["gender"] == annotated_images["gender_prediction"]))
    print("Accuracy for young: ", np.mean(annotated_images["young"] == annotated_images["young_prediction"]))
