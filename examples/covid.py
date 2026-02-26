import warnings

import pandas as pd
import skrub
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sempipes

warnings.filterwarnings("ignore")

DATA_DESCRIPTION = """
SIVEP-Gripe, Brazil’s national surveillance system for severe acute respiratory infections (SRAG).
---
# Column Meanings in the **SIVEP-Gripe** Dataset
> **Note:** Many SIVEP fields are encoded as numeric categories.
> Coding below reflects the standard definitions used in the public dataset.
---

## Clinical / Epidemiological Variables

### **antiviral**
* **Type:** Categorical (1/2/9)
* **Meaning:** Antiviral treatment (e.g., oseltamivir).
  * **1** = Yes
  * **2** = No
  * **9** = Unknown
### **ave_suino**
* **Type:** Categorical
* **Meaning:** Exposure to poultry (*aves*) or swine (*suínos*).
  * **1** = Yes
  * **2** = No
  * **9** = Unknown
### **cardiopati**
* **Type:** Categorical
* **Meaning:** Chronic heart disease as comorbidity.
  * **1** = Yes
  * **2** = No
  * **9** = Unknown
### **classi_fin**
* **Type:** Categorical (final classification)
* **Meaning:** Final SRAG case classification.
  * **1** = Influenza
  * **2** = Other respiratory virus
  * **3** = Other etiologies
  * **4** = Unspecified SRAG
  * **5** = COVID-19
### **cs_raca**
* **Type:** Categorical
* **Meaning:** Race/skin color.
  * **1** = White
  * **2** = Black
  * **3** = Asian
  * **4** = Brown (Pardo)
  * **5** = Indigenous
  * **99** = Unknown
### **cs_zona**
* **Type:** Categorical
* **Meaning:** Residence zone.
  * **1** = Urban
  * **2** = Rural
  * **3** = Peri-urban
  * **9** = Unknown
### **desc_resp**
* **Type:** Text / Categorical
* **Meaning:** Description of respiratory support (e.g., mechanical ventilation, oxygen therapy).
---
## 🩺 Symptoms (common coding: 1 = yes, 2 = no, 9 = unknown)
* **diarreia** — diarrhea
* **dispneia** — shortness of breath
* **dor_abd** — abdominal pain
* **fadiga** — fatigue
* **febre** — fever
* **garganta** — sore throat
* **perd_olft** — loss of smell
* **perd_pala** — loss of taste
* **tosse** — cough
* **vomito** — vomiting
*All coded as 1/2/9.*
---
## Exams and Clinical Measures
### **raiox_res**
* **Type:** Categorical
* **Meaning:** Chest X-ray result.
  * **1** = Normal
  * **2** = Interstitial infiltrate
  * **3** = Consolidation
  * **4** = Mixed pattern
  * **5** = Other
  * **9** = No exam / Unknown
### **saturacao**
* **Type:** Categorical
* **Meaning:** Low oxygen saturation (SpO₂ < 95%).
  * **1** = Yes
  * **2** = No
  * **9** = Unknown
---
## Vaccination
### **vacina**
* **Type:** Categorical
* **Meaning:** Influenza vaccination in the same year.
  * **1** = Yes
  * **2** = No
  * **9** = Unknown
---
## Temporal Variables
### **sem_not**
* **Type:** Integer
* **Meaning:** Epidemiological week of **notification**.
### **sem_pri**
* **Type:** Integer
* **Meaning:** Epidemiological week of **symptom onset**.
"""


def sempipes_pipeline():
    data = skrub.var("data").skb.set_description(DATA_DESCRIPTION)

    data_augmented = data.sem_augment(
        nl_prompt="""
        Augment the dataset with additional records similar to the existing records of people from the indegenous minority in Brazil, for whom the prediction model may not work as well as for the majority. The records of the indegenous minority have the `cs_raca` column set to 5 The additional data should improve the prediction quality for them, so make sure that it follows the same distribution as the original data.""",
        name="augment_data",
        number_of_rows_to_generate=600,
        generate_via_code=True,
    )

    X = data_augmented.drop(columns="due_to_covid", errors="ignore").skb.mark_as_X()
    y = (
        data_augmented["due_to_covid"]
        .skb.mark_as_y()
        .skb.set_description("Indicator whether the severe acute respiratory infections originated from COVID-19.")
    )

    X_encoded = X.skb.apply(skrub.TableVectorizer())

    return X_encoded.skb.apply(XGBClassifier(eval_metric="logloss", random_state=42), y=y)


if __name__ == "__main__":
    sempipes.update_config(
        llm_for_code_generation=sempipes.LLM(
            name="gemini/gemini-2.5-flash",
            parameters={"temperature": 0.0},
        ),
    )

    all_data = pd.read_csv("examples/data/covid.csv")
    train, test = train_test_split(all_data, test_size=0.5, random_state=42)

    pipeline = sempipes_pipeline()
    learner = pipeline.skb.make_learner()

    env_train = pipeline.skb.get_data()
    env_train["data"] = train
    learner.fit(env_train)

    env_test = pipeline.skb.get_data()
    env_test["data"] = test
    y_pred = learner.predict_proba(env_test)

    majority_groups = {1, 2, 3, 4}
    test_minority = test[~test.cs_raca.isin(majority_groups)]
    test_minority_labels = test_minority.due_to_covid

    env_test["data"] = test_minority

    predictions = learner.predict_proba(env_test)
    augmented_minority_score = roc_auc_score(test_minority_labels, predictions[:, 1])

    print(f"ROC AUC score for minority group: {augmented_minority_score}")
