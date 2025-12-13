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
