#!/bin/bash
for i in $(seq 0 9); do
#  poetry run python -m "experiments.globalprogramsynthesis.gemini25pro_intermediate_${i}" \
#    2> "experiments/globalprogramsynthesis/gemini25pro_intermediate_${i}_errors.txt"
  poetry run python -m "experiments.globalprogramsynthesis.gemini25flash_traintest_${i}" \
    2> "experiments/globalprogramsynthesis/gemini25flash_traintest_${i}_errors.txt"

done

