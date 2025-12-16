
F1 score on 0: 0.6966403162055336
F1 score on 1: 0.7065217391304348
F1 score on 2: 0.6956521739130435
F1 score on 3: 0.6373517786561265
F1 score on 4: 0.7262845849802372

Mean final score:  0.6924901185770751 0.0296863995741113

```console
(sempipes-py3.12) (py312) schelter-ldap@sr670-v2-2:~/sempipes$ poetry run python -m experiments.micromodels.minisweagent
Processing split 0
/home/schelter-ldap/.cache/pypoetry/virtualenvs/sempipes-q6T4qXBB-py3.12/lib/python3.12/site-packages/interpret/glassbox/_ebm/_ebm.py:1250: UserWarning: For multiclass we cannot currently visualize pairs and they will be stripped from the global explanations. Set interactions=0 to generate a fully interpretable glassbox model.
  warn(
F1 score on 0: 0.669973544973545
Processing split 1
/home/schelter-ldap/.cache/pypoetry/virtualenvs/sempipes-q6T4qXBB-py3.12/lib/python3.12/site-packages/interpret/glassbox/_ebm/_ebm.py:1250: UserWarning: For multiclass we cannot currently visualize pairs and they will be stripped from the global explanations. Set interactions=0 to generate a fully interpretable glassbox model.
  warn(
F1 score on 1: 0.658068783068783
Processing split 2
/home/schelter-ldap/.cache/pypoetry/virtualenvs/sempipes-q6T4qXBB-py3.12/lib/python3.12/site-packages/interpret/glassbox/_ebm/_ebm.py:1250: UserWarning: For multiclass we cannot currently visualize pairs and they will be stripped from the global explanations. Set interactions=0 to generate a fully interpretable glassbox model.
  warn(
F1 score on 2: 0.6626984126984127
Processing split 3
/home/schelter-ldap/.cache/pypoetry/virtualenvs/sempipes-q6T4qXBB-py3.12/lib/python3.12/site-packages/interpret/glassbox/_ebm/_ebm.py:1250: UserWarning: For multiclass we cannot currently visualize pairs and they will be stripped from the global explanations. Set interactions=0 to generate a fully interpretable glassbox model.
  warn(
F1 score on 3: 0.6501322751322751
Processing split 4
/home/schelter-ldap/.cache/pypoetry/virtualenvs/sempipes-q6T4qXBB-py3.12/lib/python3.12/site-packages/interpret/glassbox/_ebm/_ebm.py:1250: UserWarning: For multiclass we cannot currently visualize pairs and they will be stripped from the global explanations. Set interactions=0 to generate a fully interpretable glassbox model.
  warn(
F1 score on 4: 0.6626984126984127

Mean final score:  0.6607142857142857 0.006520506636966276
```