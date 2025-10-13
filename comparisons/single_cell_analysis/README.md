### Preliminary results (cell typing)

#### AIDE (corrected by human)

```commandline
aide data_dir="data/" goal="Predict the cell types. All cells with 'assay' == '10x 5' v2' are used as test, other as train." eval="Macro F1 score and accuracy between predicted and original cell types" 

Macro F1 Score: 0.7591244367948661
Accuracy: 0.8438367307876352
```

#### scVI

```commandline
After 5 runs AVG: Accuracy 0.819, macro F1 0.77
After 5 runs ATD: Accuracy 0.003, macro F1 0.008

```

#### PCA

```commandline
```

#### SemPipes
```commandline
```


### Preliminary results (batch correction)

#### AIDE (corrected by human)

```commandline
aide data_dir="data/" goal="Batch correction for the single-cell RNA data." eval="scib-metrics package batch, bio, and total scores"

Batch correction,Bio conservation,Total, Total (0.5 bio + 0.5 batch)
0.5571585327014565,0.6635855815089846,0.6210147619859734,0.6103720571052206
```

#### scVI

```commandline
Batch correction,Bio conservation,Total, Total (0.5 bio + 0.5 batch)
0.5760111703530759,0.6621300423335963,0.6276824935413882,0.6190706063433361
```

#### PCA

```commandline
Batch correction,Bio conservation,Total
```
