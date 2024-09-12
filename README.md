# Logistic Regression

## Requirements

### Data
Get the data. I put mine in `G12`. My file structure:

```sh
├── G12
│   ├── G12_breast_dna-meth.csv
│   └── G12_breast_gene-expr.csv
├── README.md
├── model.py
└── requirements.txt
```

### Code
I am using Python 3.8.19. 

Make a virtual env, activate

```sh
python -m venv env
source env/bin/activate
```

Install dependencies with `pip install -r requirements.txt`.

Run it with

```sh
python model.py
```

So far it does okay, can we improve?

```
Accuracy: 0.97
Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.95      0.86        22
           1       1.00      0.97      0.98       219

    accuracy                           0.97       241
   macro avg       0.89      0.96      0.92       241
weighted avg       0.98      0.97      0.97       241
```


### TODO

- tweak numbers, see if we can improve
- try a model with dna meth dataset
- how to measure? Some ideas

**NOTE**: All untested, just grabbed some snippets of the net, give them a try or research some others.


#### Confusion Matrix

```py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Solid Tissue Normal', 'Primary Tumor'], yticklabels=['Solid Tissue Normal', 'Primary Tumor'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
```

#### ROC Curve

```py
from sklearn.metrics import roc_curve, auc

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
```

#### Precision-Recall Curve

```py
from sklearn.metrics import precision_recall_curve

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```
