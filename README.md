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


