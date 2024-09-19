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

## Notes

Top 100 - common genes:

{'OXTR.5021', 'TNXB.7148'}

Top 100 - cpg sites:

cg26354493
