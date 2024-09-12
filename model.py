# First we explore the data
import pandas as pd
import code
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def read_and_summarize(f: str):
    data = pd.read_csv(f)

    samples, features = data.shape
    print(f"{f} has {samples} samples and {features} features")
    # print(f"The list of columns is - \n   {list(data.columns)}\n")
    print(f"The summary statistics are:  \n{data.describe()}")

    return data


df = read_and_summarize("./G12/G12_breast_gene-expr.csv")
# meth = read_and_summarize("./G12/G12_breast_dna-meth.csv")

# We drop the first column since it is just sample identifiers, not useful for machine learning
df = df.drop(df.columns[0], axis=1)

# Drop the label "Tumour" or "Normal Tissue" from the feature set.
# The whole point is the features do **not** include the "answer"
X = df.drop(df.columns[0], axis=1)

# y is the labels. This is **only** the Tumour or Normal Tissue labels.
y = df["Label"]  # Labels

# Convert labels to numeric values.
y = y.map({"Primary Tumor": 1, "Solid Tissue Normal": 0})

# test_size 0.2 means 80% used for training, 20% used for testing
# we should tweak it
# random_state is not necessary but makes the random seed the same so the
# algorithm runs the same every time. 
# stratify: read here => https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# new model instance
# NOTE the actual "training" is when you run `fit()`
model = LogisticRegression(max_iter=1000)

# run the model
# TODO: experiment with different iterations
model.fit(X_train, y_train)

# evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
