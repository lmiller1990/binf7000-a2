# First we explore the data
import pandas as pd
from sklearn.model_selection import train_test_split
from boruta import BorutaPy


def read_and_summarize(f: str):
    data = pd.read_csv(f)

    samples, features = data.shape
    # print(f"{f} has {samples} samples and {features} features")
    # print(f"The summary statistics are:  \n{data.describe()}")

    return data


def drop_features_below_threshold(df, percent):
    """Drop a percentage of features based on value
    Eg: 0.7 will drop the bottom 70% of features,
    leaving the remaining 30%.

    Params:
        percent: float (0 - 1)
    """

    percent = min(percent, 1)

    # ignore labels
    feature_df = df.iloc[:, 1:]

    mean_values = feature_df.mean()

    # determine the number of columns to drop
    num_columns_to_drop = int(len(mean_values) * (percent / 100))

    # get the columns to drop based on the lowest mean values
    columns_to_drop = mean_values.nsmallest(num_columns_to_drop).index.tolist()

    df_trimmed = df.drop(columns=columns_to_drop)
    print(f"Dropped columns: {len(columns_to_drop)} out of {len(df.columns)}")

    return df_trimmed


def load_data():
    """
    Load gene expression data and split into test/train
    Returns:
    """
    df = read_and_summarize("./G12/G12_breast_gene-expr.csv")
    # meth = read_and_summarize("./G12/G12_breast_dna-meth.csv")

    # We drop the first column since it is just sample identifiers, not useful for machine learning
    df = df.drop(df.columns[0], axis=1)

    df = drop_features_below_threshold(df, percent=0.8)

    # Drop the label "Tumour" or "Normal Tissue" from the feature set.
    # The whole point is the features do **not** include the "answer"
    X = df.drop(df.columns[0], axis=1)
    import code

    code.interact(local=dict(globals(), **locals()))

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

    return X_train, X_test, y_train, y_test, X, y


X_train, X_test, y_train, y_test, X, y = load_data()


def logistic_regression():
    print("\n=== Logistic Regression ===\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

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


def random_forest():
    print("\n=== Random Forest ===\n")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    return model


def run_boruta(estimator):
    boruta_selector = BorutaPy(
        estimator=estimator,
        n_estimators="auto",  # type: ignore based on estimator
        verbose=2,
        random_state=42,
    )

    boruta_selector.fit(X.values, y.values)
    selected_features = X.columns[boruta_selector.support_].to_list()

    print("Selected Features:")
    print(selected_features)


logistic_regression()
rf = random_forest()
# Warning: This takes a very long time. Let's preprocess
run_boruta(rf)
