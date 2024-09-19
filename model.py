# First we explore the data
import matplotlib.pyplot as plt
import code
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import rnanorm

import torch
import torch.nn as nn
from rnanorm import TMM
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from test import test

pd.set_option("display.max_rows", None)

top_n = 10


def load_gene_expression_data(dataset):
    """
    Load gene expression data and split into test/train
    Params:
        dataset - csv of data to load.
    Returns:
        X - training data
        y - labels
    """
    df = pd.read_csv(dataset) 

    # We drop the first column since it is just sample identifiers, not useful for machine learning
    df = df.drop(df.columns[0], axis=1)
    df = df.dropna(axis=1)

    # Drop the label "Tumour" or "Normal Tissue" from the feature set.
    # The whole point is the features do **not** include the "answer"
    X = df.drop(df.columns[0], axis=1)

    # now we do the TMM normalization
    tmm = TMM().fit(X)
    norm_factors = tmm.get_norm_factors(X)
    normalized_array = tmm.transform(X)
    normalized_data = pd.DataFrame(normalized_array, index=X.index, columns=X.columns)

    # y is the labels. This is **only** the Tumour or Normal Tissue labels.
    y = df["Label"]

    # Convert labels to numeric values.
    y = y.map({"Primary Tumor": 1, "Solid Tissue Normal": 0})

    return normalized_data, y


def load_dna_meth_data(dataset):
    """
    Load data meth data and split into test/train
    Params:
        dataset - csv of data to load.
    Returns:
        X - training data
        y - labels
    """
    df = pd.read_csv(dataset)

    # We drop the first column since it is just sample identifiers, not useful for machine learning
    df = df.drop(df.columns[0], axis=1)
    df = df.dropna(axis=1)

    # Drop the label "Tumour" or "Normal Tissue" from the feature set.
    # The whole point is the features do **not** include the "answer"
    X = df.drop(df.columns[0], axis=1)

    y = df["Label"]

    # Convert labels to numeric values.
    y = y.map({"Primary Tumor": 1, "Solid Tissue Normal": 0})

    return X, y


def logistic_regression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression

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
    # Get feature coefficients
    coefficients = model.coef_[0]  # Get the coefficients for the first class
    feature_importances = pd.DataFrame(
        {
            "Feature": X_train.columns,  # Assuming X_train is a DataFrame
            "Coefficient": coefficients,
        }
    )

    # Sort the DataFrame by the absolute value of coefficients
    feature_importances["Absolute Coefficient"] = feature_importances[
        "Coefficient"
    ].abs()
    feature_importances = feature_importances.sort_values(
        by="Absolute Coefficient", ascending=False
    )

    # Get the top N features
    top_features = feature_importances.head(top_n)
    print("Top N Features (Logistic Regression):")
    print(top_features)
    return y_pred, top_features["Feature"]


def random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=400, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame(
        {
            "Feature": X_train.columns,  # Assuming X_train is a DataFrame
            "Importance": importances,
        }
    )

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values(
        by="Importance", ascending=False
    )

    # Get the top N features
    top_features = feature_importances.head(top_n)
    print("Top N Features (Random Forest):")
    print(top_features)
    rf_feats = top_features["Feature"]
    return y_pred, model, rf_feats


def forwardfeed_neural_net(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # convert to PyTorch tensors for compat
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)

    # loading
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # copy pasted this from the net
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)  # Input layer
            self.fc2 = nn.Linear(64, 32)  # Hidden layer
            self.fc3 = nn.Linear(32, 1)  # Output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))  # Activation for first layer
            x = torch.relu(self.fc2(x))  # Activation for second layer
            x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
            return x

    input_size = X_train_scaled.shape[1]  # Number of features
    print(">>>>>>>>>", input_size)
    model = SimpleNN(input_size)

    criterion = (
        nn.BCELoss()
    )  # Binary Cross-Entropy Loss - TODO: What does this even do? Grabbed from the docs, seems to work
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        y_test_pred = model(X_test_tensor).squeeze()
        y_test_pred_binary = (
            y_test_pred > 0.5
        ).float()  # Convert probabilities to binary predictions

    # Calculate accuracy
    accuracy = (y_test_pred_binary == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy:.2f}")

    y_test_numpy = y_test_tensor.numpy()
    y_test_pred_numpy = y_test_pred_binary.numpy()
    report = classification_report(
        y_test_numpy,
        y_test_pred_numpy,
        target_names=["Solid Tissue Normal", "Primary Tumor"],
    )
    print("Classification Report:")
    print(report)

    return y_test_pred_numpy


def run_all_models(X, y, label, outdir):
    print(f"\n=== Running for dataset: {label} ===\n")
    # ==================================
    # Run all the models
    # model result for graph
    model_results = []

    print(f"\n=== Logistic Regression ===\n")
    class_labels = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred, lg_feats = logistic_regression(X_train, X_test, y_train, y_test)
    model_results.append({"model": "Logistic Regression", "y_pred": y_pred})

    print(f"\n=== Random Forest ===\n")
    y_pred, rf, rf_feats = random_forest(X_train, X_test, y_train, y_test)
    model_results.append({"model": "Random Forest", "y_pred": y_pred})

    print(f"\n=== Forward Feed Neural Network ===\n")
    y_pred = forwardfeed_neural_net(X_train, X_test, y_train, y_test)
    model_results.append({"model": "Neural Network", "y_pred": y_pred})

    report_list = []

    for model_data in model_results:
        model_name = model_data["model"]
        y_pred = model_data["y_pred"]
        report = classification_report(
            y_test, y_pred, output_dict=True
        )  # Get report as a dictionary
        report_df = pd.DataFrame(
            report
        ).transpose()  # Convert to DataFrame and transpose
        report_df["Model"] = model_name  # Add model name as a column
        report_list.append(report_df)

    final_report = pd.concat(report_list)

    # Reset index for better plotting
    final_report.reset_index(inplace=True)

    # Define metrics to plot
    metrics = ["precision", "recall", "f1-score"]

    # Create a new DataFrame for plotting
    plot_data = final_report[
        final_report["index"].isin(["0", "1"])
    ]  # Keep only class metrics
    plot_data = plot_data.pivot(index="Model", columns="index", values=metrics)

    # Create a new DataFrame for plotting
    plot_data = final_report[
        final_report["index"].isin(["0", "1"])
    ]  # Keep only class metrics
    plot_data = plot_data.pivot(index="Model", columns="index", values=metrics)

    # Map class indices to class names
    class_labels = {0: "Solid Tissue Normal", 1: "Primary Tumor"}

    # Rename columns to include class names
    plot_data.columns = pd.MultiIndex.from_tuples(
        [(metric, class_labels[int(cls)]) for metric, cls in plot_data.columns]
    )

    for i, metric in enumerate(metrics):
        # Create a new figure for each metric
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the data
        plot_data[metric].plot(
            kind="bar", ax=ax, color=["orange", "green"], legend=True
        )

        metric_data = plot_data[metric]
        bars = metric_data.plot(
            kind="bar", ax=ax, color=["orange", "green"], legend=True
        )

        for bar in bars.patches:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.05),
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(f"{metric.capitalize()} {label}")
        ax.set_xlabel("Models")
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
        ax.grid(axis="y")
        ax.legend(title="Classes", loc="lower right")

        # Adjust layout
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(f"figs/{outdir}_{metric.capitalize()}")



X, y = load_gene_expression_data("./G12/G12_breast_gene-expr.csv")
run_all_models(X, y, "Gene data", "gene")

X, y = load_dna_meth_data("./G12/G12_breast_dna-meth.csv")
run_all_models(X, y, "DNA Methylation", "meth")

# X, y = load_gene_expression_data("./G12/mystery_gene-expr.csv")
# run_all_models(X, y, "Gene data (Mystery)", "gene")
# 
# X, y = load_dna_meth_data("./G12/mystery_dna-meth.csv")
# run_all_models(X, y, "DNA Methylation (Mystery)", "meth")
