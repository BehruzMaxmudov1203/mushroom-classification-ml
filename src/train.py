import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def load_data(data_path: str = "data/mushrooms.csv") -> pd.DataFrame:
    """Load dataset from local file or remote fallback URL."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        url = "https://raw.githubusercontent.com/alishermutalov/praktikum-datasets/refs/heads/praktikum/mushrooms.csv"
        df = pd.read_csv(url)

    # Drop zero-variance column (veil-type only has one unique value 'p')
    if "veil-type" in df.columns:
        df = df.drop("veil-type", axis=1)

    return df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split and encode data cleanly.
    Encoders are fitted strictly on the train set to prevent data leakage.
    """
    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Encode target: 'e' (edible) -> 0, 'p' (poisonous) -> 1
    target_le = LabelEncoder()
    y_train_encoded = target_le.fit_transform(y_train)
    y_test_encoded = target_le.transform(y_test)

    feature_encoders = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    for col in X_train.columns:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        X_test_encoded[col] = le.transform(X_test[col])
        feature_encoders[col] = le

    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, target_le


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple classifiers and evaluate on the test split."""
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    trained_models = {}

    print("=" * 55)
    print("           MODEL EVALUATION RESULTS")
    print("=" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {"Accuracy": acc, "F1-Score": f1}
        trained_models[name] = model

        print(f"{name:<24} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    print("=" * 55)

    results_df = pd.DataFrame(results).T
    best_model_name = results_df["F1-Score"].idxmax()
    print(f"\n🏆 Best Performing Model: {best_model_name}")

    return results_df, trained_models, best_model_name


def save_visualizations(df, results_df, best_model, X_test, y_test, output_dir: str = "figures"):
    """Generate and save publication-quality plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Exploratory Data Analysis: Class Distribution & Odor Impact
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    palette = ["#2ecc71", "#e74c3c"]
    ax = sns.countplot(data=df, x="class", hue="class", palette=palette, legend=False)
    plt.title("Class Distribution (e = Edible, p = Poisonous)", fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x="odor", hue="class", palette=palette)
    plt.title("Impact of Odor on Toxicity", fontweight="bold")
    plt.xlabel("Odor Category Code")
    plt.ylabel("Count")

    plt.tight_layout()
    eda_path = os.path.join(output_dir, "eda_distribution.png")
    plt.savefig(eda_path, dpi=300)
    plt.close()
    print(f"Saved EDA chart to: {eda_path}")

    # 2. Model Performance Comparison
    plt.figure(figsize=(9, 5))
    results_df.plot(kind="bar", figsize=(9, 5), color=["#3498db", "#9b59b6"])
    plt.title("Model Performance Comparison (Accuracy vs F1-Score)", fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0.9, 1.05)
    plt.xticks(rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved Model Comparison chart to: {comp_path}")

    # 3. Confusion Matrix for Best Model
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Edible (0)", "Poisonous (1)"],
        yticklabels=["Edible (0)", "Poisonous (1)"],
    )
    plt.title("Confusion Matrix (Best Model)", fontweight="bold")
    plt.ylabel("Actual Status")
    plt.xlabel("Predicted Status")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Mushroom Toxicity Classification ML Pipeline")
    parser.add_argument("--data_path", type=str, default="data/mushrooms.csv", help="Path to mushrooms.csv")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save evaluation charts to figures/")
    args = parser.parse_args()

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test, target_le = preprocess_data(df)
    results_df, trained_models, best_model_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    if args.save_plots:
        save_visualizations(df, results_df, trained_models[best_model_name], X_test, y_test)


if __name__ == "__main__":
    main()
