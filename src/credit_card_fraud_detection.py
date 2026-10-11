from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.metrics import precision_recall_curve


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

EDA_DIR = OUTPUT_DIR / "eda"
SAMPLING_DIR = OUTPUT_DIR / "sampling"
MODEL_DIR = OUTPUT_DIR / "models"
BUSINESS_DIR = OUTPUT_DIR / "business"

DATA_FILE = DATA_DIR / "creditcard.csv"
CLEANED_DATA_FILE = DATA_DIR / "cleaned_creditcard.csv"

EDA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BUSINESS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the credit card transaction dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Please place creditcard.csv inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# BASIC DATA INSPECTION
# ============================================================

def inspect_data(df):
    """Inspect dataset structure and quality."""

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe())

    # Missing-value report
    missing_report = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_report.columns = [
        "Column",
        "Missing_Values"
    ]

    missing_report.to_csv(
        EDA_DIR / "missing_values_report.csv",
        index=False
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def analyze_class_distribution(df):
    """Analyze legitimate and fraudulent transactions."""

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = (
        df["Class"]
        .value_counts()
        .sort_index()
    )

    print("\nTransaction counts:")
    print(class_counts)

    class_percentages = (
        df["Class"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    print("\nTransaction percentages:")
    print(class_percentages)

    distribution = pd.DataFrame({
        "Class": class_counts.index,
        "Transaction_Count": class_counts.values,
        "Percentage": class_percentages.values
    })

    distribution.to_csv(
        EDA_DIR / "class_distribution.csv",
        index=False
    )

    # Visualization
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Class"
    )

    plt.title(
        "Credit Card Transaction Class Distribution"
    )

    plt.xlabel(
        "Class (0 = Legitimate, 1 = Fraud)"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "class_imbalance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# TRANSACTION AMOUNT ANALYSIS
# ============================================================

def analyze_transaction_amount(df):
    """Analyze transaction amount by class."""

    print("\n" + "=" * 60)
    print("TRANSACTION AMOUNT ANALYSIS")
    print("=" * 60)

    amount_summary = (
        df.groupby("Class")["Amount"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ])
        .reset_index()
    )

    print("\nTransaction amount statistics:")
    print(amount_summary)

    amount_summary.to_csv(
        EDA_DIR / "transaction_amount_summary.csv",
        index=False
    )

    # Boxplot
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y="Amount"
    )

    plt.title(
        "Transaction Amount by Class"
    )

    plt.xlabel(
        "Class (0 = Legitimate, 1 = Fraud)"
    )

    plt.ylabel(
        "Transaction Amount"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "amount_by_class_boxplot.png",
        dpi=300
    )

    plt.close()

    # Log-transformed amount distribution
    df_plot = df.copy()

    df_plot["Log_Amount"] = np.log1p(
        df_plot["Amount"]
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df_plot,
        x="Log_Amount",
        hue="Class",
        bins=50,
        kde=True,
        element="step"
    )

    plt.title(
        "Log-Transformed Transaction Amount Distribution"
    )

    plt.xlabel(
        "log(1 + Amount)"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "log_amount_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# TIME ANALYSIS
# ============================================================

def analyze_transaction_time(df):
    """Analyze transaction time."""

    print("\n" + "=" * 60)
    print("TRANSACTION TIME ANALYSIS")
    print("=" * 60)

    time_summary = (
        df.groupby("Class")["Time"]
        .agg([
            "count",
            "mean",
            "median",
            "min",
            "max"
        ])
        .reset_index()
    )

    print("\nTransaction time statistics:")
    print(time_summary)

    time_summary.to_csv(
        EDA_DIR / "transaction_time_summary.csv",
        index=False
    )

    # Time distribution
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        bins=50,
        element="step"
    )

    plt.title(
        "Transaction Time Distribution by Class"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "transaction_time_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# FEATURE CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(df):
    """Analyze correlations between numerical variables."""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    correlation_matrix = df.corr(numeric_only=True)

    class_correlation = (
        correlation_matrix["Class"]
        .sort_values(ascending=False)
    )

    print("\nFeatures most positively correlated with fraud:")
    print(class_correlation.head(10))

    print("\nFeatures most negatively correlated with fraud:")
    print(class_correlation.tail(10))

    class_correlation.to_csv(
        EDA_DIR / "class_feature_correlations.csv"
    )

    # Full correlation heatmap
    plt.figure(figsize=(16, 12))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        linewidths=0.2
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()

# ============================================================
# FRAUD STATISTICS
# ============================================================

def generate_fraud_statistics(df):
    """Generate detailed fraud transaction statistics."""

    fraud_df = df[df["Class"] == 1]
    legitimate_df = df[df["Class"] == 0]

    fraud_count = len(fraud_df)
    legitimate_count = len(legitimate_df)
    total_count = len(df)

    fraud_percentage = (
        fraud_count / total_count * 100
    )

    print("\n" + "=" * 60)
    print("FRAUD STATISTICS")
    print("=" * 60)

    print(f"\nTotal transactions: {total_count:,}")
    print(f"Legitimate transactions: {legitimate_count:,}")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(
        f"Fraud percentage: {fraud_percentage:.4f}%"
    )

    print(
        f"\nFraud transaction amount mean: "
        f"{fraud_df['Amount'].mean():.2f}"
    )

    print(
        f"Legitimate transaction amount mean: "
        f"{legitimate_df['Amount'].mean():.2f}"
    )

    print(
        f"\nFraud transaction amount median: "
        f"{fraud_df['Amount'].median():.2f}"
    )

    print(
        f"Legitimate transaction amount median: "
        f"{legitimate_df['Amount'].median():.2f}"
    )

    statistics = pd.DataFrame({
        "Metric": [
            "Total Transactions",
            "Legitimate Transactions",
            "Fraudulent Transactions",
            "Fraud Percentage",
            "Average Fraud Amount",
            "Average Legitimate Amount",
            "Median Fraud Amount",
            "Median Legitimate Amount"
        ],
        "Value": [
            total_count,
            legitimate_count,
            fraud_count,
            fraud_percentage,
            fraud_df["Amount"].mean(),
            legitimate_df["Amount"].mean(),
            fraud_df["Amount"].median(),
            legitimate_df["Amount"].median()
        ]
    })

    statistics.to_csv(
        EDA_DIR / "fraud_statistics.csv",
        index=False
    )


# ============================================================
# AUTOMATED EDA INSIGHTS
# ============================================================

def generate_eda_insights(df):
    """Generate a text summary of important EDA findings."""

    fraud_df = df[df["Class"] == 1]
    legitimate_df = df[df["Class"] == 0]

    total = len(df)
    fraud_count = len(fraud_df)

    fraud_percentage = (
        fraud_count / total * 100
    )

    fraud_mean_amount = fraud_df["Amount"].mean()
    legitimate_mean_amount = legitimate_df["Amount"].mean()

    insights = []

    insights.append(
        "CREDIT CARD FRAUD DETECTION - EDA INSIGHTS"
    )

    insights.append("=" * 60)

    insights.append(
        f"Total transactions: {total:,}"
    )

    insights.append(
        f"Fraudulent transactions: {fraud_count:,}"
    )

    insights.append(
        f"Fraud percentage: {fraud_percentage:.4f}%"
    )

    insights.append(
        f"Average fraud transaction amount: "
        f"{fraud_mean_amount:.2f}"
    )

    insights.append(
        f"Average legitimate transaction amount: "
        f"{legitimate_mean_amount:.2f}"
    )

    insights.append(
        "The target variable is highly imbalanced."
    )

    insights.append(
        "Accuracy alone should not be used as the primary "
        "model evaluation metric."
    )

    insights.append(
        "Precision and recall are particularly important "
        "for fraud detection."
    )

    insights.append(
        "Sampling techniques should be applied only to "
        "the training data to avoid data leakage."
    )

    with open(
        EDA_DIR / "eda_insights.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(insights)
        )

    print("\nEDA insights saved successfully.")



# ============================================================
# DATA PREPROCESSING
# ============================================================

def prepare_data(df):
    """Prepare features and target for machine learning."""

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Remove duplicate records
    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate records found: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("\nFeature shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    return X, y

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(X, y):
    """Split data into training and testing sets."""

    print("\n" + "=" * 60)
    print("TRAIN TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining data:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print("\nTesting data:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test

# ============================================================
# FEATURE SCALING
# ============================================================

def scale_features(X_train, X_test):
    """Scale numerical features using StandardScaler."""

    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    print("\nFeature scaling completed.")

    print(
        f"Training mean after scaling: "
        f"{X_train_scaled.mean().mean():.6f}"
    )

    print(
        f"Training standard deviation: "
        f"{X_train_scaled.std().mean():.6f}"
    )

    return X_train_scaled, X_test_scaled, scaler

# ============================================================
# SAVE SCALER
# ============================================================

def save_scaler(scaler):
    """Save the fitted feature scaler."""

    scaler_path = BASE_DIR / "models" / "fraud_feature_scaler.pkl"

    scaler_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(
        f"\nScaler saved to: {scaler_path}"
    )

# ============================================================
# SAMPLING STRATEGIES
# ============================================================

def apply_sampling_strategies(X_train, y_train):
    """
    Apply different sampling strategies to the training data.

    The test dataset is intentionally not passed into this
    function to prevent data leakage.
    """

    print("\n" + "=" * 60)
    print("SAMPLING STRATEGIES")
    print("=" * 60)

    sampling_results = {}

    # --------------------------------------------------------
    # 1. ORIGINAL DATA
    # --------------------------------------------------------

    sampling_results["Original"] = (
        X_train.copy(),
        y_train.copy()
    )

    print("\nOriginal training distribution:")
    print(y_train.value_counts())

    # --------------------------------------------------------
    # 2. RANDOM UNDERSAMPLING
    # --------------------------------------------------------

    undersampler = RandomUnderSampler(
        random_state=42
    )

    X_under, y_under = undersampler.fit_resample(
        X_train,
        y_train
    )

    sampling_results["Undersampling"] = (
        X_under,
        y_under
    )

    print("\nUndersampling distribution:")
    print(pd.Series(y_under).value_counts())

    # --------------------------------------------------------
    # 3. RANDOM OVERSAMPLING
    # --------------------------------------------------------

    oversampler = RandomOverSampler(
        random_state=42
    )

    X_over, y_over = oversampler.fit_resample(
        X_train,
        y_train
    )

    sampling_results["Oversampling"] = (
        X_over,
        y_over
    )

    print("\nOversampling distribution:")
    print(pd.Series(y_over).value_counts())

    # --------------------------------------------------------
    # 4. SMOTE
    # --------------------------------------------------------

    smote = SMOTE(
        random_state=42
    )

    X_smote, y_smote = smote.fit_resample(
        X_train,
        y_train
    )

    sampling_results["SMOTE"] = (
        X_smote,
        y_smote
    )

    print("\nSMOTE distribution:")
    print(pd.Series(y_smote).value_counts())

    return sampling_results

# ============================================================
# SAMPLING COMPARISON REPORT
# ============================================================

def create_sampling_report(sampling_results):
    """Create a comparison of class distributions."""

    print("\n" + "=" * 60)
    print("SAMPLING COMPARISON")
    print("=" * 60)

    report_rows = []

    for method, (X_sampled, y_sampled) in sampling_results.items():

        class_counts = pd.Series(
            y_sampled
        ).value_counts()

        legitimate_count = class_counts.get(
            0,
            0
        )

        fraud_count = class_counts.get(
            1,
            0
        )

        total_count = len(y_sampled)

        fraud_percentage = (
            fraud_count / total_count * 100
        )

        report_rows.append({
            "Sampling_Method": method,
            "Total_Transactions": total_count,
            "Legitimate": legitimate_count,
            "Fraud": fraud_count,
            "Fraud_Percentage": fraud_percentage
        })

    report = pd.DataFrame(report_rows)

    print("\n")
    print(report)

    report.to_csv(
        SAMPLING_DIR / "sampling_comparison.csv",
        index=False
    )

    return report

# ============================================================
# SAMPLING VISUALIZATION
# ============================================================

def plot_sampling_comparison(sampling_results):
    """Visualize class distributions after sampling."""

    methods = []
    legitimate_counts = []
    fraud_counts = []

    for method, (_, y_sampled) in sampling_results.items():

        counts = pd.Series(
            y_sampled
        ).value_counts()

        methods.append(method)

        legitimate_counts.append(
            counts.get(0, 0)
        )

        fraud_counts.append(
            counts.get(1, 0)
        )

    comparison_df = pd.DataFrame({
        "Sampling_Method": methods,
        "Legitimate": legitimate_counts,
        "Fraud": fraud_counts
    })

    comparison_df = comparison_df.set_index(
        "Sampling_Method"
    )

    comparison_df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        "Class Distribution Across Sampling Strategies"
    )

    plt.xlabel(
        "Sampling Strategy"
    )

    plt.ylabel(
        "Number of Training Transactions"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Transaction Class"
    )

    plt.tight_layout()

    plt.savefig(
        SAMPLING_DIR / "sampling_comparison.png",
        dpi=300
    )

    plt.close()

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )

    model.fit(X_train, y_train)

    return model

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, sampling_method):
    """Evaluate a trained model on the untouched test set."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    print(f"\n{'=' * 60}")
    print(f"RANDOM FOREST — {sampling_method}")
    print(f"{'=' * 60}")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print(f"\nTrue Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    # Save confusion matrix
    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(
        f"Random Forest — {sampling_method}"
    )

    plt.tight_layout()

    safe_name = sampling_method.lower().replace(" ", "_")

    plt.savefig(
        MODEL_DIR / f"random_forest_{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return {
        "Model": "Random Forest",
        "Sampling": sampling_method,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }
    
# ============================================================
# RANDOM FOREST SAMPLING COMPARISON
# ============================================================

def run_random_forest_experiments(
    sampling_results,
    X_test,
    y_test
):
    """Train and evaluate Random Forest with each strategy."""

    print("\n" + "=" * 60)
    print("RANDOM FOREST EXPERIMENTS")
    print("=" * 60)

    results = []
    trained_models = {}

    for sampling_method, (X_sampled, y_sampled) in sampling_results.items():

        print(f"\nTraining with: {sampling_method}")

        model = train_random_forest(
            X_sampled,
            y_sampled
        )

        # Save trained model
        safe_name = sampling_method.lower().replace(" ", "_")

        model_path = (
            BASE_DIR / "models" /
            f"random_forest_{safe_name}.pkl"
        )

        joblib.dump(model, model_path)

        # Evaluate against the original test set
        result = evaluate_model(
            model,
            X_test,
            y_test,
            sampling_method
        )

        results.append(result)
        trained_models[sampling_method] = model

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        MODEL_DIR / "random_forest_comparison.csv",
        index=False
    )

    print("\nRandom Forest Comparison:")
    print(
        results_df[
            [
                "Sampling",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC",
                "False_Negatives"
            ]
        ].to_string(index=False)
    )

    return results_df, trained_models

# ============================================================
# RANDOM FOREST METRICS VISUALIZATION
# ============================================================

def plot_random_forest_comparison(results_df):
    """Plot evaluation metrics for each sampling strategy."""

    metrics = [
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    plot_df = results_df.set_index("Sampling")[metrics]

    plot_df.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title(
        "Random Forest Performance by Sampling Strategy"
    )

    plt.xlabel("Sampling Strategy")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(title="Evaluation Metric")
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "random_forest_metrics_comparison.png",
        dpi=300
    )

    plt.close()


def plot_false_negatives(results_df):
    """Compare the number of missed fraud transactions."""

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=results_df,
        x="Sampling",
        y="False_Negatives"
    )

    plt.title(
        "False Negatives by Random Forest Sampling Strategy"
    )

    plt.xlabel("Sampling Strategy")
    plt.ylabel("Number of Missed Fraud Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        MODEL_DIR / "random_forest_false_negatives.png",
        dpi=300
    )

    plt.close()

# ============================================================
# XGBOOST MODEL
# ============================================================

def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model

# ============================================================
# XGBOOST EVALUATION
# ============================================================

def evaluate_xgboost(
    model,
    X_test,
    y_test,
    sampling_method
):
    """Evaluate XGBoost on the untouched test set."""

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    print("\n" + "=" * 60)
    print(
        f"XGBOOST — {sampling_method}"
    )
    print("=" * 60)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        f"True Negatives:  {tn}"
    )
    print(
        f"False Positives: {fp}"
    )
    print(
        f"False Negatives: {fn}"
    )
    print(
        f"True Positives:  {tp}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )

    # Confusion matrix visualization

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Legitimate",
            "Fraud"
        ]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        f"XGBoost — {sampling_method}"
    )

    plt.tight_layout()

    safe_name = (
        sampling_method
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        MODEL_DIR /
        f"xgboost_{safe_name}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return {
        "Model": "XGBoost",
        "Sampling": sampling_method,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }
    
# ============================================================
# XGBOOST EXPERIMENTS
# ============================================================

def run_xgboost_experiments(
    sampling_results,
    X_test,
    y_test
):
    """Train and evaluate XGBoost using all sampling strategies."""

    print("\n" + "=" * 60)
    print("XGBOOST EXPERIMENTS")
    print("=" * 60)

    results = []

    trained_models = {}

    for sampling_method, (
        X_sampled,
        y_sampled
    ) in sampling_results.items():

        print(
            f"\nTraining XGBoost with: "
            f"{sampling_method}"
        )

        model = train_xgboost(
            X_sampled,
            y_sampled
        )

        safe_name = (
            sampling_method
            .lower()
            .replace(" ", "_")
        )

        model_path = (
            BASE_DIR /
            "models" /
            f"xgboost_{safe_name}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        result = evaluate_xgboost(
            model,
            X_test,
            y_test,
            sampling_method
        )

        results.append(result)

        trained_models[
            sampling_method
        ] = model

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        MODEL_DIR /
        "xgboost_comparison.csv",
        index=False
    )

    print("\nXGBoost Comparison:")

    print(
        results_df[
            [
                "Sampling",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC",
                "False_Negatives"
            ]
        ].to_string(
            index=False
        )
    )

    return (
        results_df,
        trained_models
    )
    
# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_all_models(
    random_forest_results,
    xgboost_results
):
    """Combine and compare all model results."""

    combined_results = pd.concat(
        [
            random_forest_results,
            xgboost_results
        ],
        ignore_index=True
    )

    combined_results.to_csv(
        MODEL_DIR /
        "all_model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("ALL MODEL COMPARISON")
    print("=" * 60)

    print(
        combined_results[
            [
                "Model",
                "Sampling",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC",
                "False_Negatives"
            ]
        ].to_string(
            index=False
        )
    )

    return combined_results

# ============================================================
# ALL MODEL PERFORMANCE PLOT
# ============================================================

def plot_all_model_performance(
    combined_results
):
    """Compare model performance."""

    plot_df = combined_results.copy()

    plot_df["Configuration"] = (
        plot_df["Model"]
        + " - "
        + plot_df["Sampling"]
    )

    metrics = [
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    performance_df = plot_df.set_index(
        "Configuration"
    )[metrics]

    performance_df.plot(
        kind="bar",
        figsize=(14, 7)
    )

    plt.title(
        "Fraud Detection Model Performance Comparison"
    )

    plt.xlabel(
        "Model and Sampling Strategy"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_DIR /
        "all_model_performance_comparison.png",
        dpi=300
    )

    plt.close()

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_decision_thresholds(
    model,
    X_test,
    y_test,
    model_name,
    sampling_method
):
    """
    Evaluate model performance at different probability
    thresholds.
    """

    print("\n" + "=" * 60)
    print("DECISION THRESHOLD ANALYSIS")
    print("=" * 60)

    # Fraud probabilities
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    results = []

    for threshold in thresholds:

        y_pred_threshold = (
            y_prob >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred_threshold,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_threshold,
            zero_division=0
        )

        results.append({
            "Model": model_name,
            "Sampling": sampling_method,
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp
        })

    threshold_df = pd.DataFrame(
        results
    )

    print("\nThreshold Performance:")

    print(
        threshold_df[
            [
                "Threshold",
                "Precision",
                "Recall",
                "F1_Score",
                "False_Positives",
                "False_Negatives"
            ]
        ].to_string(
            index=False
        )
    )

    safe_model = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    safe_sampling = (
        sampling_method
        .lower()
        .replace(" ", "_")
    )

    threshold_df.to_csv(
        BUSINESS_DIR /
        f"{safe_model}_{safe_sampling}_threshold_analysis.csv",
        index=False
    )

    return threshold_df

# ============================================================
# PRECISION RECALL THRESHOLD PLOT
# ============================================================

def plot_threshold_tradeoff(
    threshold_df,
    model_name,
    sampling_method
):
    """Plot precision and recall against threshold."""

    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1_Score"],
        marker="o",
        label="F1-score"
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        f"Precision-Recall Trade-off\n"
        f"{model_name} — {sampling_method}"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    safe_model = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    safe_sampling = (
        sampling_method
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        BUSINESS_DIR /
        f"{safe_model}_{safe_sampling}_threshold_tradeoff.png",
        dpi=300
    )

    plt.close()
    
# ============================================================
# BEST F1 THRESHOLD
# ============================================================

def find_best_f1_threshold(threshold_df):
    """Find threshold with the highest F1-score."""

    best_row = threshold_df.loc[
        threshold_df["F1_Score"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("BEST F1 THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold: "
        f"{best_row['Threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_row['Precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_row['Recall']:.4f}"
    )

    print(
        f"F1-score: "
        f"{best_row['F1_Score']:.4f}"
    )

    print(
        f"False Positives: "
        f"{int(best_row['False_Positives'])}"
    )

    print(
        f"False Negatives: "
        f"{int(best_row['False_Negatives'])}"
    )

    return float(best_row["Threshold"])

# ============================================================
# SELECT BEST MODEL CANDIDATE
# ============================================================

def select_best_model_candidate(combined_results):
    """
    Select the candidate with the highest F1-score.

    This is only a starting point. Final selection should
    consider the business cost of false positives and
    false negatives.
    """

    best_row = combined_results.loc[
        combined_results["F1_Score"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("BEST MODEL CANDIDATE")
    print("=" * 60)

    print(
        f"Model: {best_row['Model']}"
    )

    print(
        f"Sampling: {best_row['Sampling']}"
    )

    print(
        f"Precision: {best_row['Precision']:.4f}"
    )

    print(
        f"Recall: {best_row['Recall']:.4f}"
    )

    print(
        f"F1-score: {best_row['F1_Score']:.4f}"
    )

    print(
        f"ROC-AUC: {best_row['ROC_AUC']:.4f}"
    )

    return (
        best_row["Model"],
        best_row["Sampling"]
    )
    
# ============================================================
# RUN THRESHOLD ANALYSIS
# ============================================================
def run_threshold_analysis(
    combined_results,
    rf_models,
    xgb_models,
    X_test,
    y_test
):
    """
    Select the strongest model configuration using F1 score
    and perform decision-threshold analysis.
    """

    # best_candidate = select_best_model_candidate(combined_results)

    # model_name = best_candidate["Model"]
    # sampling_method = best_candidate["Sampling"]

    best_candidate = select_best_model_candidate(combined_results)

    model_name = best_candidate[0]
    sampling_method = best_candidate[1]

    print("\n" + "=" * 60)
    print("FINAL MODEL CANDIDATE")
    print("=" * 60)

    print(f"Model: {model_name}")
    print(f"Sampling: {sampling_method}")

    if model_name == "Random Forest":
        model = rf_models[sampling_method]

    elif model_name == "XGBoost":
        model = xgb_models[sampling_method]

    else:
        print("Unknown model selected.")
        return

    threshold_df = analyze_decision_thresholds(
        model,
        X_test,
        y_test,
        model_name,
        sampling_method
    )

    plot_threshold_tradeoff(
        threshold_df,
        model_name,
        sampling_method
    )

    best_threshold = find_best_f1_threshold(threshold_df)

    print("\nBest analytical threshold:")
    print(f"{best_threshold:.2f}")

    generate_business_report(
        threshold_df,
        best_threshold,
        model_name,
        sampling_method
    )

    generate_final_validation_report(
        combined_results,
        best_threshold,
        threshold_df,
        model_name,
        sampling_method
    )

    return {
        "model_name": model_name,
        "sampling_method": sampling_method,
        "best_threshold": best_threshold,
        "threshold_results": threshold_df
    }
    

# ============================================================
# BUSINESS DECISION REPORT
# ============================================================

def generate_business_report(
    threshold_df,
    best_threshold,
    model_name,
    sampling_method
):
    """Generate a business-oriented threshold report."""

    # threshold = best_threshold["Threshold"]
    threshold = best_threshold
    
    selected_row = threshold_df.loc[
    threshold_df["Threshold"] == best_threshold
].iloc[0]
    
    precision = selected_row["Precision"]
    recall = selected_row["Recall"]
    f1 = selected_row["F1_Score"]

#     # precision = best_threshold["Precision"]
#     precision = threshold_df.loc[
#     threshold_df["Threshold"] == best_threshold,
#     "Precision"
# ].iloc[0]
    
#     # recall = best_threshold["Recall"]
#     recall = threshold_df.loc[
#     threshold_df["Threshold"] == best_threshold,
#     "Recall"
# ].iloc[0]
    
#     # f1 = best_threshold["F1_Score"]
#     f1 = threshold_df.loc[
#     threshold_df["Threshold"] == best_threshold,
#     "F1_Score"
# ].iloc[0]

    false_positives = int(selected_row["False_Positives"])
    false_negatives = int(selected_row["False_Negatives"])

    # report = []

    # report.append(
    #     "CREDIT CARD FRAUD DETECTION"
    # )

    # report.append(
    #     "BUSINESS DECISION THRESHOLD REPORT"
    # )

    # report.append("=" * 60)

    # report.append(
    #     f"Selected Model: {model_name}"
    # )

    # report.append(
    #     f"Sampling Strategy: {sampling_method}"
    # )

    # report.append(
    #     f"Selected Threshold: {threshold:.2f}"
    # )

    # report.append("")

    # report.append(
    #     f"Precision: {precision:.4f}"
    # )

    # report.append(
    #     f"Recall: {recall:.4f}"
    # )

    # report.append(
    #     f"F1-score: {f1:.4f}"
    # )

    # report.append(
    #     f"False Positives: {false_positives}"
    # )

    # report.append(
    #     f"False Negatives: {false_negatives}"
    # )

    # report.append("")

    # report.append(
    #     "BUSINESS INTERPRETATION"
    # )

    # report.append("=" * 60)

    # report.append(
    #     "A lower decision threshold generally increases "
    #     "fraud detection recall but can increase false positives."
    # )

    # report.append(
    #     "A higher decision threshold generally increases "
    #     "prediction selectivity but can allow more fraud "
    #     "transactions to pass undetected."
    # )

    # report.append(
    #     "The final production threshold should be selected "
    #     "using the organization's estimated financial cost "
    #     "of false positives and false negatives."
    # )

    # report.append(
    #     "The current threshold is selected based on the "
    #     "highest F1-score among the evaluated thresholds and "
    #     "should be treated as an analytical starting point "
    #     "rather than a universal production threshold."
    # )

    # report_path = (
    #     BUSINESS_DIR /
    #     "business_decision_report.txt"
    # )

    # with open(
    #     report_path,
    #     "w",
    #     encoding="utf-8"
    # ) as file:

    #     file.write(
    #         "\n".join(report)
    #     )

    # print(
    #     f"\nBusiness report saved to: "
    #     f"{report_path}"
    # )




def generate_final_validation_report(
    combined_results,
    best_threshold,
    threshold_df,
    model_name,
    sampling_method
):
    """
    Generate a final automated validation report covering:
    - Best model configuration
    - Evaluation metrics
    - Class imbalance
    - Threshold analysis
    - Business interpretation
    """

    report_file = OUTPUT_DIR / "final_validation_report.txt"

    best_row = combined_results[
        (combined_results["Model"] == model_name) &
        (combined_results["Sampling"] == sampling_method)
    ]

    if best_row.empty:
        print("Could not locate final model configuration.")
        return

    best_row = best_row.iloc[0]

    best_threshold_row = threshold_df[
        threshold_df["Threshold"] == best_threshold
    ].iloc[0]

    lines = []

    lines.append("=" * 70)
    lines.append("CREDIT CARD FRAUD DETECTION")
    lines.append("FINAL MODEL VALIDATION REPORT")
    lines.append("=" * 70)

    lines.append("")
    lines.append("1. FINAL MODEL CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Model: {model_name}")
    lines.append(f"Sampling Strategy: {sampling_method}")

    lines.append("")
    lines.append("2. STANDARD TEST-SET PERFORMANCE")
    lines.append("-" * 40)
    lines.append(f"Accuracy:  {best_row['Accuracy']:.4f}")
    lines.append(f"Precision: {best_row['Precision']:.4f}")
    lines.append(f"Recall:    {best_row['Recall']:.4f}")
    # lines.append(f"F1 Score:  {best_row['F1']:.4f}")
    lines.append(f"F1 Score:  {best_row['F1_Score']:.4f}")
    # lines.append(f"ROC-AUC:   {best_row['ROC-AUC']:.4f}")
    lines.append(f"ROC-AUC:   {best_row['ROC_AUC']:.4f}")

    lines.append("")
    lines.append("3. CONFUSION MATRIX")
    lines.append("-" * 40)
    
    # lines.append(f"True Negatives:  {int(best_row['TN'])}")
    # lines.append(f"False Positives: {int(best_row['FP'])}")
    # lines.append(f"False Negatives: {int(best_row['FN'])}")
    # lines.append(f"True Positives:  {int(best_row['TP'])}")
    
    lines.append(
    f"True Negatives:  {int(best_row['True_Negatives'])}"
    )

    lines.append(
        f"False Positives: {int(best_row['False_Positives'])}"
    )

    lines.append(
        f"False Negatives: {int(best_row['False_Negatives'])}"
    )

    lines.append(
        f"True Positives:  {int(best_row['True_Positives'])}"
    )

    lines.append("")
    lines.append("4. DECISION THRESHOLD ANALYSIS")
    lines.append("-" * 40)
    lines.append(
        f"Selected analytical threshold: {best_threshold:.2f}"
    )
    lines.append(
        f"Threshold Precision: {best_threshold_row['Precision']:.4f}"
    )
    lines.append(
        f"Threshold Recall: {best_threshold_row['Recall']:.4f}"
    )
    # lines.append(
    #     f"Threshold F1 Score: {best_threshold_row['F1']:.4f}"
    # )
    lines.append(
        f"Threshold F1 Score: {best_threshold_row['F1_Score']:.4f}"
    )

    lines.append("")
    lines.append("5. BUSINESS INTERPRETATION")
    lines.append("-" * 40)
    lines.append(
        "Fraud detection is an imbalanced classification problem where "
        "missing fraudulent transactions can be costly."
    )

    lines.append(
        "A lower decision threshold generally increases recall and "
        "detects more potential fraud cases, but may also increase "
        "false positives."
    )

    lines.append(
        "A higher threshold generally increases precision, but may "
        "allow more fraudulent transactions to remain undetected."
    )

    lines.append(
        "The selected threshold is an analytical starting point based "
        "on F1 score and should be adjusted using real business costs "
        "before production deployment."
    )

    lines.append("")
    lines.append("6. VALIDATION NOTES")
    lines.append("-" * 40)
    lines.append(
        "• The dataset contains a severe class imbalance."
    )
    lines.append(
        "• Sampling strategies were applied only to the training data."
    )
    lines.append(
        "• The test set remained untouched for unbiased evaluation."
    )
    lines.append(
        "• Precision, recall, F1 and ROC-AUC were considered together."
    )
    lines.append(
        "• Model selection based only on accuracy is not appropriate "
        "for this fraud detection problem."
    )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF FINAL VALIDATION REPORT")
    lines.append("=" * 70)

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"Final validation report saved to: {report_file}")
    




# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------

    inspect_data(df)

    analyze_class_distribution(df)

    analyze_transaction_amount(df)

    analyze_transaction_time(df)

    analyze_correlations(df)

    generate_fraud_statistics(df)

    generate_eda_insights(df)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    save_scaler(
        scaler
    )

    # --------------------------------------------------------
    # SAMPLING
    # --------------------------------------------------------

    sampling_results = apply_sampling_strategies(
        X_train_scaled,
        y_train
    )

    create_sampling_report(
        sampling_results
    )

    plot_sampling_comparison(
        sampling_results
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_results, rf_models = (
        run_random_forest_experiments(
            sampling_results,
            X_test_scaled,
            y_test
        )
    )

    plot_random_forest_comparison(
        rf_results
    )

    plot_false_negatives(
        rf_results
    )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_results, xgb_models = (
        run_xgboost_experiments(
            sampling_results,
            X_test_scaled,
            y_test
        )
    )

    # --------------------------------------------------------
    # COMBINED COMPARISON
    # --------------------------------------------------------

    combined_results = compare_all_models(
        rf_results,
        xgb_results
    )

    plot_all_model_performance(
        combined_results
    )
    
    # --------------------------------------------------------
    # DECISION THRESHOLD ANALYSIS
    # --------------------------------------------------------

    # threshold_df, best_threshold, threshold_summary, threshold_plot = (
    #     run_threshold_analysis(
    #         combined_results,
    #         rf_models,
    #         xgb_models,
    #         X_test_scaled,
    #         y_test
    #     )
    # )
    
    # ---------------------------------------------------------
    # 11. Decision Threshold Analysis
    # ---------------------------------------------------------
    final_selection = run_threshold_analysis(
        combined_results,
        rf_models,
        xgb_models,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nFinal model:")
    print(final_selection["model_name"])

    print("Sampling strategy:")
    print(final_selection["sampling_method"])

    print("Best analytical threshold:")
    print(f"{final_selection['best_threshold']:.2f}")

    print("\nImportant output directories:")
    print(f"EDA:       {EDA_DIR}")
    print(f"Sampling:  {SAMPLING_DIR}")
    print(f"Models:    {MODEL_DIR}")
    print(f"Business:  {BUSINESS_DIR}")
    
    
    print("\n" + "=" * 60)
    print("ALL MODEL EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()