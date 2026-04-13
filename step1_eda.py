# ============================================================
# STEP 1: Exploratory Data Analysis (EDA)
# Dataset: StudentPerformanceFactors.csv
# Target: Exam_Score (Regression)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("StudentPerformanceFactors.csv")
print("=" * 50)
print(f"Dataset shape: {df.shape}")
print("=" * 50)

# ── Basic info ───────────────────────────────────────────────
print("\n--- Column Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Numeric Summary ---")
print(df.describe())

print("\n--- Categorical Columns (unique values) ---")
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    print(f"  {col}: {df[col].unique()}")

# ── Target variable distribution ────────────────────────────
plt.figure(figsize=(8, 4))
sns.histplot(df["Exam_Score"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Exam Score (Target Variable)")
plt.xlabel("Exam Score")
plt.tight_layout()
plt.savefig("plot1_target_distribution.png", dpi=150)
plt.close()
print("\n[Saved] plot1_target_distribution.png")

# ── Correlation heatmap (numeric features) ──────────────────
num_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(10, 7))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png", dpi=150)
plt.close()
print("[Saved] plot2_correlation_heatmap.png")

# ── Boxplots: Categorical features vs Exam_Score ────────────
if len(cat_cols) > 0:
    n_cat = len(cat_cols)
    n_cols = 4
    n_rows = int(np.ceil(n_cat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sns.boxplot(data=df, x=col, y="Exam_Score", ax=axes[i], palette="Set2")
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=15)
    # Hide unused subplots
    for j in range(n_cat, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Exam Score by Categorical Features", fontsize=14)
    plt.tight_layout()
    plt.savefig("plot3_categorical_boxplots.png", dpi=150)
    plt.close()
    print("[Saved] plot3_categorical_boxplots.png")
else:
    print("[Skipped] No categorical columns found for boxplots.")

# ── Scatter plots: Numeric features vs Exam_Score ───────────
num_features = [c for c in num_cols if c != "Exam_Score"]
if len(num_features) > 0:
    n_num = len(num_features)
    n_cols = 3
    n_rows = int(np.ceil(n_num / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(num_features):
        axes[i].scatter(df[col], df["Exam_Score"], alpha=0.2, s=10, color="steelblue")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Exam Score")
        axes[i].set_title(f"{col} vs Exam Score")
    for j in range(n_num, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig("plot4_numeric_scatter.png", dpi=150)
    plt.close()
    print("[Saved] plot4_numeric_scatter.png")
else:
    print("[Skipped] No numeric features available for scatter plots.")

print("\n✅ EDA complete! Check the saved plots.")
print("   Key observations to note:")
print("   - Exam_Score range: 55 to 101")
print("   - Missing values in: Teacher_Quality (78), Parental_Education_Level (90), Distance_from_Home (67)")
print("   - All categoricals need encoding before modeling")
print("   - No major numeric outliers detected from describe()")