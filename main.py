import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE


# Task A
print("====== Task A ======")
df = pd.read_csv("Lung_Cancer_Trends_Realistic.csv", keep_default_na=False)

# General information
print("== First 5 rows of DataFrame ==")
print(df.head())
print("\n=== General information about DataFrame ===")
print(df.info())
print("\n=== Number of missing values ===")
print(df.isnull().sum())
print("\n=== Statistical summary ===")
print(df.describe())

# Visualization: hist
df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation of numeric features")
plt.show()

sns.countplot(x='Chronic_Lung_Disease', data=df)
plt.title("Chronic Disease Distribution")
plt.show()

# Task B
print("\n====== Task B ======")

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df = df.drop_duplicates()

# Drop rows with any missing values
df_cleaned = df.dropna()
df = df_cleaned.copy()

# Task C
# In my code it is not necessary for Random Forest, but I use it for NaiveBayes it is called StandardScaler

# Task D
print("\n====== Task D ======")

print("=== Lung Cancer Stage Distribution ===")
print(df['Lung_Cancer_Stage'].value_counts())

df['Actual_Has_Cancer'] = df['Lung_Cancer_Stage'].map({
    'None': 0,
    'Stage I': 1,
    'Stage II': 1,
    'Stage III': 1,
    'Stage IV': 1
})

print("\nActual distribution:")
print(f"No Cancer (None): {(df['Actual_Has_Cancer'] == 0).sum()}")
print(f"Has Cancer (any stage): {(df['Actual_Has_Cancer'] == 1).sum()}")

risk_factors = [
    'Age', 'Smoking_Status', 'Years_Smoking', 'Cigarettes_Per_Day',
    'Air_Pollution_Level', 'Family_History', 'BMI', 'Gender'
]

X = df[risk_factors].copy()
y = df['Actual_Has_Cancer']

# Creating new encoders
le_smoking = LabelEncoder()
le_pollution = LabelEncoder()
le_family = LabelEncoder()
le_gender = LabelEncoder()
X['Smoking_Status'] = le_smoking.fit_transform(X['Smoking_Status'])
X['Air_Pollution_Level'] = le_pollution.fit_transform(X['Air_Pollution_Level'])
X['Family_History'] = le_family.fit_transform(X['Family_History'])
X['Gender'] = le_gender.fit_transform(X['Gender'])

# Balancing via SMOTE(This is necessary so that it doesn't happen that everyone has cancer or that no one is sick.)
sm = SMOTE(random_state=42)
X_balanced, y_balanced = sm.fit_resample(X, y) # redistributes data

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Test set - No cancer: {(y_test == 0).sum()}, Has cancer: {(y_test == 1).sum()}")

# === MODELS ===

# Random Forest with class weight
# Each tree learns separately on its own random subset.
# If out of 100 trees, 70 said "cancer" and 30 said "no", the final answer is "cancer".
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = rf_model.score(X_test, y_test) # Calculates accuracy: how many of the predictions were correct.

# Naive Bayes with scaled features
# It transforms each numeric feature so that: the mean becomes 0 and the standard deviation becomes 1
scaler = StandardScaler()
# We train the scaler only on X_train! Otherwise, the model would see the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train) # We train the model
nb_predictions = nb_model.predict(X_test_scaled) # Зробити остаточний прогноз класу (0 або 1) для кожного об'єкта з тестової вибірки
nb_accuracy = nb_model.score(X_test_scaled, y_test)

def interpret_model_performance(y_true, y_pred, model_name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nResults for {model_name}:")
    print(f"Correctly identified healthy individuals (True Negative): {tn}")
    print(f"Missed cancer cases (False Negative): {fn}")
    print(f"Correctly identified cancer cases (True Positive): {tp}")
    print(f"Incorrectly flagged healthy individuals as having cancer (False Positive): {fp}")


# === RESULT ===
print("\n=== RESULT ===")
print("=== MODEL PERFORMANCE ===")
# Accuracy — це лише одна точка.
# Але вона не завжди чесно відображає якість класифікації, особливо при незбалансованих даних.
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
print(f"Naive Bayes Accuracy: {nb_accuracy:.3f}")

# It is used to assess the quality of the classification model.
print("\n=== CLASSIFICATION REPORTS ===")
print("Random Forest:")
print(classification_report(y_test, rf_predictions))
print("Naive Bayes:")
print(classification_report(y_test, nb_predictions))

print("\n=== RISK FACTORS IMPORTANCE (Random Forest) ===")
importance_df = pd.DataFrame({
    'Risk_Factor': risk_factors,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in importance_df.iterrows():
    print(f"{row['Risk_Factor']}: {row['Importance']:.3f}")

print("\n=== PREDICTIONS vs REALITY ===")
print(f"Random Forest predicted cancer cases: {rf_predictions.sum()}")
print(f"Naive Bayes predicted cancer cases: {nb_predictions.sum()}")
print(f"Actual cancer cases in test set: {y_test.sum()}")

# === ROC CURVES AND AUC SCORES ===
rf_probs = rf_model.predict_proba(X_test)[:, 1] # По осі X: False Positive Rate (FPR). По осі Y: True Positive Rate (TPR)
nb_probs = nb_model.predict_proba(X_test_scaled)[:, 1] # метод, який повертає ймовірності належності до кожного класу (наприклад, клас 0 і клас 1).

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs) # thresholds — list of thresholds at which fpr and tpr were calculated
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

rf_auc = roc_auc_score(y_test, rf_probs)
nb_auc = roc_auc_score(y_test, nb_probs)

# AUC = Area Under the Curve — площа під ROC-кривою. Це одне число, яке каже:
# 1.0 — ідеальна модель.
# 0.5 — модель не краще випадкового вибору.
# < 0.5 — модель плутає класи
print("\n=== ROC AUC SCORES ===")
print(f"Random Forest AUC: {rf_auc:.3f}")
print(f"Naive Bayes AUC: {nb_auc:.3f}")

interpret_model_performance(y_test, rf_predictions, "Random Forest")
interpret_model_performance(y_test, nb_predictions, "Naive Bayes")

# ROC-GRAPH
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})")
plt.plot(nb_fpr, nb_tpr, label=f"Naive Bayes (AUC = {nb_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')  # базова лінія
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ====== GRAPHS ======
# This allows you to understand where the model is wrong.
cm_rf = confusion_matrix(y_test, rf_predictions)
cm_nb = confusion_matrix(y_test, nb_predictions)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(cm_rf, display_labels=["No Cancer", "Has Cancer"]).plot(ax=axes[0])
axes[0].set_title("Random Forest")
ConfusionMatrixDisplay(cm_nb, display_labels=["No Cancer", "Has Cancer"]).plot(ax=axes[1])
axes[1].set_title("Naive Bayes")
plt.tight_layout()
plt.show()

# What characteristics (risk factors) most influenced the decision of the method?
sns.barplot(data=importance_df, x='Importance', y='Risk_Factor')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Risk Factor")
plt.tight_layout()
plt.show()

# Comparison of model accuracy
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Naive Bayes'],
    'Accuracy': [rf_accuracy, nb_accuracy],
    'Predicted Cancer Cases': [rf_predictions.sum(), nb_predictions.sum()],
    'Actual Cancer Cases': [y_test.sum(), y_test.sum()]  # same for both
})

metrics_df.set_index("Model")[['Accuracy']].plot(kind='bar', legend=True, ylim=(0, 1), title="Model Accuracy")
plt.ylabel("Accuracy")
plt.show()
