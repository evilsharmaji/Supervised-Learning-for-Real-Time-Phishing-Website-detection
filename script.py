# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

sns.set_theme(style="whitegrid")

# Step 2: Load and Prepare Data
df = pd.read_csv("phishing_data.csv")
df = df.drop(columns=["url"], errors='ignore')

if 'status' in df.columns:
    df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})
df.replace({'zero': 0, 'one': 1}, inplace=True)

# Display Sample Data
df.head().style.set_table_styles([
    {'selector': 'th', 'props': [('background-color', '#f7f7f9'), ('color', 'black')]},
    {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
])

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(6, 4))
sns.countplot(x='status', data=df, palette='Set2')
plt.title("Distribution of Website Status")
plt.xlabel("Status (0: Legitimate, 1: Phishing)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 4: Data Splitting and Scaling
X = df.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Model Training
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Step 6: Model Evaluation and Visualization
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    fpr, tpr, _ = roc_curve(y_test, probas)
    auc_score = auc(fpr, tpr)
    results[name] = {'accuracy': acc, 'fpr': fpr, 'tpr': tpr, 'auc': auc_score}

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, preds))

# Step 7: ROC Curve Comparison
plt.figure(figsize=(10, 6))
for name, res in results.items():
    plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves of Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Accuracy Comparison
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]

plt.figure(figsize=(8, 5))
sns.barplot(x=accuracies, y=model_names, palette="pastel")
plt.xlabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# Step 9: Real-Time Prediction Simulation
best_model = models['Random Forest']
sample_input = X_test[0].reshape(1, -1)
prediction = best_model.predict(sample_input)
print("\nReal-Time Prediction Result:", "Phishing" if prediction[0] else "Legitimate")

# exit
