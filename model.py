import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load Cleaned Dataset
df = pd.read_csv("cleaned_churn_dataset.csv")

print("\n===== DATA LOADED =====")
print(df.head())

# Split Features & Target

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

# KNN Model

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

# Evaluation Function

def evaluate_model(name, y_test, y_pred):
    print(f"\n===== {name} RESULTS =====")
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return acc

# Evaluate Both Models

dt_acc = evaluate_model("Decision Tree", y_test, dt_pred)
knn_acc = evaluate_model("KNN", y_test, knn_pred)

# Comparison

print("\n===== MODEL COMPARISON =====")

results = pd.DataFrame({
    "Model": ["Decision Tree", "KNN"],
    "Accuracy": [dt_acc, knn_acc]
})

print(results)

# Bar Chart Comparison
plt.figure()
plt.bar(results["Model"], results["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# Final Conclusion

best_model = results.loc[results["Accuracy"].idxmax()]

print("\n===== FINAL RESULT =====")
print(f"Best Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.4f}")