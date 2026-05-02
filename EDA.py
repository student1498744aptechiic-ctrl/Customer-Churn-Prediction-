import pandas as pd
import numpy as np

# Load Dataset

df = pd.read_csv("customer_churn_dataset.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# Basic Info

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== DATASET INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# Missing Values

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Handle missing values
df.ffill()

# Remove Irrelevant Columns

if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

# Target Distribution

print("\n===== CHURN DISTRIBUTION =====")
print(df['Churn'].value_counts())

# Convert Categorical → Numerical

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = ['Gender', 'ContractType', 'InternetService', 'PaymentMethod', 'Churn']

for col in categorical_cols:
    df[col] = pd.Series(le.fit_transform(df[col]))

print("\n===== DATA AFTER ENCODING =====")
print(df.head())

# Feature Analysis (Numerical Insights)

print("\n===== TENURE VS CHURN =====")
print(df.groupby('Churn')['Tenure'].mean())

print("\n===== MONTHLY CHARGES VS CHURN =====")
print(df.groupby('Churn')['MonthlyCharges'].mean())

print("\n===== SUPPORT CALLS VS CHURN =====")
print(df.groupby('Churn')['SupportCalls'].mean())

print("\n===== CONTRACT TYPE VS CHURN =====")
print(pd.crosstab(df['ContractType'], df['Churn']))

# Correlation (Only Numeric)

print("\n===== CORRELATION MATRIX =====")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
print(numeric_df.corr())

# Save Cleaned Dataset

df.to_csv("cleaned_churn_dataset.csv", index=False)

print("\nCLEANED DATASET SAVED AS: cleaned_churn_dataset.csv")