import pandas as pd
import numpy as np
import random

# Number of records
n = 500

data = []

for i in range(n):
    customer_id = f"CUST{i+1}"
    
    gender = random.choice(["Male", "Female"])
    
    age = random.randint(18, 70)
    
    tenure = random.randint(1, 72)
    
    contract = random.choice(["Monthly", "Yearly"])
    
    internet = random.choice(["DSL", "Fiber", "None"])
    
    support_calls = random.randint(0, 10)
    
    payment = random.choice(["Card", "Cash", "Online"])
    
    monthly_charges = random.randint(20, 120)
    
    total_charges = monthly_charges * tenure
    
    # Churn Logic (IMPORTANT)
    churn = "No"
    
    if (
        tenure < 12 and 
        monthly_charges > 70 and 
        support_calls > 3 and 
        contract == "Monthly"
    ):
        churn = "Yes"
    elif support_calls > 7:
        churn = "Yes"
    elif tenure > 24 and contract == "Yearly":
        churn = "No"
    else:
        churn = random.choice(["Yes", "No"])
    
    data.append([
        customer_id, gender, age, tenure,
        monthly_charges, total_charges,
        contract, internet, support_calls,
        payment, churn
    ])

columns = [
    "CustomerID", "Gender", "Age", "Tenure",
    "MonthlyCharges", "TotalCharges",
    "ContractType", "InternetService",
    "SupportCalls", "PaymentMethod", "Churn"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("customer_churn_dataset.csv", index=False)

print("Dataset created successfully!")