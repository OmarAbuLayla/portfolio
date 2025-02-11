from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

# Load the dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Features to test
features_to_test = ["anaemia", "creatinine_phosphokinase", "diabetes", "smoking", "high_blood_pressure"]

# Separate features (X) and target (y)
X = df.drop(columns="DEATH_EVENT")
y = df["DEATH_EVENT"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Dictionary to store feature performance
feature_perf = {}

# Testing for anaemia
log_reg_AN = LogisticRegression(max_iter=1000,class_weight="balanced")   # Ensure convergence
log_reg_AN.fit(X_train[["anaemia"]], y_train)  # Use double brackets for 2D array
y_pred_AN = log_reg_AN.predict(X_test[["anaemia"]])
f1_AN = metrics.f1_score(y_test, y_pred_AN)  # Calculate F1 score
feature_perf["anaemia"] = f1_AN  # Use consistent naming
print(f"The F1 score for anaemia is {f1_AN}.")

# Testing for creatinine_phosphokinase
log_reg_CP = LogisticRegression(max_iter=1000,class_weight="balanced")  # Ensure convergence
log_reg_CP.fit(X_train[["creatinine_phosphokinase"]], y_train)  # Use double brackets for 2D array
y_pred_CP = log_reg_CP.predict(X_test[["creatinine_phosphokinase"]])
f1_CP = metrics.f1_score(y_test, y_pred_CP)  # Calculate F1 score
feature_perf["creatinine_phosphokinase"] = f1_CP  # Use consistent naming
print(f"The F1 score for creatinine_phosphokinase is {f1_CP}.")

# Testing for high_blood_pressure
log_reg_HB = LogisticRegression(max_iter=1000,class_weight="balanced")  # Ensure convergence
log_reg_HB.fit(X_train[["high_blood_pressure"]], y_train)  # Use double brackets for 2D array
y_pred_HB = log_reg_HB.predict(X_test[["high_blood_pressure"]])
f1_HB = metrics.f1_score(y_test, y_pred_HB)  # Calculate F1 score
feature_perf["high_blood_pressure"] = f1_HB  # Use consistent naming
print(f"The F1 score for high_blood_pressure is {f1_HB}.")


# Testing for diabetes
log_reg_diabetes = LogisticRegression(max_iter=1000,class_weight="balanced")  # Ensure convergence
log_reg_diabetes.fit(X_train[["diabetes"]], y_train)  # Use double brackets for 2D array
y_pred_diabetes = log_reg_diabetes.predict(X_test[["diabetes"]])
f1_diabetes = metrics.f1_score(y_test, y_pred_diabetes)  # Calculate F1 score
feature_perf["diabetes"] = f1_diabetes  # Use consistent naming
print(f"The F1 score for diabetes is {f1_diabetes}.")



# Testing for smoking
log_reg_smoking = LogisticRegression(max_iter=1000,class_weight="balanced")  # Ensure convergence
log_reg_smoking.fit(X_train[["smoking"]], y_train)  # Use double brackets for 2D array
y_pred_smoking = log_reg_smoking.predict(X_test[["smoking"]])
f1_smoking = metrics.f1_score(y_test, y_pred_smoking)  # Calculate F1 score
feature_perf["smoking"] = f1_smoking  # Use consistent naming
print(f"The F1 score for smoking is {f1_smoking}.")


# Testing for all features
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
f1 = metrics.f1_score(y_test, y_pred)
print(f"The F1 score using all features is {f1}.")



# Train the model on user input
log_reg_3 = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg_3.fit(X_train[["smoking", "diabetes", "high_blood_pressure"]], y_train)
y_pred_3 = log_reg_3.predict(X_test[["smoking", "diabetes", "high_blood_pressure"]])
f1_3 = metrics.f1_score(y_test, y_pred_3)
print(f"The F1 score using smoking, diabetes, and high_blood_pressure is {f1_3}.")

print("\n")
print("\n")
print("\n")

X = df[["smoking", "diabetes", "high_blood_pressure"]]
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=12)
log = LogisticRegression(max_iter=1000, class_weight="balanced")
log.fit(X_train, y_train)

smoking = int(input("Does the patient smoke? (1 for Yes, 0 for No): "))
print("\n")
diabetes = int(input("Does the patient have diabetes? (1 for Yes, 0 for No): "))
print("\n")
high_blood_pressure = int(input("Does the patient have high blood pressure? (1 for Yes, 0 for No): "))
print("\n")
user_input = pd.DataFrame({
    "smoking": [smoking],
    "diabetes": [diabetes],
    "high_blood_pressure": [high_blood_pressure]
    })

pred = log.predict(user_input)
if pred == 1:
    print("High risk of Heart failure detected.")
else:
    print("Low risk of heart failure.")

