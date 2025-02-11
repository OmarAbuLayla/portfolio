from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

# Load the dataset
df = pd.read_csv("soil_measures.csv")

# Print unique crop types (for reference)
print(df.crop.unique())

# Separate features (X) and target (y)
X = df.drop(columns="crop")
y = df["crop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Dictionary to store feature performance
feature_perf = {}

# Evaluate feature "N"
log_reg_N = LogisticRegression(multi_class="multinomial", max_iter=1000)
log_reg_N.fit(X_train[["N"]], y_train)
y_pred_N = log_reg_N.predict(X_test[["N"]])
f1_N = metrics.f1_score(y_test, y_pred_N, average="weighted")
feature_perf["N"] = f1_N
print(f"F1-score for N: {f1_N}")

# Evaluate feature "P"
log_reg_P = LogisticRegression(multi_class="multinomial", max_iter=1000)
log_reg_P.fit(X_train[["P"]], y_train)
y_pred_P = log_reg_P.predict(X_test[["P"]])
f1_P = metrics.f1_score(y_test, y_pred_P, average="weighted")
feature_perf["P"] = f1_P
print(f"F1-score for P: {f1_P}")

# Evaluate feature "K"
log_reg_K = LogisticRegression(multi_class="multinomial", max_iter=1000)
log_reg_K.fit(X_train[["K"]], y_train)
y_pred_K = log_reg_K.predict(X_test[["K"]])
f1_K = metrics.f1_score(y_test, y_pred_K, average="weighted")
feature_perf["K"] = f1_K
print(f"F1-score for K: {f1_K}")

# Evaluate feature "ph"
log_reg_ph = LogisticRegression(multi_class="multinomial", max_iter=1000)
log_reg_ph.fit(X_train[["ph"]], y_train)
y_pred_ph = log_reg_ph.predict(X_test[["ph"]])
f1_ph = metrics.f1_score(y_test, y_pred_ph, average="weighted")
feature_perf["ph"] = f1_ph
print(f"F1-score for ph: {f1_ph}")

# Find the feature with the highest F1-score
best_feature = max(feature_perf, key=feature_perf.get)
best_f1 = feature_perf[best_feature]

# Store in best_predictive_feature dictionary
best_predictive_feature = {best_feature: best_f1}
print("Best predictive feature:", best_predictive_feature)