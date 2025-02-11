import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Load data
df = pd.read_csv("rental_info.csv")

# Convert dates to datetime format and calculate rental length
df['rental_date'] = pd.to_datetime(df['rental_date'])
df['return_date'] = pd.to_datetime(df['return_date'])
df["rental_length_days"] = (df["return_date"] - df["rental_date"]).dt.days

# Define new features and target
X = df[["amount", "release_year", "rental_rate", "length"]]
y = df["rental_length_days"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Train a new KNeighborsRegressor model using the reduced features
best_model = KNeighborsRegressor(n_neighbors=5)
best_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
best_mse = MSE(y_test, y_pred)
print(f"Best model MSE (using 4 features): {best_mse:.3f}")

print("Enter the following details:")
amount = float(input("Amount paid: "))
release_year = int(input("Release year of the movie: "))
rental_rate = float(input("Rental rate: "))
length = int(input("Length of the movie (in minutes): "))

user_data = np.array([[amount, release_year, rental_rate, length]])
predicted_duration = best_model.predict(user_data)[0]

print(f"Predicted rental duration: {predicted_duration:.2f} days")

