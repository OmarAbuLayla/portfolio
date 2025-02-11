import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# Read the dataset
df = pd.read_csv(r"C:\Users\amoor\OneDrive\Desktop\Data Analysis Certification\Data Science Track\Projects\Project8 Cat Insurance\car_insurance.csv")

# Map the categorical values of "driving_experience" to numeric
experience_map = {
    "0-9y": 0,
    "10-19y": 1,
    "20-29y": 2,
    "30y+": 3
}

# Convert the "driving_experience" column in the dataset to numeric values
df["driving_experience_numeric"] = df["driving_experience"].map(experience_map)

# Train the logistic regression model
model = logit("outcome ~ driving_experience_numeric", data=df).fit()

# Take user input and convert it to numeric
user_input = input("Enter the driving experience (e.g., '0-9y', '10-19y', '20-29y', '30y+'): ")
user_input_numeric = experience_map.get(user_input.strip(), None)

# Check if the input is valid
if user_input_numeric is None:
    print("Invalid input. Please enter one of the following options: '0-9y', '10-19y', '20-29y', '30y+'.")
else:
    # Create a DataFrame for prediction
    user_input_data = pd.DataFrame({"driving_experience_numeric": [user_input_numeric]})
    
    # Predict the outcome based on the input
    predict_from_user = model.predict(user_input_data)
    predict_from_user_outcome = np.round(predict_from_user).iloc[0]
    
    if predict_from_user_outcome == 1:
        print("The insurance will file a claim!")
    else:
        print("Unfortunately, no claim will be filed.")
