import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Diabetes\diabetes.csv")
df["case"] = df["class"]

model = logit("case ~ glucose", data=df).fit()
print("\n")
print("Welcome to the Diabetes prediction model")
print("This model aims to predict if a patient has diabetes or not via entering the glucouse level.")
user_input = float(input("Please enter the glucose level for the patient: "))
user_input_data = pd.DataFrame({"glucose": [user_input]})
predict_from_user = model.predict(user_input_data)

predict_from_user_outcome = np.round(predict_from_user).iloc[0]



if predict_from_user_outcome == 1:
    print("The patient has diabetes. ")
else:
    print("The patient does not have diabetes. ")

conf_matrix = model.pred_table()
mosaic(conf_matrix)
plt.show()

sns.regplot(x="glucose", y="case", data=df)
plt.show()