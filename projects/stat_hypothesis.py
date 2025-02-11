from spicy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alpha = 0.1

## Null Hypo - there is no diff in goals b/w men and women
## Hypo 1- there is a diff in goals b/w men and women

men1 = pd.read_csv("men_results.csv")
women1 = pd.read_csv("women_results.csv")

men = men1[men1["tournament"] == "FIFA World Cup"]
women = women1[women1["tournament"] == "FIFA World Cup"]

men["total_goals"] = men["home_score"] + men["away_score"]
women["total_goals"] = women["home_score"] + women["away_score"]

men = men[men["date"] > "2002-01-01"]
women = women[women["date"] > "2002-01-01"]



sns.histplot(x="total_goals", data=men, kde=True, bins=5)
plt.title("Men goals")
plt.show()

sns.histplot(x="total_goals", data=women, kde=True, bins=5)
plt.title("Women goals")
plt.show()


u_stat, p_val = stats.mannwhitneyu(men["total_goals"], women["total_goals"], alternative='less')
# should be like this women["total_goals"], men["total_goals"], alternative = "greater"

if p_val < alpha:
    result = "reject"
else:
    result = "fail to reject"
    
result_dict = {"p_val": p_val, "result": result}

print(result_dict)



