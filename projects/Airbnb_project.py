import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("airbnb_price.csv")
df2 = pd.read_excel(airbnb_room_type.xlsx")
df3 = pd.read_csv(airbnb_last_review.tsv", sep="\t")

df_merge = pd.merge(df1,df2, on= "listing_id", how="inner")
df = pd.merge(df_merge, df3, on="listing_id", how="inner")

print(df.columns)

df["last_review"] = pd.to_datetime(df["last_review"])

first_reviewed = df["last_review"].min()
last_reviewed = df["last_review"].max()

df["room_type"] = df["room_type"].str.lower()
df_rooms = df[df["room_type"] == "private room"]
nb_private_rooms = int(df_rooms.shape[0])



df["price"] = df["price"].str.replace("dollars", "")
df["price"] = df["price"].astype(float)
avg_price = round(df["price"].mean(), 2)\

print(df.head(20))
print("---------------------------------------------------------------------------------------------------------------------------------------")
print(f"Earliest Review Date: {first_reviewed}")
print(f"Most Recent Review Date: {last_reviewed}")
print(f"Number of private rooms: {nb_private_rooms}")
print(f"Average listing price: {avg_price}")


review_dates = pd.DataFrame({
    "first_reviewed": [first_reviewed],
    "last_reviewed": [last_reviewed],
    "nb_private_rooms": [nb_private_rooms],
    "avg_price": [avg_price]
})

print(review_dates)

# 1. Average Price by Room Type
plt.figure(figsize=(10,6))
avg_price_by_room_type = df.groupby('room_type')['price'].mean().sort_values(ascending=False)
avg_price_by_room_type.plot(kind='bar', color='skyblue')
plt.title('Average Price by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Average Price (in dollars)')
plt.xticks(rotation=45)
plt.show()

# 2. Count of Listings by Room Type
plt.figure(figsize=(10,6))
listing_count_by_room_type = df['room_type'].value_counts()
listing_count_by_room_type.plot(kind='bar', color='lightgreen')
plt.title('Count of Listings by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Count of Listings')
plt.xticks(rotation=45)
plt.show()

# 3. Number of Listings by Neighborhood
plt.figure(figsize=(12,8))
listing_count_by_nbhood = df['nbhood_full'].value_counts().head(10)  # Show top 10 neighborhoods
listing_count_by_nbhood.plot(kind='bar', color='orange')
plt.title('Top 10 Neighborhoods by Number of Listings')
plt.xlabel('Neighborhood')
plt.ylabel('Count of Listings')
plt.xticks(rotation=45)
plt.show()

# 4. Review Count by Room Type (if 'last_review' can be used as a proxy for reviews)
plt.figure(figsize=(10,6))
review_count_by_room_type = df.groupby('room_type')['last_review'].count().sort_values(ascending=False)
review_count_by_room_type.plot(kind='bar', color='lightcoral')
plt.title('Number of Reviews by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.show()












