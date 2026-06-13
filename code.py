#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#loading the data
df = pd.read_csv("USA Housing Dataset.csv")
#making the price column more readable
df["price"] = df["price"].astype(int)
#we don't need date column so we remove it
df=df.drop(columns=["date"])
#here , because the data in the df was messy,we transform it into a more readable format to display all columns
from IPython.display import display
pd.set_option("display.max_columns", None)
display(df.head())
# we check for NaN values , there were none
df.isna().sum()
#since the majority of years in this column is not listed  we use a better approach wich is to instead of years we only know if the house was renovated on no based on 0( not renovated) and 1(renovated)
df["renovated"] = df["yr_renovated"].apply(lambda x : 1 if x > 1 else 0)
# we drop the old year of renovation columns since it has been replaced by the new approach  and we drop the columns wich we don't need wich are street name and the country because we know all the houses are in the USA
df = df.drop(columns = ["yr_renovated","street","statezip","country"])
# we encode the strings wich are the city to numerical values since  this column is so important
df["city"] = LabelEncoder().fit_transform(df["city"])
display(df)
# we calculate the correlation and the relationship between the data and we plot them into a heatmap and plot it
corr = df.corr()
plt.figure(figsize = (12 , 10))
sns.heatmap(corr , annot = True )
plt.plot(corr)
plt.show()
