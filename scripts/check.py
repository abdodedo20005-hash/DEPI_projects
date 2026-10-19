import pandas as pd

df = pd.read_csv("data/amazon.csv")
print(df.columns.tolist())
print(len(df))
print(df.head(3))