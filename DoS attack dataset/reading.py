import pandas as pd
df = pd.read_csv("mixed2.csv", nrows = 2060000)

print("Dataframe shape:", df.shape)

df.to_csv('mixed2million.csv')
