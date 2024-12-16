import pandas as pd


data = pd.read_csv("BBox_List_2017.csv")
print(data["Finding Label"].unique())