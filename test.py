import pandas as pd 

labels = pd.read_csv("D:/MSCS/ADL/Data/train.csv")
print(len(labels))
print(labels.iloc[[0,1], 0])
print(labels.iloc[[0,1], 1])