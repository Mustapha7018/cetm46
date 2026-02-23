import pandas as pd

with open("dataset.csv", "r") as f:
    data = pd.read_csv(f)

    # total number of rows and columns
    print(data.shape)
    print(data.head())
    print(data.Country.nunique())