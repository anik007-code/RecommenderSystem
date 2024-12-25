import pandas as pd
from configs.config import path

def read_data(path):
    data = pd.read_csv(path)
    return data.head(5)

print(read_data(path))


