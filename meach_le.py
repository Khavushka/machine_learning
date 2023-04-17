import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

'''
Dataset
'''

cols = ["first", "second", "third"]
df = pd.read_csv('spiral.txt', names=cols,  sep='\t')
df.head()

print(df.head())

# til at convertere, hvis man har g og h til at være 1 0g 0
# df['first'] = (df['first'] == "g").astype(int) 


for label in cols[:-1]:
    plt.hist(df[df["third"]==1][label], color="blue", label="gamma", alpha = 0.7, density=True)
    plt.hist(df[df["third"]==0][label], color="red", label="gamma", alpha = 0.7, density=True)
    plt.title(label)
    plt.ylabel("Prob")
    plt.xlabel(label)
    plt.legend()
    plt.show()
    
# Train, validation, test datasets
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe):
    x = dataframe[dataframe.cols[:-1]].values
    y = dataframe[dataframe.cols[-1]].values
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    
    return data, x, y