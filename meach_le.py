import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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