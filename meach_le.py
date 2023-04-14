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

df['first'] = (df['first'] == "g").astype(int)