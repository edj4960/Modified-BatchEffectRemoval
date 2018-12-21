import h5py
import numpy as np
import matplotlib
import math
from numpy import genfromtxt
import csv
import pandas as pd

myfilePath = "/home/ejones36/Desktop/BatchEffectRemoval-master/Data/tidy_batches_balanced.csv"

df = pd.read_csv(myfilePath)

source_output = df.loc[df['Batch'] == 'A']
target_output = df.loc[df['Batch'] == 'B']

source_output.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)
target_output.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)

source_output.to_csv('source_converted.csv', index=False, header=False)
target_output.to_csv('target_converted.csv', index=False, header=False)

# df.to_csv('test.csv', index=False, header=False)