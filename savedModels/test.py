

import argparse
import h5py
import os.path
import numpy as np
import matplotlib
import math
import pandas as pd
from pandas import HDFStore
import pdb
import os


os.system('h5dump -o dset.csv -y -w 400 abc.h5')

# filename = 'abc.h5'
# f = h5py.File(filename, 'r')
# np.savetxt('datafile.ascii')
# f.close()

# hdfPath = 'abc.h5'
# hdfKey = 'dfkey'

# store = pd.HDFStore(hdfPath)

# a = store.keys()

# print(len(a))

#store.select(key="/data/table1") # the next table would be /data/table2

# with h5py.File(hdfPath,'r') as f:
#    hdfKey = f.keys()

# with pd.HDFStore(hdfPath) as hdf:
#     index = hdf.select_column(hdfKey, 'index').values
# store = HDFStore('abc.h5')
# store[index].to_csv('outputFileForTable1.csv')

# print('PRINTING KEYS:\n')

# with h5py.File("abc.h5",'r') as f:
#      print(f.keys())

'''
with pd.HDFStore('abc.h5', 'r') as d:
    df = d.get(f.keys())
    df.to_csv('myfile.csv')
'''
# np.savetxt('out.csv', h5py.File('abc.h5'), '%g', ',')

#h5dump -o 'dset.asci' -y -w 400 abc.h5

#file = h5py.File('abc.h5','r+')

#dataset1 = file['/dense_1/dense_1/kernel'] # suppose kernel contains the weights
#numpy.savetxt("dataset1.csv",dataset1)
