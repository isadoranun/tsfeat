import os,sys,time
import numpy as np
import pandas as pd
from scipy import stats

path ='/Users/isadoranun/Desktop/Features/F_1/'

for j in os.listdir(path):

	if j.endswith("_2.csv"):

		lc = j[4:8]

		if os.path.isfile(path + 'F_1_' + lc + '.csv'):

			data1 = pd.read_csv(path + 'F_1_' + lc + '.csv',index_col=0)
			data2 = pd.read_csv(path + j,index_col=0)

			data1.Rcs = data2.Rcs

			features = data2.columns[1:-1]
			data2 = data2[features]

			concatenate = pd.merge(data1,data2, left_index=True, right_index=True, how='outer')

			concatenate.to_csv('F_1_' + lc + '_complete.csv') 
