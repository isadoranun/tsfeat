

from Feature import FeatureSpace
import numpy as np

#"test"

data = np.random.uniform(-5,-3, 1000)
second_data = np.random.uniform(-5,-3, 1000)
error= np.random.uniform(0.000001,1, 1000)
mjd= np.random.uniform(40000,50000, 1000)
# minper=1.
# maxper=100.
# subsample=1 
# Npeaks=1 
# clip=5.0 
# clipiter=1 
# whiten=0



# a = FeatureSpace(category='all',featureList=None, automean=[0,0], StetsonL=second_data ,  B_R=second_data, Beyond1Std=error, StetsonJ=second_data, MaxSlope=mjd, LinearTrend=mjd, Eta_B_R=second_data, Eta_e=mjd, Q31B_R=second_data, PeriodLS=mjd)

# PeriodLS=[mjd,error,minper, maxper, subsample, Npeaks, clip, clipiter, whiten]
a = FeatureSpace(category='basic', automean=[0,0])
#print a.featureList
a=a.calculateFeature(data)
#print a.result(method='')


np.savetxt('test.txt',a.result(method='array'))

print a.result(method='dict')
