#!/usr/bin/env python

from Feature import FeatureSpace
import numpy as np
from import_lc_cluster import LeerLC_MACHO
from PreprocessLC import Preprocess_LC
from alignLC import Align_LC
import os.path
import tarfile
import sys
import pandas as pd
import pytest

@pytest.fixture
def white_noise():
	data = np.random.normal(size=10000)
	mjd=np.arange(10000)
	error = np.random.normal(loc=0.01, scale =0.8, size=10000)
	second_data = np.random.normal(size=10000)
	mjd2=np.arange(10000)
	error2 = np.random.normal(loc=0.01, scale =0.01, size=10000)
	aligned_data = data
	aligned_second_data = second_data
	aligned_mjd = mjd
	return data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd

@pytest.fixture
def periodic_lc():
	N=100
	mjd_periodic = np.arange(N)
	Period = 20
	cov = np.zeros([N,N])
	mean = np.zeros(N)
	for i in np.arange(N):
	    for j in np.arange(N):
	        cov[i,j] = np.exp( -(np.sin( (np.pi/Period) *(i-j))**2))
	data_periodic=np.random.multivariate_normal(mean, cov)
	return data_periodic, mjd_periodic


@pytest.fixture
def uniform_lc():
	mjd_uniform=np.arange(1000000)
	data_uniform=np.random.uniform(size=1000000)
	return data_uniform, mjd_uniform

# def test_Amplitude(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['Amplitude'])
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

# def test_Autocor(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['Autocor'] )
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

# def test_Automean(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['Automean'] , Automean=[0,0])
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

# def test_B_R(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['B_R'] , B_R=second_data)
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

def test_Beyond1Std(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Beyond1Std'] , Beyond1Std= white_noise[2])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= 0.30 and a.result(method='array') <= 0.34)

def test_Bmean(white_noise):

	a = FeatureSpace(featureList=['Bmean'])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= -0.1 and a.result(method='array') <= 0.1)

# def test_CAR(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['CAR_sigma', 'CAR_tau', 'CAR_tmean'] , CAR_sigma=[mjd, error])
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)


def test_Con(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Con'] , Con=1)
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= 0.04 and a.result(method='array') <= 0.05)

def test_Eta_B_R(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Eta_B_R'] , Eta_B_R=[white_noise[5], white_noise[4], white_noise[6]])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= 1.9 and a.result(method='array') <= 2.1)

def test_Eta_e(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Eta_e'], Eta_e = white_noise[1] )
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= 1.9 and a.result(method='array') <= 2.1)

def test_FluxPercentile(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['FluxPercentileRatioMid20','FluxPercentileRatioMid35','FluxPercentileRatioMid50','FluxPercentileRatioMid65','FluxPercentileRatioMid80'] )
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array')[0] >= 0.145 and a.result(method='array')[0] <= 0.160)
	assert(a.result(method='array')[1] >= 0.260 and a.result(method='array')[1] <= 0.290)
	assert(a.result(method='array')[2] >= 0.390 and a.result(method='array')[2] <= 0.420)
	assert(a.result(method='array')[3] >= 0.550 and a.result(method='array')[3] <= 0.580)
	assert(a.result(method='array')[4] >= 0.760 and a.result(method='array')[4] <= 0.800)



def test_LinearTrend(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['LinearTrend'] , LinearTrend = white_noise[1])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= -0.1 and a.result(method='array') <= 0.1)

# def test_MaxSlope(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['MaxSlope'] , MaxSlope=mjd)
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

def test_Meanvariance(uniform_lc):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Meanvariance'])
	a=a.calculateFeature(uniform_lc[0])

	assert(a.result(method='array') >= 0.576 and a.result(method='array') <= 0.578)

def test_MedianAbsDev(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['MedianAbsDev'])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= 0.670 and a.result(method='array') <= 0.680)

# def test_MedianBRP(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['MedianBRP'] , MaxSlope=mjd)
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

def test_PairSlopeTrend(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['PairSlopeTrend'])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= -0.25 and a.result(method='array') <= 0.25)

# def test_PercentAmplitude(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['PercentAmplitude'])
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

# def test_PercentDifferenceFluxPercentile(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['PercentDifferenceFluxPercentile'])
# 	a=a.calculateFeature(white_noise[0])

# 	assert(a.result(method='array') >= 0.043 and a.result(method='array') <= 0.046)

def test_Period_Psi(periodic_lc):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['PeriodLS', 'Period_fit','Psi_CS','Psi_eta'], PeriodLS = periodic_lc[1], Psi_CS= periodic_lc[1])
	a=a.calculateFeature(periodic_lc[0])
	# print a.result(method='array'), len(periodic_lc[0])
	assert(a.result(method='array')[0] >= 19 and a.result(method='array')[0] <= 21)

# def test_Q31(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['Q31'])
# 	a=a.calculateFeature(white_noise[0])

# def test_Q31B_R(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['Q31B_R'], Q31B_R = [aligned_second_data, aligned_data])
# 	a=a.calculateFeature(white_noise[0])

def test_Rcs(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Rcs'])
	a=a.calculateFeature(white_noise[0])
	assert(a.result(method='array') >= 0 and a.result(method='array') <= 0.1)

# def test_Skew(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['Skew'])
# 	a=a.calculateFeature(white_noise[0])

# def test_SlottedA(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['SlottedA'], SlottedA = [mjd, 1])
# 	a=a.calculateFeature(white_noise[0])

# def test_SmallKurtosis(white_noise):
# 	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

# 	a = FeatureSpace(featureList=['SmallKurtosis'])
# 	a=a.calculateFeature(white_noise[0])

def test_Std(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['Std'])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array') >= 0.9 and a.result(method='array') <= 1.1)


def test_Stetson(white_noise):
	# data, mjd, error, second_data, aligned_data, aligned_second_data, aligned_mjd = white_noise()

	a = FeatureSpace(featureList=['SlottedA','StetsonK', 'StetsonK_AC', 'StetsonJ', 'StetsonL'], SlottedA = [white_noise[1], 4] ,StetsonJ = [white_noise[5] , white_noise[4]], StetsonL = [white_noise[5] , white_noise[4]])
	a=a.calculateFeature(white_noise[0])

	assert(a.result(method='array')[1] >= 0.793 and a.result(method='array')[1] <= 0.81)
	assert(a.result(method='array')[2] >= 0.25 and a.result(method='array')[2] <= 0.44)
	assert(a.result(method='array')[3] >= -0.1 and a.result(method='array')[3] <= 0.1)
	assert(a.result(method='array')[4] >= -0.1 and a.result(method='array')[4] <= 0.1)





