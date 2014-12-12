import os
import sys
import time
import math
import bisect

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa import stattools

from Base import Base
import lomb


class Amplitude(Base):
    """Half the difference between the maximum and the minimum magnitude"""

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        N = len(data)
        sorted = np.sort(data)
        return (np.median(sorted[-0.05 * N:]) -
                np.median(sorted[0:0.05 * N])) / 2


class Rcs(Base):
    #Range of cumulative sum
    def __init__(self):
        self.category = 'timeSeries'

    def fit(self, data):
        sigma = np.std(data)
        N = len(data)
        m = np.mean(data)
        s = np.cumsum(data - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R


class StetsonK(Base):
    def __init__(self):
        self.category = 'timeSeries'

    def fit(self, data):
        N = len(data)
        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (data - np.mean(data)) / np.std(data))

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        return K


class Automean(Base):
    """This is just a prototype, not a real feature"""

    def __init__(self, length):
        self.category = 'basic'
        if len(length) != 2:
            print "need 2 parameters for feature automean"
            sys.exit(1)
        self.length = length[0]
        self.length2 = length[1]

    def fit(self, data):
        return np.mean(data) + self.length + self.length2


class Meanvariance(Base):
    """variability index"""
    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        return np.std(data) / np.mean(data)


class Autocor_length(Base):

    def __init__(self):
        self.category = 'timeSeries'

    def fit(self, data):

        AC = stattools.acf(data, nlags=100)
        k = next((index for index, value in
                 enumerate(AC) if value < np.exp(-1)), None)

        return k


class SlottedA_length(Base):

    def __init__(self, entry):
        """
        lc: MACHO lightcurve in a pandas DataFrame
        k: lag (default: 1)
        T: tau (slot size in days. default: 4)
        """
        self.category = 'timeSeries'
        SlottedA_length.SAC = []

        self.mjd = entry[0]
        self.T = entry[1]

    def slotted_autocorrelation(self, data, mjd, T, K,
                                second_round=False, K1=100):

        slots = np.zeros((K, 1))
        i = 1

        # make time start from 0
        mjd = mjd - np.min(mjd)

        # subtract mean from mag values
        m = np.mean(data)
        data = data - m

        prod = np.zeros((K, 1))
        pairs = np.subtract.outer(mjd, mjd)
        pairs[np.tril_indices_from(pairs)] = 10000000

        ks = np.int64(np.floor(np.abs(pairs) / T + 0.5))

        #We calculate the slotted autocorrelation for k=0 separately
        idx = np.where(ks == 0)
        prod[0] = ((sum(data ** 2) + sum(data[idx[0]] *
                   data[idx[1]])) / (len(idx[0]) + len(data)))
        slots[0] = 0

        #We calculate it for the rest of the ks
        if second_round is False:
            for k in np.arange(1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
        else:
            for k in np.arange(K1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i - 1] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
            np.trim_zeros(prod, trim='b')

        slots = np.trim_zeros(slots, trim='b')
        return prod / prod[0], np.int64(slots).flatten()

    def fit(self, data):

        # T=4
        K1 = 100
        [SAC, slots] = self.slotted_autocorrelation(data, self.mjd, self.T, K1)
        SlottedA_length.SAC = SAC
        SlottedA_length.slots = slots

        SAC2 = SAC[slots]
        k = next((index for index, value in
                 enumerate(SAC2) if value < np.exp(-1)), None)

        if k is None:
            K2 = 200
            [SAC, slots] = self.slotted_autocorrelation(data, self.mjd, self.T,
                                                        K2, second_round=True,
                                                        K1=K1)
            SAC2 = SAC[slots]
            k = next((index for index, value in
                     enumerate(SAC2) if value < np.exp(-1)), None)

        return slots[k] * self.T

    def getAtt(self):
        return SlottedA_length.SAC, SlottedA_length.slots


class StetsonK_AC(SlottedA_length):

    def __init__(self):

        self.category = 'timeSeries'

    def fit(self, data):

        a = StetsonK_AC()
        [autocor_vector, slots] = a.getAtt()

        autocor_vector = autocor_vector[slots]
        N_autocor = len(autocor_vector)
        sigmap = (np.sqrt(N_autocor * 1.0 / (N_autocor - 1)) *
                 (autocor_vector - np.mean(autocor_vector)) /
                  np.std(autocor_vector))

        K = (1 / np.sqrt(N_autocor * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        return K


class StetsonL(Base):
    def __init__(self, entry):
        self.category = 'timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute StetsonL"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]

    def fit(self, data):

        N = len(self.aligned_data)

            #sys.exit(1)

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                 (self.aligned_data[:N] - np.mean(self.aligned_data)) /
                  np.std(self.aligned_data))

        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                 (self.data2[:N] - np.mean(self.data2)) /
                  np.std(self.data2))
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) *
             np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i))))

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i ** 2)))

        return J * K / 0.798


class Con(Base):
    """Index introduced for selection of variable starts from OGLE database.


    To calculate Con, we counted the number of three consecutive measurements
    that are out of 2sigma range, and normalized by N-2
    Pavlos not happy
    """
    def __init__(self, consecutiveStar=3):
        self.category = 'timeSeries'
        self.consecutiveStar = consecutiveStar

    def fit(self, data):

        N = len(data)
        if N < self.consecutiveStar:
            return 0
        sigma = np.std(data)
        m = np.mean(data)
        count = 0

        for i in xrange(N - self.consecutiveStar + 1):
            flag = 0
            for j in xrange(self.consecutiveStar):
                if(data[i + j] > m + 2 * sigma or data[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        return count * 1.0 / (N - self.consecutiveStar + 1)


# class VariabilityIndex(Base):

#     # Eta. Removed, it is not invariant to time sampling
#     '''
#     The index is the ratio of mean of the square of successive difference to
#     the variance of data points
#     '''
#     def __init__(self):
#         self.category='timeSeries'


#     def fit(self, data):

#         N = len(data)
#         sigma2 = np.var(data)

#         return 1.0/((N-1)*sigma2) * np.sum(np.power(data[1:] - data[:-1] , 2)
    #)


class Color(Base):
    """Average color for each MACHO lightcurve
    mean(B1) - mean(B2)
    """
    def __init__(self, second_data):
        self.category = 'timeSeries'
        if second_data is None:
            print "please provide another data series to compute Color"
            sys.exit(1)
        self.data2 = second_data

    def fit(self, data):
        return np.mean(data) - np.mean(self.data2)


# The categories of the following featurs should be revised

class Beyond1Std(Base):
    """Percentage of points beyond one st. dev. from the weighted
    (by photometric errors) mean
    """

    def __init__(self, error):
        self.category = 'basic'
        if error is None:
            print "please provide the measurement erros to compute Beyond1Std"
            sys.exit(1)
        self.error = error

    def fit(self, data):
        n = len(data)

        weighted_mean = np.average(data, weights=1 / self.error ** 2)

        # Standard deviation with respect to the weighted mean

        var = sum((data - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(np.logical_or(data > weighted_mean + std,
                                     data < weighted_mean - std))

        return float(count) / n


class SmallKurtosis(Base):
    """Small sample kurtosis of the magnitudes.

    See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    """

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)

        S = sum(((data - mean) / std) ** 4)

        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return c1 * S - c2


class Std(Base):
    """Standard deviation of the magnitudes"""

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        return np.std(data)


class Skew(Base):
    """Skewness of the magnitudes"""

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        return stats.skew(data)


class StetsonJ(Base):
    """Stetson (1996) variability index, a robust standard deviation"""

    def __init__(self, entry):
        self.category = 'timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute StetsonJ"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]

    def fit(self, data):

        N = len(self.data2)

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                 (self.aligned_data[:N] - np.mean(self.aligned_data)) /
                  np.std(self.aligned_data))
        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                 (self.data2[:N] - np.mean(self.data2)) /
                  np.std(self.data2))
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) * np.sum(np.sign(sigma_i) *
             np.sqrt(np.abs(sigma_i))))

        return J


class MaxSlope(Base):
    """
    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)
    """

    def __init__(self, mjd):
        self.category = 'timeSeries'
        if mjd is None:
            print "please provide the measurement times to compute MaxSlope"
            sys.exit(1)
        self.mjd = mjd

    def fit(self, data):

        slope = np.abs(data[1:] - data[:-1]) / (self.mjd[1:] - self.mjd[:-1])
        np.max(slope)

        return np.max(slope)


class MedianAbsDev(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        median = np.median(data)

        devs = (abs(data - median))

        return np.median(devs)


class MedianBRP(Base):
    """Median buffer range percentage

    Fraction (<= 1) of photometric points within amplitude/10
    of the median magnitude
    """

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        median = np.median(data)
        amplitude = (np.max(data) - np.min(data)) / 10
        n = len(data)

        count = np.sum(np.logical_and(data < median + amplitude,
                                      data > median - amplitude))

        return float(count) / n


class PairSlopeTrend(Base):
    """
    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    """

    def __init__(self):
        self.category = 'timeSeries'

    def fit(self, data):
        data_last = data[-30:]

        return (float(len(np.where(np.diff(data_last) > 0)[0]) -
                len(np.where(np.diff(data_last) <= 0)[0])) / 30)


class FluxPercentileRatioMid20(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        sorted_data = np.sort(data)
        lc_length = len(sorted_data)

        F_60_index = int(0.60 * lc_length)
        F_40_index = int(0.40 * lc_length)
        F_5_index = int(0.05 * lc_length)
        F_95_index = int(0.95 * lc_length)

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        return F_mid20


class FluxPercentileRatioMid35(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        sorted_data = np.sort(data)
        lc_length = len(sorted_data)

        F_325_index = int(0.325 * lc_length)
        F_675_index = int(0.675 * lc_length)
        F_5_index = int(0.05 * lc_length)
        F_95_index = int(0.95 * lc_length)

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        return F_mid35


class FluxPercentileRatioMid50(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        sorted_data = np.sort(data)
        lc_length = len(sorted_data)

        F_25_index = int(0.25 * lc_length)
        F_75_index = int(0.75 * lc_length)
        F_5_index = int(0.05 * lc_length)
        F_95_index = int(0.95 * lc_length)

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        return F_mid50


class FluxPercentileRatioMid65(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        sorted_data = np.sort(data)
        lc_length = len(sorted_data)

        F_175_index = int(0.175 * lc_length)
        F_825_index = int(0.825 * lc_length)
        F_5_index = int(0.05 * lc_length)
        F_95_index = int(0.95 * lc_length)

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        return F_mid65


class FluxPercentileRatioMid80(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):
        sorted_data = np.sort(data)
        lc_length = len(sorted_data)

        F_10_index = int(0.10 * lc_length)
        F_90_index = int(0.90 * lc_length)
        F_5_index = int(0.05 * lc_length)
        F_95_index = int(0.95 * lc_length)

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        return F_mid80


class PercentDifferenceFluxPercentile(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):

        median_data = np.median(data)

        sorted_data = np.sort(data)
        lc_length = len(sorted_data)
        F_5_index = int(0.05 * lc_length)
        F_95_index = int(0.95 * lc_length)
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

        percent_difference = F_5_95 / median_data

        return percent_difference


class PercentAmplitude(Base):

    def __init__(self):
        self.category = 'basic'

    def fit(self, data):

        median_data = np.median(data)
        distance_median = np.abs(data - median_data)
        max_distance = np.max(distance_median)

        percent_amplitude = max_distance / median_data

        return percent_amplitude


class LinearTrend(Base):

    def __init__(self, mjd):
        self.category = 'timeSeries'

        if mjd is None:
            print "please provide the measurement times to compute LinearTrend"
            sys.exit(1)
        self.mjd = mjd

    def fit(self, data):

        regression_slope = stats.linregress(self.mjd, data)[0]

        return regression_slope


class Eta_color(Base):

    def __init__(self, entry):

        self.category = 'timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute Eta_B_R"
        #     sys.exit(1)
        self.data2 = np.asarray(entry[0])
        self.aligned_data = np.asarray(entry[1])
        self.mjd = np.asarray(entry[2])

    def fit(self, data):

        N = len(self.aligned_data)
        B_Rdata = self.aligned_data - self.data2
        # # N = len(B_Rdata)
        # sigma2 = np.var(B_Rdata)

        # return 1.0/((N-1)*sigma2) * np.sum(np.power(B_Rdata[1:] -
            #B_Rdata[:-1] , 2))

        w = 1.0 / np.power(self.mjd[1:] - self.mjd[:-1], 2)
        w_mean = np.mean(w)

        N = len(self.mjd)
        sigma2 = np.var(B_Rdata)

        S1 = sum(w * (B_Rdata[1:] - B_Rdata[:-1]) ** 2)
        S2 = sum(w)

        eta_B_R = (w_mean * np.power(self.mjd[N - 1] -
                   self.mjd[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        return eta_B_R


class Eta_e(Base):

    def __init__(self, mjd):

        self.category = 'timeSeries'

        if mjd is None:
            print "please provide the measurement times to compute Eta_e"
            sys.exit(1)
        self.mjd = mjd

    def fit(self, data):

        w = 1.0 / np.power(self.mjd[1:] - self.mjd[:-1], 2)
        w_mean = np.mean(w)

        N = len(self.mjd)
        sigma2 = np.var(data)

        S1 = sum(w * (data[1:] - data[:-1]) ** 2)
        S2 = sum(w)

        eta_e = (w_mean * np.power(self.mjd[N - 1] -
                 self.mjd[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        return eta_e


class Mean(Base):

    def __init__(self):

        self.category = 'basic'

    def fit(self, data):

        B_mean = np.mean(data)

        return B_mean


class Q31(Base):

    def __init__(self):

        self.category = 'basic'

    def fit(self, data):

        return np.percentile(data, 75) - np.percentile(data, 25)


class Q31_color(Base):

    def __init__(self, entry):

        self.category = 'timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute Q31B_R"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]

    def fit(self, data):

        N = len(self.data2)
        b_r = self.aligned_data[:N] - self.data2[:N]

        return np.percentile(b_r, 75) - np.percentile(b_r, 25)


class AndersonDarling(Base):

    def __init__(self):

        self.category = 'timeSeries'

    def fit(self, data):

        ander = stats.anderson(data)[0]
        #return ander
        return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))


class PeriodLS(Base):

    def __init__(self, mjd):

        self.category = 'timeSeries'

        if mjd is None:
            print "please provide the measurement times to compute PeriodLS"
            sys.exit(1)
        self.mjd = mjd

    def fit(self, data):

        global new_mjd
        global prob

        fx, fy, nout, jmax, prob = lomb.fasper(self.mjd, data, 6., 100.)
        T = 1.0 / fx[jmax]
        new_mjd = np.mod(self.mjd, 2 * T) / (2 * T)

        return T


class Period_fit(Base):

    def __init__(self):

        self.category = 'timeSeries'

    def fit(self, data):

        # a = Period_fit()
        # return a.getPeriod_fit()

        return prob


class Psi_CS(Base):

    def __init__(self, mjd):

        self.category = 'timeSeries'
        self.mjd = mjd

    def fit(self, data):

        folded_data = data[np.argsort(new_mjd)]

        sigma = np.std(folded_data)
        N = len(folded_data)
        m = np.mean(folded_data)
        s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)

        return R


class Psi_eta(Base):

    def __init__(self):

        self.category = 'timeSeries'

    def fit(self, data):

        # folded_mjd = np.sort(new_mjd)
        folded_data = data[np.argsort(new_mjd)]

        # w = 1.0 / np.power(folded_mjd[1:]-folded_mjd[:-1] ,2)
        # w_mean = np.mean(w)

        # N = len(folded_mjd)
        # sigma2=np.var(folded_data)

        # S1 = sum(w*(folded_data[1:]-folded_data[:-1])**2)
        # S2 = sum(w)

        # Psi_eta = w_mean * np.power(folded_mjd[N-1]-folded_mjd[0],2) * S1 /
        # (sigma2 * S2 * N**2)

        N = len(folded_data)
        sigma2 = np.var(folded_data)

        Psi_eta = (1.0 / ((N - 1) * sigma2) *
                   np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))

        return Psi_eta


class CAR_sigma(Base):

    def __init__(self, entry):

        self.category = 'timeSeries'
        self.N = len(entry[0])
        self.error = entry[1].reshape((self.N, 1)) ** 2
        self.mjd = entry[0].reshape((self.N, 1))

    def CAR_Lik(self, parameters, t, x, error_vars):

        sigma = parameters[0]
        tau = parameters[1]
       #b = parameters[1] #comment it to do 2 pars estimation
       #tau = params(1,1);
       #sigma = sqrt(2*var(x)/tau);

        b = np.mean(x) / tau
        epsilon = 1e-300
        cte_neg = -np.infty
        num_datos = np.size(x)

        Omega = []
        x_hat = []
        a = []
        x_ast = []

        # Omega = np.zeros((num_datos,1))
        # x_hat = np.zeros((num_datos,1))
        # a = np.zeros((num_datos,1))
        # x_ast = np.zeros((num_datos,1))

        # Omega[0]=(tau*(sigma**2))/2.
        # x_hat[0]=0.
        # a[0]=0.
        # x_ast[0]=x[0] - b*tau

        Omega.append((tau * (sigma ** 2)) / 2.)
        x_hat.append(0.)
        a.append(0.)
        x_ast.append(x[0] - b * tau)

        loglik = 0.

        for i in range(1, num_datos):

            a_new = np.exp(-(t[i] - t[i - 1]) / tau)
            x_ast.append(x[i] - b * tau)
            x_hat.append(
                a_new * x_hat[i - 1] +
                (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
                (x_ast[i - 1] - x_hat[i - 1]))

            Omega.append(
                Omega[0] * (1 - (a_new ** 2)) + ((a_new ** 2)) * Omega[i - 1] *
                (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1]))))

            # x_ast[i]=x[i] - b*tau
            # x_hat[i]=a_new*x_hat[i-1] + (a_new*Omega[i-1]/(Omega[i-1] +
                #error_vars[i-1]))*(x_ast[i-1]-x_hat[i-1])
            # Omega[i]=Omega[0]*(1-(a_new**2)) + ((a_new**2))*Omega[i-1]*
            #( 1 - (Omega[i-1]/(Omega[i-1]+ error_vars[i-1])))

            loglik_inter = np.log(
                ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
                (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
                 (Omega[i] + error_vars[i]))) + epsilon))

            loglik = loglik + loglik_inter

            if(loglik <= cte_neg):
                print('CAR lik se fue a inf')
                return None

        # the minus one is to perfor maximization using the minimize function
        return -loglik

    def calculateCAR(self, mjd, data, error):
        x0 = [10, 0.5]
        bnds = ((0, 100), (0, 100))
        # res = minimize(self.CAR_Lik, x0, args=(LC[:,0],LC[:,1],LC[:,2]) ,
            #method='nelder-mead',bounds = bnds)

        res = minimize(self.CAR_Lik, x0, args=(mjd, data, error),
                       method='nelder-mead', bounds=bnds)
        # options={'disp': True}
        sigma = res.x[0]
        CAR_sigma.tau = res.x[1]
        return sigma

    def getAtt(self):
        return CAR_sigma.tau

    def fit(self, data):
        # LC = np.hstack((self.mjd , data.reshape((self.N,1)), self.error))
        a = self.calculateCAR(self.mjd, data.reshape((self.N, 1)), self.error)

        return a


class CAR_tau(CAR_sigma):

    def __init__(self):

        self.category = 'timeSeries'

    def fit(self, data):

        a = CAR_tau()

        return a.getAtt()


class CAR_tmean(CAR_sigma):

    def __init__(self):

        self.category = 'timeSeries'

    def fit(self, data):

        a = CAR_tmean()
        return np.mean(data) / a.getAtt()
