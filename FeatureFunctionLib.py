import os,sys,time
import numpy as np
import pandas as pd
from scipy import stats

from Base import Base

from scipy.optimize import minimize

import lomb
import math


class Rcs(Base):
    #Range of cumulative sum
    def __init__(self):
        self.category='timeSeries'

    def fit(self, data):
        sigma = np.std(data)
        N = len(data)
        m = np.mean(data)
        s = (np.cumsum(data)-m)*1.0/(N*sigma)
        R = np.max(s) - np.min(s)
        return R
   
class StetsonK(Base):
    def __init__(self):
        self.category='timeSeries'
    def fit(self, data):
        N = len(data)
        sigmap = np.sqrt(N*1.0/(N-1)) * (data-np.mean(data))/np.std(data)
    
        K = 1/np.sqrt(N*1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap**2))

        return K


class automean(Base):
    '''
    This is just a prototype, not a real feature
    '''
    def __init__(self, length): 
        self.category='basic'
        if len(length)!=2:
            print "need 2 parameters for feature automean"
            sys.exit(1)
        self.length = length[0]
        self.length2 = length[1]
    def fit(self, data):
        return np.mean(data)+self.length+self.length2

class meanvariance(Base):
    # variability index
    def __init__(self): 
        self.category='basic'
      
    def fit(self, data):
        return np.std(data)/np.mean(data)



    
class autocor(Base):
    def __init__(self):
        self.category='timeSeries'

    def autocorrelation(self, data, lag):
        N=len(data)
        std= np.std(data)
        m = np.mean(data)
        suma = 0

        for i in xrange(N-lag):
            suma += (data[i]- m)*(data[i+lag] - m)

        ac = 1/((N-lag)* std**2) * suma 

        return ac
        
    def fit(self, data):
        threshold = math.exp(-1)
        norm_value = self.autocorrelation(data, lag = 0 )
        lag = 1
        current_autocorr_value = 1

        while current_autocorr_value > threshold:
            current_autocorr_value = self.autocorrelation(data, lag=lag)/norm_value
            lag = lag + 1
        return lag


class StetsonK_AC(Base):
    def __init__(self):

        self.category='timeSeries'

    def autocorrelation(self, data, lag):
        N=len(data)
        std= np.std(data)
        m = np.mean(data)
        suma = 0

        for i in xrange(N-lag):
            suma += (data[i]- m)*(data[i+lag] - m)

        ac = 1/((N-lag)* std**2) * suma 

        return ac

    def fit(self, data):
        autocor_vector=[]

        for i in xrange(len(data)/2):
            autocor_vector.append(self.autocorrelation(data, i))

        N_autocor = len(autocor_vector)
        sigmap = np.sqrt(N_autocor*1.0/(N_autocor-1)) * (data-np.mean(autocor_vector))/np.std(autocor_vector)
    
        K = 1/np.sqrt(N_autocor*1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap**2))

        return K



class StetsonL(Base):
    def __init__(self, entry):
        self.category='timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute StetsonL"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]

    def fit(self, data):
        
        N = len(self.aligned_data)            
        
            #sys.exit(1)

        sigmap = np.sqrt(N*1.0/(N-1)) * (self.aligned_data[:N]-np.mean(self.aligned_data))/np.std(self.aligned_data)
        sigmaq = np.sqrt(N*1.0/(N-1)) * (self.data2[:N]-np.mean(self.data2))/np.std(self.data2)
        sigma_i = sigmap * sigmaq
        J= 1.0/len(sigma_i) * np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i)))
        K = 1/np.sqrt(N*1.0) * np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i**2))

       
        return J*K/0.798        

class Con(Base):
    '''
    Index introduced for selection of variable starts from OGLE database. 
    To calculate Con, we counted the number of three consecutive starts that are out of 2sigma range, and normalized by N-2
    '''
    def __init__(self, consecutiveStar=3):
        self.category='timeSeries'
        self.consecutiveStar = consecutiveStar

    def fit(self, data):

        N = len(data)
        if N < self.consecutiveStar:
            return 0
        sigma = np.std(data)
        m = np.mean(data)
        count=0
        
        for i in xrange(N-self.consecutiveStar+1):
            flag = 0
            for j in xrange(self.consecutiveStar):
                if (data[i+j] > m+2*sigma or data[i+j] < m-2*sigma) :
                    flag = 1
                else:
                    flag=0
                    break
            if flag:
                count = count+1
        return count*1.0/(N-self.consecutiveStar+1)


class VariabilityIndex(Base):

    # Eta
    '''
    The index is the ratio of mean of the square of successive difference to the variance of data points
    '''
    def __init__(self):
        self.category='timeSeries'
        

    def fit(self, data):

        N = len(data)
        sigma2 = np.var(data)
        
        return 1.0/((N-1)*sigma2) * np.sum(np.power(data[1:] - data[:-1] , 2))


class B_R(Base):
    '''
    average color for each MACHO lightcurve 
    mean(B1) - mean(B2)
    '''
    def __init__(self, second_data):
        self.category='timeSeries'
        if second_data is None:
            print "please provide another data series to compute B_R"
            sys.exit(1)
        self.data2 = second_data
      
    def fit(self, data):
        return np.mean(data) - np.mean(self.data2)


# The categories of the following featurs should be revised

class Amplitude(Base):
    '''
    Half the difference between the maximum and the minimum magnitude
    '''

    def __init__(self):
        self.category='basic'

    def fit(self, data):
        return (np.max(data) - np.min(data)) / 2

class Beyond1Std(Base):
    '''
    Percentage of points beyond one st. dev. from the weighted (by photometric errors) mean
    '''

    def __init__(self, error):
        self.category='basic'
        if error is None:
            print "please provide the measurement erros to compute Beyond1Std"
            sys.exit(1)
        self.error = error

    def fit(self, data):
        n = len(data)

        weighted_mean = np.average(data, weights= 1 / self.error**2)

        # Standard deviation with respect to the weighted mean
        var = 0
        for i in xrange(n):
            var += ((data[i]) - weighted_mean)**2
        std = np.sqrt( (1.0/(n-1)) * var )

        fraction = 0.0
        for i in xrange(n):
            if data[i] > weighted_mean + std or data[i] < weighted_mean - std:
                fraction += 1

        return fraction / n

class SmallKurtosis(Base):
    '''
    small sample kurtosis of the magnitudes. See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    '''

    def __init__(self):
        self.category='basic'

    def fit(self, data):
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)

        suma = 0
        for i in xrange(n):
            suma += ((data[i] - mean) / std)**4

        c1 = float(n*(n + 1)) / ((n - 1)*(n - 2)*(n - 3))
        c2 = float(3 * (n - 1)**2) / ((n-2)*(n-3))

        return c1 * suma - c2

class Std(Base):
    '''
    standard deviation of the magnitudes.
    '''

    def __init__(self):
        self.category='basic'

    def fit(self, data):
        return np.std(data)

class Skew(Base):
    '''
    skewness of the magnitudes
    '''

    def __init__(self):
        self.category='basic'

    def fit(self, data):
        return stats.skew(data)

class StetsonJ(Base):
    '''
    Stetson (1996) variability index, a robust standard deviation
    '''
    def __init__(self, entry):
        self.category='timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute StetsonJ"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]

    def fit(self, data):
        
        N = len(self.data2)

        sigmap = np.sqrt(N*1.0/(N-1)) * (self.aligned_data[:N]-np.mean(self.aligned_data))/np.std(self.aligned_data)
        sigmaq = np.sqrt(N*1.0/(N-1)) * (self.data2[:N]-np.mean(self.data2))/np.std(self.data2)
        
        sigma_i = sigmap * sigmaq
        
        J= 1.0/len(sigma_i) * np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i)))

        return J

class MaxSlope(Base):
    '''
    Examining successive (time-sorted) magnitudes, the maximal first difference (value of delta magnitude over delta time)
    '''

    def __init__(self, mjd):
        self.category='timeSeries'
        if mjd is None:
            print "please provide the measurement times to compute MaxSlope"
            sys.exit(1)
        self.mjd = mjd

    def fit(self, data):
        max_slope = 0

        index = self.mjd

        for i in xrange(len(data) - 1):
            slope = float(np.abs(data[i+1] - data[i]) / (index[i+1] - index[i]))

            if slope > max_slope:
                max_slope = slope

        return max_slope

class MedianAbsDev(Base):

    def __init__(self):
        self.category='basic'

    def fit(self, data):
        median = np.median(data)

        devs = []
        for i in xrange(len(data)):
            devs.append(abs(data[i] - median))

        return np.median(devs)

class MedianBRP(Base):
    '''
    Median buffer range percentage
    fraction (<= 1) of photometric points within amplitude/10 of the median magnitude
    '''

    def __init__(self):
        self.category='basic'

    def fit(self, data):
        median = np.median(data)
        amplitude = ( np.max(data) - np.min(data) ) / 10
        n = len(data)

        fraction = 0.0
        for i in xrange(n):
            if data[i] < median + amplitude and data[i] > median - amplitude:
                fraction += 1

        return fraction / n

class PairSlopeTrend(Base):
    '''
    considering the last 30 (time-sorted) measurements of source magnitude, 
    the fraction of increasing first differences minus the fraction of decreasing first differences.
    '''

    def __init__(self):
        self.category='timeSeries'

    def fit(self, data):
        data_last = data[-30:]

        inc = 0.0
        dec = 0.0

        for i in xrange(29):
            if data_last[i + 1] - data_last[i] > 0:
                inc += 1
            else:
                dec += 1

        return (inc - dec) / 30

class FluxPercentileRatioMid20(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        sorted_data=np.sort(data)
        lc_length=len(sorted_data)

        F_60_index=int(0.60 * lc_length)
        F_40_index=int(0.40 * lc_length)
        F_5_index=int(0.05 * lc_length)
        F_95_index=int(0.95 * lc_length)
        
        F_40_60=  sorted_data[F_60_index]-sorted_data[F_40_index]
        F_5_95= sorted_data[F_95_index]-sorted_data[F_5_index]
        F_mid20=F_40_60 / F_5_95

        return F_mid20

class FluxPercentileRatioMid35(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        sorted_data=np.sort(data)
        lc_length=len(sorted_data)
        
        F_325_index=int(0.325 * lc_length)
        F_675_index=int(0.675 * lc_length)
        F_5_index=int(0.05 * lc_length)
        F_95_index=int(0.95 * lc_length)
        
        F_325_675= sorted_data[F_675_index]- sorted_data[F_325_index]
        F_5_95= sorted_data[F_95_index]-sorted_data[F_5_index]
        F_mid35=F_325_675 / F_5_95

        return F_mid35

class FluxPercentileRatioMid50(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        sorted_data=np.sort(data)
        lc_length=len(sorted_data)
        
        F_25_index=int(0.25 * lc_length)
        F_75_index=int(0.75 * lc_length)
        F_5_index=int(0.05 * lc_length)
        F_95_index=int(0.95 * lc_length)
        
        F_25_75= sorted_data[F_75_index]- sorted_data[F_25_index]
        F_5_95= sorted_data[F_95_index]-sorted_data[F_5_index]
        F_mid50=F_25_75 / F_5_95

        return F_mid50

class FluxPercentileRatioMid65(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        sorted_data=np.sort(data)
        lc_length=len(sorted_data)
        
        F_175_index=int(0.175 * lc_length)
        F_825_index=int(0.825 * lc_length)
        F_5_index=int(0.05 * lc_length)
        F_95_index=int(0.95 * lc_length)
        
        F_175_825= sorted_data[F_825_index]- sorted_data[F_175_index]
        F_5_95=  sorted_data[F_95_index]-sorted_data[F_5_index]
        F_mid65=F_175_825 / F_5_95

        return F_mid65

class FluxPercentileRatioMid80(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        sorted_data=np.sort(data)
        lc_length=len(sorted_data)
        
        F_10_index=int(0.10 * lc_length)
        F_90_index=int(0.90 * lc_length)
        F_5_index=int(0.05 * lc_length)
        F_95_index=int(0.95 * lc_length)
        
        F_10_90=  sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95= sorted_data[F_95_index]-sorted_data[F_5_index]
        F_mid80=F_10_90 / F_5_95

        return F_mid80

class PercentDifferenceFluxPercentile(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        
        median_data=np.median(data)

        sorted_data=np.sort(data)
        lc_length=len(sorted_data)
        F_5_index=int(0.05 * lc_length)
        F_95_index=int(0.95 * lc_length)
        F_5_95= sorted_data[F_95_index]-sorted_data[F_5_index]

        percent_difference=F_5_95/median_data

        return percent_difference

class PercentAmplitude(Base):

    def __init__(self):
        self.category='basic'

    def fit(self,data):
        
        median_data=np.median(data)
        distance_median=np.abs(data-median_data)
        max_distance=np.max(distance_median)

        percent_amplitude=max_distance / median_data

        return percent_amplitude

class LinearTrend(Base):

    def __init__(self, mjd):
        self.category='timeSeries'

        if mjd is None:
            print "please provide the measurement times to compute LinearTrend"
            sys.exit(1)
        self.mjd = mjd


    def fit(self,data):

        regression_slope = stats.linregress(self.mjd, data)[0]

        return regression_slope

class Eta_B_R(Base):

    
    def __init__(self,entry):

        self.category='timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute Eta_B_R"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]
        

    def fit(self, data):


        N = len(self.aligned_data)
        B_Rdata=self.aligned_data-self.data2;
        # N = len(B_Rdata)
        sigma2 = np.var(B_Rdata)
        
        return 1.0/((N-1)*sigma2) * np.sum(np.power(B_Rdata[1:] - B_Rdata[:-1] , 2))


class Eta_e(Base):

    def __init__(self,mjd):

        self.category='timeSeries'

        if mjd is None:
            print "please provide the measurement times to compute Eta_e"
            sys.exit(1)
        self.mjd = mjd

    def fit(self,data):

        w = 1.0 / np.power(self.mjd[1:]-self.mjd[:-1] ,2)
        w_mean = np.mean(w)

        N = len(self.mjd)
        sigma2=np.var(data)

        suma = 0
        suma2 = 0
        for i in xrange(N-1):
            suma += w[i]*(data[i+1]-data[i])**2
            suma2 += w[i]

        eta_e = w_mean * np.power(self.mjd[N-1]-self.mjd[0],2) * suma / (sigma2 * suma2 * N**2)

        return eta_e
       

class Bmean(Base):

    def __init__(self):

        self.category='basic'


    def fit(self,data):

        B_mean = np.mean(data)

        return B_mean



class Q31(Base):

    def __init__(self):

        self.category='basic'

    def fit(self,data):

        return np.percentile(data,75) - np.percentile(data,25)

class Q31B_R(Base):

    def __init__(self,entry):

        self.category='timeSeries'
        # if second_data is None:
        #     print "please provide another data series to compute Q31B_R"
        #     sys.exit(1)
        self.data2 = entry[0]
        self.aligned_data = entry[1]

    def fit(self,data):

        N = len(self.data2)
        b_r=self.aligned_data[:N]-self.data2[:N];

        return np.percentile(b_r,75) - np.percentile(b_r,25)

class AndersonDarling(Base):

    def __init__(self):

        self.category='timeSeries'

    def fit(self,data):


        return stats.anderson(data)[0]



class PeriodLS(Base):

    def __init__(self,mjd):

        self.category='timeSeries'

        if mjd is None:
            print "please provide the measurement times to compute PeriodLS"
            sys.exit(1)
        self.mjd = mjd

    def fit(self,data):

        fx,fy, nout, jmax, prob = lomb.fasper(self.mjd,data, 6., 100.)
        PeriodLS.prob = prob

        return 1.0 / fx[jmax] 

    def getPeriod_fit(self):

        return PeriodLS.prob


class Period_fit(PeriodLS):

    def __init__(self):
        self.category='timeSeries'

    def fit(self, data):

        a = Period_fit()
        return a.getPeriod_fit()


class CAR_sigma(Base):
   

    def __init__(self, entry):

        self.category='timeSeries'
        self.N = len(entry[0])
        self.error= entry[1].reshape((self.N,1))**2
        self.mjd = entry[0].reshape((self.N,1))


    def CAR_Lik(self, parameters,t,x,error_vars):

        sigma = parameters[0]
        tau = parameters[1]
       #b = parameters[1] #comment it to do 2 pars estimation
       #tau = params(1,1);
       #sigma = sqrt(2*var(x)/tau);

        b = np.mean(x)/tau
        epsilon = 1e-300
        cte_neg = -np.infty
        num_datos = np.size(x)

        Omega = []
        x_hat = []
        a = []
        x_ast = []

        Omega.append((tau*(sigma**2))/2.)
        x_hat.append(0.)
        a.append(0.)
        x_ast.append(x[0] - b*tau)

        loglik = 0.

        for i in range(1,num_datos):

            a_new = np.exp(-(t[i]-t[i-1])/tau)
            x_ast.append(x[i] - b*tau)
            x_hat.append(a_new*x_hat[i-1] + (a_new*Omega[i-1]/(Omega[i-1] + error_vars[i-1]))*(x_ast[i-1]-x_hat[i-1]))
            Omega.append(Omega[0]*(1-(a_new**2)) + ((a_new**2))*Omega[i-1]*( 1 - (Omega[i-1]/(Omega[i-1]+ error_vars[i-1]))))

            loglik_inter = np.log( ((2*np.pi*(Omega[i] + error_vars[i]))**-0.5) * (np.exp( -0.5 * ( ((x_hat[i]-x_ast[i])**2) / (Omega[i] + error_vars[i]))) + epsilon))
            loglik = loglik + loglik_inter

            if(loglik <= cte_neg):
                print('CAR lik se fue a inf')
                return None

        return -loglik #the minus one is to perfor maximization using the minimize function

    def calculateCAR(self, LC):
        x0 = [10, 0.5]
        bnds = ((0, 100), (0, 100))
        res = minimize(self.CAR_Lik, x0, args=(LC[:,0],LC[:,1],LC[:,2]) ,method='nelder-mead',bounds = bnds)
        # options={'disp': True}
        sigma = res.x[0]
        CAR_sigma.tau = res.x[1] 
        return sigma

    def getAtt(self):
        return  CAR_sigma.tau   


    def fit(self, data):
        LC = np.hstack((self.mjd , data.reshape((self.N,1)), self.error))
        a = self.calculateCAR(LC)

        return a



class CAR_tau(CAR_sigma):
   

    def __init__(self):

        self.category='timeSeries'


    def fit(self, data):

        a = CAR_tau()

        return a.getAtt()


class CAR_tmean(CAR_sigma):
    
    def __init__(self):

        self.category='timeSeries'
    
    def fit(self, data):

        a = CAR_tmean()
        #return np.mean(data) / a.getAtt()
        return np.mean(data) / a.getAtt()



class SlottedA(Base):

    def __init__(self, mjd):
        """
        lc: MACHO lightcurve in a pandas DataFrame
        k: lag (default: 1)
        T: tau (slot size in days. default: 4)
        """
        self.category = 'timeSeries'


        self.mjd = mjd


    def slotted_autocorrelation(self, lc, k , T):


        # make time start from 0
        lc.index = map(lambda x: x - min(lc.index), lc.index)

        # subtract mean from mag values
        lc2 = lc.copy()

        lc2['mag'] = lc2['mag'].subtract(lc2['mag'].mean())

        min_time = min(lc2.index)
        max_time = max(lc2.index)
        current_time = lc2.index[0]
        lag_time = current_time + k * T

        N = 0
        product_sum = 0
        while lag_time < max_time - T/2.0:
            # get all the points in the two bins (current_time bin and lag_time bin)
            lc_points = lc2[np.logical_and(lc2.index >= current_time - T/2.0, lc2.index <= current_time + T/2.0)]
            lc_points_lag = lc2[np.logical_and(lc2.index >= lag_time - T/2.0, lc2.index <= lag_time + T/2.0)]

            current_time = current_time + T
            lag_time = lag_time + T

            if len(lc_points) == 0 or len(lc_points_lag) == 0:
                continue

            current_time_points = np.array(lc_points['mag'].tolist()).reshape((len(lc_points), 1))
            lag_time_points = np.array(lc_points_lag['mag'].tolist()).reshape((1, len(lc_points_lag)))
            mult_matrix = current_time_points.dot(lag_time_points)

            product_sum = product_sum + mult_matrix.sum()
            N = N + 1

        return product_sum/float(N - 1)


    def fit(self, data):


        lc = pd.DataFrame(data, index = self.mjd, columns = ['mag'])

        threshold = math.exp(-1)
        norm_value = self.slotted_autocorrelation(lc, k=0, T=4)

        T = 4
        k = 1
        current_autocorr_value = 1

        while current_autocorr_value > threshold:
            current_autocorr_value = self.slotted_autocorrelation(lc, k=k, T=T)/norm_value
            k = k + 1

        return k*T

   


