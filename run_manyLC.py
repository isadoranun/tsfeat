
from Feature import FeatureSpace
import numpy as np
from import_lightcurve import LeerLC_MACHO
from PreprocessLC import Preprocess_LC
import os.path

count = 0
folder = 1

path = '/Users/isadoranun/Dropbox/lightcurves/'

for j in os.listdir(path):
    
    if os.path.isdir(path + j):

        for i in os.listdir(path + j):

            if i.endswith("B.mjd") and os.path.isfile(path + j +'/'+ i[2:-5] + 'R.mjd'):

                count = count + 1

                lc_B = LeerLC_MACHO(path + j +'/'+ i[2:])
                lc_R = LeerLC_MACHO(path + j +'/'+ i[2:-5] + 'R.mjd')

        #Opening the light curve

                [data, mjd, error] = lc_B.leerLC()
                [data2, mjd2, error2] = lc_R.leerLC()

                preproccesed_data = Preprocess_LC(data, mjd, error)
                [data, mjd, error] = preproccesed_data.Preprocess()

                preproccesed_data = Preprocess_LC(data2, mjd2, error2)
                [second_data, mjd2, error2] = preproccesed_data.Preprocess()

                a = FeatureSpace(category='all',featureList=None, automean=[0,0], StetsonL=second_data ,  B_R=second_data, Beyond1Std=error, StetsonJ=second_data, MaxSlope=mjd, LinearTrend=mjd, Eta_B_R=second_data, Eta_e=mjd, Q31B_R=second_data, PeriodLS=mjd, CAR_sigma=[mjd, error], SlottedA = mjd)
                a=a.calculateFeature(data)

                if count == 1:
                	nombres = np.hstack(("MACHO_Id" , a.result(method='features') , "Class"))
                	guardar = np.vstack((nombres, np.hstack((i[5:-6] , a.result(method='array') , folder ))))
                	np.savetxt('test_real.csv', guardar, delimiter="," ,fmt="%s")

                else:
                	my_data = np.genfromtxt('test_real.csv', delimiter=',', dtype=None)
                	guardar = np.vstack((nombres,my_data[1:], np.hstack((i[5:-6] , a.result(method='array') , folder ))))
                	np.savetxt('test_real.csv', guardar, delimiter="," ,fmt="%s")


        folder = folder + 1            

        #B_R = second_data, Eta_B_R = second_data, Eta_e = mjd, MaxSlope = mjd, PeriodLS = mjd, Q31B_R = second_data, StetsonJ = second_data, StetsonL = second_data)