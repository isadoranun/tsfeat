import numpy as np

from Feature import FeatureSpace
from import_lightcurve import LeerLC_MACHO
from PreprocessLC import Preprocess_LC
from alignLC import Align_LC


#Opening the light curve
lc_B = LeerLC_MACHO('lc_58.6272.729.B.mjd')
lc_R = LeerLC_MACHO('lc_58.6272.729.R.mjd')

[data, mjd, error] = lc_B.leerLC()
[data2, mjd2, error2] = lc_R.leerLC()

preproccesed_data = Preprocess_LC(data, mjd, error)
[data, mjd, error] = preproccesed_data.Preprocess()

preproccesed_data = Preprocess_LC(data2, mjd2, error2)
[second_data, mjd2, error2] = preproccesed_data.Preprocess()


if len(data) != len(second_data):
    [aligned_data, aligned_second_data, aligned_mjd] = Align_LC(mjd, mjd2,
                                                                data,
                                                                second_data,
                                                                error, error2)


#Calculating the features
a = FeatureSpace(featureList=['PeriodLS', 'Psi_CS', 'Psi_eta'],
                 automean=[0, 0], StetsonL=[aligned_second_data, aligned_data],
                 B_R=second_data, Beyond1Std=error,
                 StetsonJ=[aligned_second_data, aligned_data], MaxSlope=mjd,
                 LinearTrend=mjd,
                 Eta_B_R=[aligned_second_data, aligned_data, aligned_mjd],
                 Eta_e=mjd, Q31B_R=[aligned_second_data, aligned_data],
                 PeriodLS=mjd, Psi_CS=mjd, CAR_sigma=[mjd, error],
                 SlottedA=mjd
                 )

a = a.calculateFeature(data)

print a.result(method='dict')

# nombres = a.result(method='features')
# guardar = np.vstack((nombres,a.result(method='array')))
# # a=np.vstack((previous_data,a))
# np.savetxt('test_real.csv', guardar, delimiter="," ,fmt="%s")

#B_R = second_data, Eta_B_R = second_data, Eta_e = mjd, MaxSlope = mjd,
#PeriodLS = mjd, Q31B_R = second_data, StetsonJ = second_data,
#StetsonL = second_data)
