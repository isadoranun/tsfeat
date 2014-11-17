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

import warnings
warnings.filterwarnings('ignore')
#for subdir in /n/seasfs03/IACS/TSC/MACHO/MACHO_LMC/F*; do cd "$subdir"; for f in ./*tar; do readlink -f "$f" ;done; cd ..; done; 
#for f in /n/seasfs03/IACS/TSC/MACHO/MACHO_LMC/F_1/*tar;do sbatch /n/home10/inun/Extract_features/run_extractF2.sh  "$readlink -f "$f"" ; done; 


def main(argv):

	check = False

	if tarfile.is_tarfile(self.path):
		df = []
		contador = 0

		tar = tarfile.open(self.path, 'r')

		for member in tar.getmembers():

			if member.name.endswith("B.mjd"):

				id = member.name.split('lc_')[1]

				for member2 in tar.getmembers():

					if member2.name == (member.name[:-5] + 'R.mjd'):

						check = True
						f = tar.extractfile(member)
						g = tar.extractfile(member2)

						content1 = f.read().split('\n')
						content2 = g.read().split('\n')

						lc_B = LeerLC_MACHO(content1)
	    				lc_R = LeerLC_MACHO(content2)
	    				[data, mjd, error] = lc_B.leerLC()
	    				[data2, mjd2, error2] = lc_R.leerLC()

	    				preproccesed_data = Preprocess_LC(data, mjd, error)
	    				[data, mjd, error] = preproccesed_data.Preprocess()

	    				preproccesed_data = Preprocess_LC(data2, mjd2, error2)
	    				[second_data, mjd2, error2] = preproccesed_data.Preprocess()

	    				if len(data) != len(second_data):
	    					[aligned_data, aligned_second_data, aligned_mjd] = Align_LC(mjd, mjd2, data, second_data, error, error2)
	    				else:
	    					aligned_data = data
	    					aligned_second_data = second_data
	    					aligned_mjd = mjd

	    				a = FeatureSpace(featureList=['Bmean'], automean=[0,0], StetsonL=[aligned_second_data, aligned_data] , B_R=second_data, Beyond1Std=error, StetsonJ=[aligned_second_data, aligned_data], MaxSlope=mjd, LinearTrend=mjd, Eta_B_R=[aligned_second_data, aligned_data, aligned_mjd], Eta_e=mjd, Q31B_R=[aligned_second_data, aligned_data], PeriodLS=mjd, CAR_sigma=[mjd, error], SlottedA = mjd)

	    				try:
	    					a=a.calculateFeature(data)
	    					idx = [id[:-6]]
	    					contador = contador + 1

	    					if contador == 1:
	    						df = pd.DataFrame(a.result(method='array').reshape((1,len(a.result(method='array')))), columns = a.result(method='features'), index =[idx])

	    					#df.to_csv('sabrina.csv')
	    					else:
	    						df2 = pd.DataFrame(a.result(method='array').reshape((1,len(a.result(method='array')))), columns = a.result(method='features'), index =[idx])
	    						df = pd.concat([df, df2])

	    				except:
	    					pass

	if check:
		folder = (member.name.split('lc')[0]).split('/')[0]
		field = (member.name.split('lc')[0]).split('/')[1]
		file_name = folder + '_' + field + '.csv'
		df.to_csv(file_name) 


