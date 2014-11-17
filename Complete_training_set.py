import numpy as np
import os.path
import shutil
import tarfile

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


count = 0
folder = 1

path = '/Users/isadoranun/Dropbox/lightcurves/'
path2 = '/Volumes/LaCie/MACHO_LMC/'
path3 = '/Volumes/LaCie/Resultados/Ordenando/'


for j in os.listdir(path):
    
    if os.path.isdir(path + j):

        for i in os.listdir(path + j):

            if i.endswith("B.mjd") and not i.startswith('.') and os.path.isfile(path + j +'/'+ i[:-5] + 'R.mjd')== False:
        		
            	a = find_between(i, "_", ".")
            	b = find_between(i, ".", ".")
            	c = find_between(i, b+".", ".B")

            	if os.path.isfile(path3 + "F_" + a + "/" + b + "/" + i[:-5] + 'R.mjd'):
            		shutil.copy(path3 + "F_" + a + "/" + b + "/" + i[:-5] + 'R.mjd', path + j)
            	else:
            		
            		if os.path.isdir(path2 + "F_" + a ):

            			
		            	with tarfile.open(path2 + "F_" + a + "/" + b + ".tar") as tar:
		    				subdir_and_files = [
		    					tarinfo for tarinfo in tar.getmembers()
		    					if tarinfo.name.endswith(c+ ".R.mjd")
		    					]

			        		tar.extractall(path + j ,members=subdir_and_files )
			        		
		            	
		            	#shutil.copy(path2 + "F_" + a + "/" + b + "/" + i[:-5] + 'R.mjd', path + j)

		            
										          



