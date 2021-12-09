

import os
import numpy as np

import pyvista

path = './predict/'

file_list = os.listdir(path)


i = 0

for f in file_list:
	
	print("Step:"+str(i+1)+'/'+str(len(file_list)))
	data = np.load(path+f)
	
	pdata = pyvista.PolyData(data)
	
	pdata.save('./render/output_'+str(i)+'.vtk')
	
	i+=1
	
