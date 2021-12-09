

import os
import numpy as np

import pyvista as pv

path = './predict/'

file_list = os.listdir(path)

file_list  = np.sort(file_list)
i = 0



#plotter = pv.Plotter(notebook=False, off_screen=True)
#plotter.open_gif("wave.gif")


for f in file_list:
	
	print("Step:"+str(i+1)+'/'+str(len(file_list)))
	data = np.load(path+f)
	
	pdata = pv.PolyData(data)
	
	plotter = pv.Plotter(off_screen=True)
	
	#if i%200 == 0:
	sphere = pv.Sphere(radius=0.025, phi_resolution=10, theta_resolution=10)
	pc = pdata.glyph(scale=False, geom=sphere)
	plotter.add_mesh(pc)
		
	plotter.show(screenshot='./pic/'+str(i)+'.png')
		#pc.plot(cmap='Reds')
	
	#pdata.save('./render/output_'+str(i)+'.vtk')
	
	i+=1
	
