

import os
import numpy as np
import pickle

import pyvista as pv


with open('./rollouts/Water-3D/rollout_test_0.pkl', "rb") as file:
	data = pickle.load(file)

data = np.array(data['predicted_rollout'])



i = 0

for d in data:
	
	print("Step:"+str(i+1)+'/'+str(data.shape[0]))
	
	
	pdata = pv.PolyData(d)
	
	plotter = pv.Plotter(off_screen=True)
	
	#if i%200 == 0:
	sphere = pv.Sphere(radius=0.025, phi_resolution=10, theta_resolution=10)
	pc = pdata.glyph(scale=False, geom=sphere)
	plotter.add_mesh(pc)
		
	plotter.show(screenshot='./pic/'+str(i)+'.png')
	#pc.plot(cmap='Reds')
	
	#pdata.save('./render/output_'+str(i)+'.vtk')
	
	i+=1
	
