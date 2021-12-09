
import tensorflow as tf
import numpy as np
import os

from sklearn.neighbors import KDTree


import matplotlib
import matplotlib.pyplot as plt

def find_neighbor(X,neighbor_number):
	
	
	max_neigh = neighbor_number
	
	tree = KDTree(X[:,:2], leaf_size=3)  
	#ind = tree.query(X[:,:3], k=neighbor_number,return_distance=False)
	ind = tree.query_radius(X[:,:2], r=0.1)  
	
	par = 0
	
	Y = []
	
	for indx in ind:
		
		sel_ind = indx[1:]
		if sel_ind.shape[0] > max_neigh:
			neighbor_particle = X[sel_ind[:max_neigh],:5]
			neighbor_particle[:,:2] = abs(neighbor_particle[:,:2]-X[par,:2])
			#np.random.shuffle(neighbor_particle)
			Y.append(neighbor_particle)
		elif sel_ind.shape[0] < max_neigh:
			fill_num = max_neigh - sel_ind.shape[0]
			neighbor_particle = X[sel_ind,:5]
			neighbor_particle[:,:2] = abs(neighbor_particle[:,:2]-X[par,:2])
			#fill_particles = np.concatenate(((np.random.randint(0, 2, size=(fill_num, 3)) * 2 - 1) * 0.1 * 3,np.zeros([fill_num, 4])), axis=1)
			fill_particles = np.zeros([fill_num, 5])
			neighbor_particle = np.concatenate((neighbor_particle, fill_particles), axis=0)
			#np.random.shuffle(neighbor_particle)
			Y.append(neighbor_particle)
		else:
			neighbor_particle = X[sel_ind,:5]
			neighbor_particle[:,:2] = abs(neighbor_particle[:,:2]-X[par,:2])
			#np.random.shuffle(neighbor_particle)
			Y.append(neighbor_particle)
			
			
		par += 1
	
	Y = np.array(Y)
	
	return Y
	
	
file = './start2d.bin'
data = np.fromfile(file, dtype=np.float32).reshape(-1,10)

data = np.delete(data, 2, 1)
data = np.delete(data, 4, 1)
data = np.delete(data, 7, 1)

data = data[:,:5]



model = tf.keras.models.load_model('./models/model')

model.summary()

time = 0.001

fps = 25


for i in range(5000):
	
	print('Step:',i*time)
	
	x1 = data[:,:4]
	x2 = find_neighbor(data,32)
	#label = data[:,7:]
	
	x1 = x1[data[:,4] == 0,:]
	x2 = x2[data[:,4] == 0,:]
	
	y = np.zeros((x1.shape[0],2))
	
	v = model.predict([x1,x2,y])
	
	
	
	data[data[:,4] == 0,2:4] = v[:]
	data[data[:,4] == 0,:2] += time * data[data[:,4] == 0,2:4]
	
	#if (time*1000)%fps == 0:
	np.save('./predict/file_'+str(time*i)+'.npy', data[data[:,4] == 0,:2])
	
	
