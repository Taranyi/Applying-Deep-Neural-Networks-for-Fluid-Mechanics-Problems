
import tensorflow as tf
import numpy as np
import os

from sklearn.neighbors import KDTree


import matplotlib
import matplotlib.pyplot as plt

def find_neighbor(X,neighbor_number):
	
	
	max_neigh = 216
	
	tree = KDTree(X[:,:3], leaf_size=3)  
	#ind = tree.query(X[:,:3], k=neighbor_number,return_distance=False)
	ind = tree.query_radius(X[:,:3], r=0.1)  
	
	par = 0
	
	Y = []
	
	for indx in ind:
		
		sel_ind = indx[1:]
		if sel_ind.shape[0] > max_neigh:
			neighbor_particle = X[sel_ind[:max_neigh],:7]
			neighbor_particle[:,:3] = abs(neighbor_particle[:,:3]-X[par,:3])
			Y.append(neighbor_particle)
		elif sel_ind.shape[0] < max_neigh:
			fill_num = max_neigh - sel_ind.shape[0]
			neighbor_particle = X[sel_ind,:7]
			neighbor_particle[:,:3] = abs(neighbor_particle[:,:3]-X[par,:3])
			fill_particles = np.concatenate(((np.random.randint(0, 2, size=(fill_num, 3)) * 2 - 1) * 0.1 * 3,np.zeros([fill_num, 4])), axis=1)
			neighbor_particle = np.concatenate((neighbor_particle, fill_particles), axis=0)
			Y.append(neighbor_particle)
		else:
			neighbor_particle = X[sel_ind,:7]
			neighbor_particle[:,:3] = abs(neighbor_particle[:,:3]-X[par,:3])
			Y.append(neighbor_particle)
			
			
		par += 1
	
	Y = np.array(Y)
	
	return Y
	
	
#def build_input(path,files,indexes):
#    final_data = []
#    final_label = []
#    final_neighbor = []
#    
#    idx = 0
#    for i in indexes:
#        
#	    file = path + files[i]
#	    
#	    
#	    print(str(idx+1) + '/' + str(len(indexes)))
#	    
#	    data = np.fromfile(file, dtype=np.float32).reshape(-1,10)
#	    np.random.shuffle(data)
#	    
#	    
#	    final_data.append(data[:, :7])
#	    final_label.append(data[:, 7:])
#	    
#	    
#	    final_neighbor.append(find_neighbor(data,216))
#	    
#	    idx += 1
#	    
#	    
#    #ret = [final_data, final_neighbor, final_label]
#    ret = [np.concatenate(final_data,axis=0),np.concatenate(final_neighbor,axis=0),np.concatenate(final_label,axis=0)]
#    ret[0] = ret[0][~np.all(ret[2]  == 0, axis=1)]
#    ret[1] = ret[1][~np.all(ret[2]  == 0, axis=1)]
#    ret[2]  = ret[2][~np.all(ret[2]  == 0, axis=1)]
#    ret[0] = ret[0][:,:6]
#    print(len(ret))
#    print(ret[0].shape)
#    print(ret[1].shape)
#    print(ret[2].shape)
#    #input()
#    return ret

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

file = './start2.bin'
data = np.fromfile(file, dtype=np.float32).reshape(-1,10)
data = data[:,:7]



model = tf.keras.models.load_model('./models/model')

model.summary()

time = 0.001

fps = 25


for i in range(5000):
	
	print('Step:',i*time)
	
	x1 = data[:,:6]
	x2 = find_neighbor(data,216)
	#label = data[:,7:]
	
	x1 = x1[data[:,6] == 0,:]
	x2 = x2[data[:,6] == 0,:]
	
	y = np.zeros((x1.shape[0],3))
	
	v = model.predict([x1,x2,y])
	
	
	
	data[data[:,6] == 0,3:6] = v[:]
	data[data[:,6] == 0,:3] += time * data[data[:,6] == 0,3:6]
	
	#if (time*1000)%fps == 0:
	np.save('./predict/file_'+str(time*i)+'.npy', data[data[:,6] == 0,:3])
	
	
