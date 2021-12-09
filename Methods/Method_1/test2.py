import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import random
import shutil
import os
import matplotlib
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np

from sklearn.neighbors import KDTree

def find_neighbor(X,neighbor_number):
	
	
	max_neigh = neighbor_number
	
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
	
	
def build_input(path,files,indexes):
    final_data = []
    final_label = []
    final_neighbor = []
    
    max_neigh = 216
    
    idx = 0
    for i in indexes:
        
	    file = path + files[i]
	    
	    
	    print(str(idx+1) + '/' + str(len(indexes)))
	    
	    data = np.fromfile(file, dtype=np.float32).reshape(-1,10)
	    np.random.shuffle(data)
	    
	    final_data.append(data[:, :7][~np.all(data[:, 7:]  == 0, axis=1)])
	    final_label.append(data[:, 7:][~np.all(data[:, 7:]  == 0, axis=1)])
	    final_neighbor.append(find_neighbor(data,max_neigh)[~np.all(data[:, 7:]  == 0, axis=1)])
	    
	    idx += 1
	    
	    
    #ret = [np.concatenate(final_data,axis=0),np.concatenate(final_neighbor,axis=0),np.concatenate(final_label,axis=0)]
    #ret[0] = ret[0][~np.all(ret[2]  == 0, axis=1)]
    #ret[1] = ret[1][~np.all(ret[2]  == 0, axis=1)]
    #ret[2]  = ret[2][~np.all(ret[2]  == 0, axis=1)]
    #ret[0] = ret[0][:,:6]
    #print(len(ret))
    #print(ret[0].shape)
    #print(ret[1].shape)
    #print(ret[2].shape)
    #input()
    return [np.array(final_data), np.array(final_neighbor), np.array(final_label)]



class FluidNet(tf.keras.Model):

	def __init__(self):
		super(FluidNet, self).__init__()
		
		self.fcn1 = tf.keras.layers.Dense(32, input_shape=(7,),activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(),name='fcn1')
		#self.fcn1b = tf.keras.layers.Dense(29, input_shape=(7,),activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(),name='fcn1b')
		#self.fcnx =  RBFLayer(7,initializer=tf.random_normal_initializer(),betas=2.0,input_shape=(7,))
		
		self.fcn2 = tf.keras.layers.Dense(64, name="fcn4", activation=tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn3 = tf.keras.layers.Dense(32, name="fcn5", activation=tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn4 = tf.keras.layers.Dense(16, name="fcn6", activation=tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn5 = tf.keras.layers.Dense(8, name="fcn7",activation=tf.nn.leaky_relu,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn6 = tf.keras.layers.Dense(3, name="fcn8", kernel_initializer=tf.random_normal_initializer())
		

	def call(self, inputs):
		feature, neighbor,label = inputs
		x = self.fcn1(neighbor)
		x = tf.reshape((tf.reduce_mean(x, axis=2, name="mean")),[tf.shape(x)[0],tf.shape(x)[1],32])
		#x = self.fcnx(x)
		#x = tf.reduce_mean(x, axis=1, name="mean")
		x_orig = tf.concat([feature, x], axis=2)
		x = self.fcn2(x_orig)
		x = self.fcn3(x)
		x = self.fcn4(x)
		x = self.fcn5(x)
		x = self.fcn6(x)
		final = x # * tf.expand_dims(tf.ones([tf.shape(x_orig)[0]]) - x_orig[:, 6], axis=1)
		#sf = tf.cast(x_orig[:, 6], tf.int32)
		
		
		#loss_full = tf.abs(label - final)
		#fluid_divide = tf.dynamic_partition(loss_full, sf, 2)
		#self.add_loss((tf.reduce_sum(fluid_divide[0])) /(tf.cast(tf.shape(fluid_divide[0])[0], tf.float32)))
		#self.add_loss((tf.reduce_sum(loss_full,1)) /(tf.cast(tf.shape(loss_full)[1], tf.float32)))
		self.add_loss(tf.reduce_mean(tf.abs(label - final)))
		
		return final




train = True


if train == True :

	#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=models_path, verbose=1,save_weights_only=False,monitor='val_loss',save_freq='epoch')
	path = "./data/"

	train_list = os.listdir(path)
	#train_list.sort()
	random.shuffle(train_list)
	train_list = train_list[:50]

	files_idx = [i for i in range(len(train_list))]
	files_idx = np.array(files_idx)



	#print(files)

	#input()

	print('Press enter to start')
	input()
	

	if os.path.exists("models"):
		shutil.rmtree("models")

	os.mkdir("./models")

	fluidnet = FluidNet()
	fluidnet.compile(optimizer="Adam")
	
	#fluidnet = tf.keras.models.load_model('./models/model')
	
	data = build_input(path,train_list ,files_idx)

	x1_tr, x1_te, x2_tr, x2_te, y_tr, y_te = train_test_split(data[0],data[1],data[2],test_size=0.2,shuffle=True)


	train_history = fluidnet.fit([x1_tr,x2_tr,y_tr],y_tr,epochs=150,batch_size=2,shuffle=True,callbacks=[],validation_data=([x1_te,x2_te,y_te],y_te),validation_freq=1)
	#train_history = fluidnet.fit([data[0],data[1],data[2]],data[2],epochs=500,batch_size=1,shuffle=True)

	fluidnet.save('./models/model')


	print("Average val loss: ", np.average(train_history.history['val_loss']))

	#idx = np.random.randint(0,(x1_te.shape)[0],(1))


	#for i in idx:
	#	x1 = np.expand_dims(x1_te[i], axis=0)
	#	x2 = np.expand_dims(x2_te[i], axis=0)
	#	y = np.expand_dims(y_te[i], axis=0)
	#	p = fluidnet.predict([x1,x2,y])
	#	print("Pred:",p)
	#	print("Original:",y)
	#	

	train_list = os.listdir(path)
	#train_list.sort()
	random.shuffle(train_list)
	train_list = train_list[:3]

	files_idx = [i for i in range(len(train_list))]
	files_idx = np.array(files_idx)

	data = build_input(path,train_list ,files_idx)




	x1 = data[0]
	x2 = data[1]
	y = data[2]
	p = fluidnet.predict([x1,x2,y])
	print("Pred:",p)
	print("Original:",y)


	percent = 100 * abs(p - y) / abs(y)

	print(percent.shape)

	color = ['r','g','b','c','m','y']
	percent = percent.reshape(percent.shape[0]*percent.shape[1],3)
	print(percent.shape)
	for i in range(0,percent.shape[1]):
		plt.hist(x=percent[:,i].flatten(), bins=[0,1,2,3,4,5,6,7,8,9,10,15, 20, 30, 40, 50, 100], color=color[i],alpha=0.7, rwidth=0.85)
		plt.savefig('./histplot'+str(i)+'.png', bbox_inches='tight')
		plt.clf()


else:
	path = "./data3/"

	train_list = os.listdir(path)
	random.shuffle(train_list)
	train_list = train_list[:5]


	files_idx = [i for i in range(len(train_list))]
	files_idx = np.array(files_idx)

	data = build_input(path,train_list ,files_idx)


	fluidnet = tf.keras.models.load_model('./models/model')

	x1 = data[0]
	x2 = data[1]
	y = data[2]
	p = fluidnet.predict([x1,x2,y])
	print("Pred:",p)
	print("Original:",y)


	percent = 100 * abs(p - y) / abs(y)

	print(percent.shape)

	color = ['r','g','b','c','m','y']
	for i in range(0,percent.shape[1]):
		plt.hist(x=percent[:,i].flatten(), bins=[0,1,2,3,4,5,6,7,8,9,10,15, 20, 30, 40, 50, 100], color=color[i],alpha=0.7, rwidth=0.85)
		plt.savefig('./histplot'+str(i)+'.png', bbox_inches='tight')
		plt.clf()




