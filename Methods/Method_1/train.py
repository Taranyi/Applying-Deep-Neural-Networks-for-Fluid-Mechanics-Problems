import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import random
import shutil
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np

from keras.utils import plot_model


from keras.layers import GaussianNoise
from sklearn.neighbors import KDTree

def find_neighbor(X,neighbor_number,rad):
	
	
	max_neigh = neighbor_number
	
	tree = KDTree(X[:,:3], leaf_size=40)  
	#ind = tree.query(X[:,:3], k=neighbor_number,return_distance=False)
	ind = tree.query_radius(X[:,:3], r=rad)  
	
	par = 0
	
	Y = []
	
	for indx in ind:
		
		sel_ind = indx[1:]
		if sel_ind.shape[0] > max_neigh:
			neighbor_particle = X[sel_ind[:max_neigh],:7]
			neighbor_particle[:,:3] = abs(neighbor_particle[:,:3]-X[par,:3])
			#np.random.shuffle(neighbor_particle)
			Y.append(neighbor_particle)
		elif sel_ind.shape[0] < max_neigh:
			fill_num = max_neigh - sel_ind.shape[0]
			neighbor_particle = X[sel_ind,:7]
			neighbor_particle[:,:3] = abs(neighbor_particle[:,:3]-X[par,:3])
			#fill_particles = np.concatenate(((np.random.randint(0, 2, size=(fill_num, 3)) * 2 - 1) * 0.1 * 3,np.zeros([fill_num, 4])), axis=1)
			fill_particles = np.zeros([fill_num, 7])
			neighbor_particle = np.concatenate((neighbor_particle, fill_particles), axis=0)
			#np.random.shuffle(neighbor_particle)
			Y.append(neighbor_particle)
		else:
			neighbor_particle = X[sel_ind,:7]
			neighbor_particle[:,:3] = abs(neighbor_particle[:,:3]-X[par,:3])
			#np.random.shuffle(neighbor_particle)
			Y.append(neighbor_particle)
			
			
		par += 1
	
	Y = np.array(Y)
	
	return Y
	
	
def build_input(path,files,indexes):
    final_data = []
    final_label = []
    final_neighbor = []
    
    
    max_neigh = 216
    radius = 0.1
    
    idx = 0
    for i in indexes:
        
	    file = path + files[i]
	    
	    
	    print(str(idx+1) + '/' + str(len(indexes)))
	    
	    data = np.fromfile(file, dtype=np.float32).reshape(-1,10)
	    np.random.shuffle(data)
	    
	    
	    final_data.append(data[:, :7])
	    final_label.append(data[:, 7:])
	    
	    
	    final_neighbor.append(find_neighbor(data,max_neigh,radius))
	    
	    idx += 1
	    
	    
    #ret = [final_data, final_neighbor, final_label]
    ret = [np.concatenate(final_data,axis=0),np.concatenate(final_neighbor,axis=0),np.concatenate(final_label,axis=0)]
    ret[0] = ret[0][~np.all(ret[2]  == 0, axis=1)]
    ret[1] = ret[1][~np.all(ret[2]  == 0, axis=1)]
    ret[2]  = ret[2][~np.all(ret[2]  == 0, axis=1)]
    ret[0] = ret[0][:,:6]
    #print(len(ret))
    #print(ret[0].shape)
    #print(ret[1].shape)
    #print(ret[2].shape)
    #input()
    return ret


class FluidNet(tf.keras.Model):

	def __init__(self):
		super(FluidNet, self).__init__()
		
		self.fcn1 = tf.keras.layers.Dense(26, input_shape=(7,), kernel_initializer=tf.random_normal_initializer())
		
		self.fcn2 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn3 = tf.keras.layers.Dense(16,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn4 = tf.keras.layers.Dense(8,kernel_initializer=tf.random_normal_initializer())
		
		
		#self.b1 = tf.keras.layers.BatchNormalization()
		#self.b2 = tf.keras.layers.BatchNormalization()
		#self.b3 = tf.keras.layers.BatchNormalization()
		#self.b4 = tf.keras.layers.BatchNormalization()
		
		#self.d1 = tf.keras.layers.Dropout(.2)
		#self.d2 = tf.keras.layers.Dropout(.2)
		#self.d3 = tf.keras.layers.Dropout(.2)
		
		
		self.fcn5 = tf.keras.layers.Dense(3, kernel_initializer=tf.random_normal_initializer())
		

	def call(self, inputs):
		feature, neighbor,label = inputs
		
		x = tf.nn.leaky_relu((self.fcn1((neighbor))))
		x = tf.reduce_mean(x, axis=1, name="mean")
		x = tf.concat([(feature), x], axis=1)

		
		x = tf.nn.leaky_relu((self.fcn2(x)))
		x = tf.nn.leaky_relu((self.fcn3(x)))
		x = tf.nn.leaky_relu((self.fcn4(x)))
		x = self.fcn5(x)
		
		
		
		self.add_loss(tf.reduce_mean(tf.abs(label - x)))
		
		return x







cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./models/model", verbose=1,save_weights_only=False,monitor='val_loss',save_freq='epoch')
path = "./dataEND/"


train_list = os.listdir(path)
#train_list.sort()
random.shuffle(train_list)
train_list = train_list[:]

files_idx = [i for i in range(len(train_list))]
files_idx = np.array(files_idx)


print('Press enter to start')
input()



fluidnet = FluidNet()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
fluidnet.compile(optimizer=opt)


full = int(len(files_idx)/15)

selected_files = np.array_split(files_idx,full)

epoch = 10

f = 0

for e in range(epoch):
	random.shuffle(train_list)
	for idx in selected_files[:]:
		print('Epoch:',e)
		print('Files:' + str(f+1) + '/' + str(full))
		if f > 0:
			fluidnet.load_weights("./models/model")
			
		data = build_input(path,train_list ,idx)

		x1_tr, x1_te, x2_tr, x2_te, y_tr, y_te = train_test_split(data[0],data[1],data[2],test_size=0.2,shuffle=True)


		train_history = fluidnet.fit([x1_tr,x2_tr,y_tr],y_tr,epochs=1,batch_size=512,shuffle=True,callbacks=[cp_callback],validation_data=([x1_te,x2_te,y_te],y_te),validation_freq=1)

		fluidnet.save('./models/model')
		f+=1
			

print("Average val loss: ", np.average(train_history.history['val_loss']))


train_list = os.listdir(path)
#train_list.sort()
random.shuffle(train_list)
train_list = train_list[:10]

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


color = ['r','g','b','c','m','y']
for i in range(0,percent.shape[1]):
	plt.hist(x=percent[:,i].flatten(), bins=[0,1,2,3,4,5,6,7,8,9,10,15, 20, 30, 40, 50, 100], color=color[i],alpha=0.7, rwidth=0.85)
	plt.savefig('./histplot'+str(i)+'.png', bbox_inches='tight')
	plt.clf()




