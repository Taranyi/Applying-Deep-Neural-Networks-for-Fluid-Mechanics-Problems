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
#from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np

from keras.utils import plot_model


from keras.layers import GaussianNoise
#class InitCentersRandom(Initializer):
#    """ Initializer for initialization of centers of RBF network
#        as random samples from the given data set.

#    # Arguments
#        X: matrix, dataset to choose the centers from (random rows
#          are taken as centers)
#    """

#    def __init__(self, X):
#        self.X = X
#        super().__init__()

#    def __call__(self, shape, dtype=None):
#        assert shape[1:] == self.X.shape[1:]  # check dimension

#        # np.random.randint returns ints from [low, high) !
#        idx = np.random.randint(self.X.shape[0], size=shape[0])

#        return self.X[idx, :]


#class RBFLayer(Layer):
#    """ Layer of Gaussian RBF units.

#    # Example

#    ```python
#        model = Sequential()
#        model.add(RBFLayer(10,
#                           initializer=InitCentersRandom(X),
#                           betas=1.0,
#                           input_shape=(1,)))
#        model.add(Dense(1))
#    ```


#    # Arguments
#        output_dim: number of hidden units (i.e. number of outputs of the
#                    layer)
#        initializer: instance of initiliazer to initialize centers
#        betas: float, initial value for betas

#    """

#    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):

#        self.output_dim = output_dim

#        # betas is either initializer object or float
#        if isinstance(betas, Initializer):
#            self.betas_initializer = betas
#        else:
#            self.betas_initializer = Constant(value=betas)

#        self.initializer = initializer if initializer else RandomUniform(
#            0.0, 1.0)

#        super().__init__(**kwargs)

#    def build(self, input_shape):

#        self.centers = self.add_weight(name='centers',
#                                       shape=(self.output_dim, input_shape[1]),
#                                       initializer=self.initializer,
#                                       trainable=True)
#        self.betas = self.add_weight(name='betas',
#                                     shape=(self.output_dim,),
#                                     initializer=self.betas_initializer,
#                                     # initializer='ones',
#                                     trainable=True)

#        super().build(input_shape)

#    def call(self, x):

#        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
#        H = tf.transpose(C-tf.transpose(x))  # matrix of differences
#        return tf.exp(-self.betas * tf.math.reduce_sum(H**2, axis=1))

#    def compute_output_shape(self, input_shape):
#        return (input_shape[0], self.output_dim)

#    def get_config(self):
#        # have to define get_config to be able to use model_from_json
#        config = {
#            'output_dim': self.output_dim
#        }
#        base_config = super().get_config()
#        return dict(list(base_config.items()) + list(config.items()))

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
    print(len(ret))
    print(ret[0].shape)
    print(ret[1].shape)
    print(ret[2].shape)
    #input()
    return ret

#class DenseBlock(tf.keras.layers.Layer):
#	def __init__(self,units,input_dim):
#		super(DenseBlock,self).__init__()
#		
#		self.D1 = tf.keras.layers.Dense(units, input_shape=input_dim, kernel_initializer=tf.random_normal_initializer())
#		self.B1 = tf.keras.layers.BatchNormalization()
#		
#		self.D2 = tf.keras.layers.Dense(units/2, kernel_initializer=tf.random_normal_initializer())
#		self.B2 = tf.keras.layers.BatchNormalization()
#		
#		self.D3 = tf.keras.layers.Dense(units/4, kernel_initializer=tf.random_normal_initializer())
#		self.B3 = tf.keras.layers.BatchNormalization()
#		
#		self.DR = tf.keras.layers.Dense(units/4, kernel_initializer=tf.random_normal_initializer())
#		self.BR = tf.keras.layers.BatchNormalization()
#		
#	def call(self, inputs):
#		
#		x = tf.nn.relu(self.B1(self.D1(inputs)))
#		x = tf.nn.relu(self.B2(self.D2(x)))
#		x = tf.nn.relu(self.BR(self.DR(inputs)) + self.B3(self.D3(x)))
#		
#		return x
#		

#class IdentityBlock(tf.keras.layers.Layer):
#	def __init__(self,units,input_dim):
#		super(IdentityBlock,self).__init__()
#		
#		self.D1 = tf.keras.layers.Dense(units, input_shape=input_dim, kernel_initializer=tf.random_normal_initializer())
#		self.B1 = tf.keras.layers.BatchNormalization()
#		
#		self.D2 = tf.keras.layers.Dense(units/2, kernel_initializer=tf.random_normal_initializer())
#		self.B2 = tf.keras.layers.BatchNormalization()
#		
#		self.D3 = tf.keras.layers.Dense(units/4, kernel_initializer=tf.random_normal_initializer())
#		self.B3 = tf.keras.layers.BatchNormalization()
#		
#		
#		
#	def call(self, inputs):
#		
#		x = tf.nn.relu(self.B1(self.D1(inputs)))
#		x = tf.nn.relu(self.B2(self.D2(x)))
#		x = tf.nn.relu(inputs + self.B3(self.D3(x)))
#		
#		return x
#		


class FluidNet(tf.keras.Model):

	def __init__(self):
		super(FluidNet, self).__init__()
		
		self.fcn1 = tf.keras.layers.Dense(26, input_shape=(7,), kernel_initializer=tf.random_normal_initializer())
		#self.fcn1a = tf.keras.layers.Dense(16, input_shape=(7,), kernel_initializer=tf.random_normal_initializer())
		#self.fcn1b = tf.keras.layers.Dense(16,input_shape=(6,), kernel_initializer=tf.random_normal_initializer())
		#self.fcn1c = tf.keras.layers.Dense(32,activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer())
		#self.fcnx =  RBFLayer(7,initializer=tf.random_normal_initializer(),betas=2.0,input_shape=(7,))
		
		self.fcn2 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn3 = tf.keras.layers.Dense(16,kernel_initializer=tf.random_normal_initializer())
		
		self.fcn4 = tf.keras.layers.Dense(8,kernel_initializer=tf.random_normal_initializer())
		
		#self.gru1 = tf.keras.layers.GRU(32,return_sequences=True)
		#self.gru2 = tf.keras.layers.GRU(16,return_sequences=True)
		#self.gru3 = tf.keras.layers.GRU(8)
		
		#self.b1 = tf.keras.layers.BatchNormalization()
		#self.b2 = tf.keras.layers.BatchNormalization()
		#self.b3 = tf.keras.layers.BatchNormalization()
		#self.b4 = tf.keras.layers.BatchNormalization()
		
		#self.gl1 = tf.keras.layers.GaussianNoise(0.0003,input_shape=(6,))
		#self.gl2 = tf.keras.layers.GaussianNoise(0.0003,input_shape=(7,))
		#self.d1 = tf.keras.layers.Dropout(.2)
		#self.d2 = tf.keras.layers.Dropout(.2)
		#self.d3 = tf.keras.layers.Dropout(.2)
		
		
		self.fcn5 = tf.keras.layers.Dense(3, kernel_initializer=tf.random_normal_initializer())
		

	def call(self, inputs):
		feature, neighbor,label = inputs
		#x = self.fcn1c(self.fcn1b(self.fcn1a(neighbor)))
		#boolean_mask = tf.cast(tf.reduce_sum(tf.abs(neighbor),2), dtype=tf.bool)
		#x = tf.boolean_mask(neighbor, boolean_mask, axis=0)
		x = tf.nn.leaky_relu((self.fcn1((neighbor))))
		x = tf.reduce_mean(x, axis=1, name="mean")
		x = tf.concat([(feature), x], axis=1)
		#x = tf.nn.relu(self.fcn1b(feature) +  tf.reduce_mean(self.fcn1a(neighbor), axis=1))
		#x = self.d1(tf.nn.relu(self.b1(self.fcn2(x))))
		#x = self.d2(tf.nn.relu(self.b2(self.fcn3(x))))
		##x = self.d3(tf.nn.relu(self.b3(self.fcn4(x))))
		#x = self.gru1(tf.keras.layers.TimeDistributed(x))
		#x = self.gru2(x)
		#x = self.gru3(x)
		#final = self.fcn5(x)
		#final = x # * tf.expand_dims(tf.ones([tf.shape(x_orig)[0]]) - x_orig[:, 6], axis=1)
		#sf = tf.cast(x_orig[:, 6], tf.int32)
		
		x = tf.nn.leaky_relu((self.fcn2(x)))
		x = tf.nn.leaky_relu((self.fcn3(x)))
		x = tf.nn.leaky_relu((self.fcn4(x)))
		x = self.fcn5(x)
		
		
		
		#loss_full = tf.abs(label - final)
		#fluid_divide = tf.dynamic_partition(loss_full, sf, 2)
		#self.add_loss((tf.reduce_sum(fluid_divide[0])) /(tf.cast(tf.shape(fluid_divide[0])[0], tf.float32)))
		#self.add_loss((tf.reduce_sum(loss_full)) /(tf.cast(tf.shape(loss_full)[0], tf.float32)))
		self.add_loss(tf.reduce_mean(tf.abs(label - x)))
		
		return x

#class FluidNet(tf.keras.Model):

#	def __init__(self):
#		super(FluidNet, self).__init__()
#		
#		self.featureNet = tf.keras.layers.Dense(26, input_shape=(7,),activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer())
#		
#		self.block = DenseBlock(64,(32,))
#		
#		self.blockId1 = IdentityBlock(64,(16,))
#		self.blockId2 = IdentityBlock(64,(16,))
#		
#		self.fcnOut = tf.keras.layers.Dense(3, kernel_initializer=tf.random_normal_initializer())
#		

#	def call(self, inputs):
#		feature, neighbor,label = inputs
#		x = self.featureNet(neighbor)
#		x = tf.reduce_mean(x, axis=1)
#		x = tf.concat([feature, x], axis=1)
#		
#		x = self.block(x)
#		x = self.blockId1(x)
#		x = self.blockId2(x)
#		
#		final = self.fcnOut(x)
#		
#		self.add_loss(tf.reduce_mean(tf.abs(label - final)))
#		
#		return final



train = True


if train == True :

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./models/", verbose=1,save_weights_only=False,monitor='val_loss',save_freq='epoch')
	path = "./dataEND/"
	
	#log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq='batch')


	train_list = os.listdir(path)
	#train_list.sort()
	random.shuffle(train_list)
	train_list = train_list[:]

	files_idx = [i for i in range(len(train_list))]
	files_idx = np.array(files_idx)



	#print(files)

	#input()

	print('Press enter to start')
	input()
	

	#if os.path.exists("models"):
	#	shutil.rmtree("models")

	#os.mkdir("./models")

	fluidnet = FluidNet()
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	fluidnet.compile(optimizer=opt)
	
	
	
	#fluidnet = tf.keras.models.load_model('./models/model')
	
	full = int(len(files_idx)/30)
	
	selected_files = np.array_split(files_idx,full)
	
	epoch = 1
	
	f = 0
	
	for e in range(epoch):
		random.shuffle(train_list)
		for idx in selected_files[:1]:
			print('Epoch:',e)
			print('Files:' + str(f+1) + '/' + str(full))
			if f > 0:
				#fluidnet = tf.keras.models.load_model('./models/model')
				fluidnet.load_weights("./models/")
			
			data = build_input(path,train_list ,idx)

			x1_tr, x1_te, x2_tr, x2_te, y_tr, y_te = train_test_split(data[0],data[1],data[2],test_size=0.2,shuffle=True)

			log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
			tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq='batch')


			train_history = fluidnet.fit([x1_tr,x2_tr,y_tr],y_tr,epochs=1,batch_size=512,shuffle=True,callbacks=[tensorboard_callback,cp_callback],validation_data=([x1_te,x2_te,y_te],y_te),validation_freq=1)

			fluidnet.save('./models/model')
			f+=1
			
	
	print("Average val loss: ", np.average(train_history.history['val_loss']))
	
	#plot_model(fluidnet, to_file='model.png')

	#idx = np.random.randint(0,(x1_te.shape)[0],(15))


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

	print(percent.shape)

	color = ['r','g','b','c','m','y']
	for i in range(0,percent.shape[1]):
		plt.hist(x=percent[:,i].flatten(), bins=[0,1,2,3,4,5,6,7,8,9,10,15, 20, 30, 40, 50, 100], color=color[i],alpha=0.7, rwidth=0.85)
		plt.savefig('./histplot'+str(i)+'.png', bbox_inches='tight')
		plt.clf()


#else:
#	path = "./data3/"

#	train_list = os.listdir(path)
#	random.shuffle(train_list)
#	train_list = train_list[:5]


#	files_idx = [i for i in range(len(train_list))]
#	files_idx = np.array(files_idx)

#	data = build_input(path,train_list ,files_idx)


#	fluidnet = tf.keras.models.load_model('./models/model')

#	x1 = data[0]
#	x2 = data[1]
#	y = data[2]
#	p = fluidnet.predict([x1,x2,y])
#	print("Pred:",p)
#	print("Original:",y)


#	percent = 100 * abs(p - y) / abs(y)

#	print(percent.shape)

#	color = ['r','g','b','c','m','y']
#	for i in range(0,percent.shape[1]):
#		plt.hist(x=percent[:,i].flatten(), bins=[0,1,2,3,4,5,6,7,8,9,10,15, 20, 30, 40, 50, 100], color=color[i],alpha=0.7, rwidth=0.85)
#		plt.savefig('./histplot'+str(i)+'.png', bbox_inches='tight')
#		plt.clf()




