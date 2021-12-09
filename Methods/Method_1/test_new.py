import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

#import prepare as cfg
#from model import *
#from tools import *
#from tf_init import *
#from build import *
import random
import shutil
import os

import h5py

#train_list = cfg.predict_csv_list
#train_list = cfg.train_csv_list


data_path = './newdata/'

files = os.listdir(data_path)
files = files[:]

files_idx = [i for i in range(len(files))]
files_idx = np.array(files_idx)



print(files)

input()

class FluidNet(tf.keras.Model):

	def __init__(self):
		super(FluidNet, self).__init__()
		
		self.fcn1 = tf.keras.layers.Dense(9, input_shape=(7,),activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer,name='fcn1')
		
		
		self.fcn2 = tf.keras.layers.Dense(32, name="fcn2", activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer)
		
		self.fcn3 = tf.keras.layers.Dense(16, name="fcn3", activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer)
		
		self.fcn4 = tf.keras.layers.Dense(8, name="fcn4", activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer)
		
		self.fcn5 = tf.keras.layers.Dense(3, name="fcn5", kernel_initializer=tf.random_normal_initializer)
		

	def call(self, inputs):
		feature, neighbor,label = inputs
		x = self.fcn1(neighbor)
		x = tf.reduce_mean(x, axis=1, name="mean")
		x_orig = tf.concat([feature, x], axis=1)
		x = self.fcn2(x_orig)
		x = self.fcn3(x)
		x = self.fcn4(x)
		x = self.fcn5(x)
		final = x #* tf.expand_dims(tf.ones([tf.shape(x_orig)[0]]) - x_orig[:, 6], axis=1)
		#sf = tf.cast(x_orig[:, 6], tf.int32)
		
		
		loss_full = tf.abs(label - final)
		#fluid_divide = tf.dynamic_partition(loss_full, sf, 2)
		#fluid_truth = tf.dynamic_partition(label, sf, 2)
		#self.add_loss((tf.reduce_sum(fluid_divide[0])) /(tf.cast(tf.shape(fluid_divide[0])[0], tf.float32)))
		self.add_loss((tf.reduce_sum(loss_full)) /(tf.cast(tf.shape(loss_full)[0], tf.float32)))
		
		return final


models_path = './models'

if os.path.exists("models"):
	shutil.rmtree("models")

os.mkdir("./models")

fluidnet = FluidNet()
fluidnet.compile(optimizer="Adam")

#fluidnet.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=models_path, verbose=1,save_weights_only=True,monitor='val_loss',save_freq='epoch')

fluidnet.save_weights(models_path)



def get_data(files,files_idx):

	list_x1 = np.empty((0,7), np.float32)
	list_x2 = np.empty((0,27,7), np.float32)
	list_y = np.empty((0,3), np.float32)


	for i in files_idx:
		print('Loading:'+str(i+1)+'/'+str(len(files_idx)))
		hf = h5py.File('./newdata/'+files[i], 'r')
		x1 = np.array(hf.get('features'))
		x2 = np.array(hf.get('neighbor'))
		y = np.array(hf.get('labels'))
		hf.close()
		
		list_x1 = np.append(list_x1, x1, axis=0)
		list_x2 = np.append(list_x2, x2, axis=0)
		list_y = np.append(list_y, y, axis=0)
		
		
	
	return [list_x1,list_x2,list_y]
	
	

print('Press enter to start')
input()

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
max_epoch = 1


divide = 50
selected_files = np.array_split(files_idx,divide)



for e in range(max_epoch):
	
	random.shuffle(files)
	
	step = 0
	
	for f_idx in selected_files:
		
		#for i in range(count):
		print('Out Epoch:',e)
		print('Step:',step)
		fluidnet.load_weights(models_path)
		data = get_data(files,f_idx)
		x1_tr, x1_te, x2_tr, x2_te, y_tr, y_te = train_test_split(data[0],data[1],data[2],test_size=0.333,shuffle=True)
			
			
		train_history = fluidnet.fit([x1_tr,x2_tr,y_tr],y_tr,epochs=1,batch_size=128,shuffle=True,callbacks=[cp_callback],validation_data=([x1_te,x2_te,y_te],y_te),validation_freq=1)
			
		idx = np.random.randint(0,(x1_te.shape)[0],(5))
			
		for i in idx:
			x1 = np.expand_dims(x1_te[i], axis=0)
			x2 = np.expand_dims(x2_te[i], axis=0)
			y = np.expand_dims(y_te[i], axis=0)
			p = fluidnet.predict([x1,x2,y])
			print("Pred:",p)
			print("Original:",y)
		
		#p = fluidnet.predict([x1_te,x2_te,y_te])
		
		#rel = np.mean(np.divide(abs(y_te - p), abs(y_te), out=np.zeros_like(abs(y_te - p)), where=abs(y_te)!=0)) * 100.0
		#rel = abs(1.0-np.mean(p/y_te)) * 100
		
		#print('Relative error:',rel)
		
		del x1_tr,x1_te,x2_tr,x2_te,y_tr,y_te,data
		step += 1



#for e in range(max_epoch):
#	random.shuffle(files)
#	
#	for i in files_idx:
#		print('Out Epoch:',e)
#		print(str(i+1)+'/'+str(len(files_idx)))
#		x1_tr, x1_te, x2_tr, x2_te, y_tr, y_te = train_test_split(x1,x2,y,test_size=0.333,shuffle=True)
#		
#		del x1,x2,y
#		
#		epoch = 1
#		batch = 128

#		#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#		train_history = fluidnet.fit([x1_tr,x2_tr,y_tr],y_tr,epochs=epoch,batch_size=batch,shuffle=True,callbacks=[cp_callback],validation_data=([x1_te,x2_te,y_te],y_te),validation_freq=1)
#		
#		del x1_tr,x1_te,x2_tr,x2_te,y_tr,y_te



print("Average val loss: ", np.average(train_history.history['val_loss']))











