
import tensorflow as tf
import numpy as np
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import prepare as cfg
from model import *
from tools import *
from tf_init import *
from build import *

# 准备数据集文件
#train_list = cfg.train_csv_list
#train_list = train_list[:1]

def build_input_data(data):

    
    _, large_voxel_index = Draw_voxels(data, cfg.voxel_size, cfg.grid_size, cfg.lidar_coord)
    data, voxel_index = Draw_voxels(_, cfg.mini_voxel_size, cfg.mini_grid_size, cfg.mini_lidar_coord)
    assert data.shape[0] == _.shape[0]
    
    final_data = data[:, :7]
    final_label = data[:, 7:]
    final_voxel_index = build_voxel(large_voxel_index, cfg.RANDOM_NUMBER, cfg.VOXEL_NUMBER, cfg.max_particles_voxel)
    final_neighbor = build_mini_voxel(cfg.MINI_VOXEL_NUMBER, cfg.mini_max_particles_voxel, cfg.mini_voxel_size, data,voxel_index, cfg.bias_voxel)
    
    return [final_data, final_neighbor, final_label, final_voxel_index]



file = './start.bin'
data = np.fromfile(file, dtype=np.float32).reshape(-1,10)
#data = data[:,:7]


time = 0.001



with tf.Graph().as_default():
    with tf.Session(config=GPUInitial(cfg.GPU_MEMORY_FRACTION, cfg.GPU_AVAILABLE)) as sess:
    
        model = Model(
            cfg.learning_rate,
            cfg.GPU_AVAILABLE.split(','),
            cfg.mini_max_particles_voxel,
            cfg.max_particles_voxel,
            cfg.grid_size,
            cfg.VOXEL_NUMBER,
            cfg.RANDOM_NUMBER
        )

        paramInitial(model, sess, cfg.save_model_dir)
        summary_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

        for i in range(5000):
            print("Step:"+str(i*time))
            data[data[:,6] == 0,3:6] = np.squeeze(np.array(model.train(sess,build_input_data(data), train=False, summary=False,onlypred=True)[0]))[data[:,6] == 0,:]
            data[data[:,6] == 0,:3] += time * data[data[:,6] == 0,3:6]
            
            np.save('./predict/file_'+str(time*i)+'.npy', data[data[:,6] == 0,:3])





