import numpy as np
from functools import reduce

from config import *
from tools import *

voxel_size = timex * np.array([0.3, 0.3, 0.3], dtype=np.float32)  # [-4.2, 4.2]x [-2.2, 6.2]x [-4.2, 4.2] (21,21,21)
grid_size = np.array([11, 11, 11], dtype=np.int32)
lidar_coord = timex * np.array([1.5, 0.25, 1.5], dtype=np.float32)
max_particles_voxel = int(reduce(lambda x, y: x * y, voxel_size)/(0.05*0.05*0.05))

mini_voxel_size = np.array([0.1, 0.1, 0.1], dtype=np.float32)  # [-4.2, 4.2]x [-4.2, 4.2]x [-0.25, 8.15] (21,21,21)
mini_grid_size = np.array(timex * np.array([32, 32, 32], dtype=np.int32), dtype=np.int32)
mini_lidar_coord = timex * np.array([1.5, 0.25, 1.5], dtype=np.float32)
mini_max_particles_voxel = int(reduce(lambda x, y: x * y, mini_voxel_size)/(0.05*0.05*0.05))

VOXEL_NUMBER = reduce(lambda x, y: x * y, grid_size)  # 21*21*21
MINI_VOXEL_NUMBER = reduce(lambda x, y: x * y, mini_grid_size)


#lidar_coord = 
#mini_lidar_coord = 


print('max_particles_voxel:',max_particles_voxel)
print('mini_max_particles_voxel:',mini_max_particles_voxel)
print('VOXEL_NUMBER:',VOXEL_NUMBER)
print('MINI_VOXEL_NUMBER',MINI_VOXEL_NUMBER)


print('Press enter to start')
input()


GPU_USE_COUNT = len(GPU_AVAILABLE.split(','))

os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(trans_data_dir, exist_ok=True)

train_csv_list = Find_csv_from_parent_folder(csv_folders, csv_range)
predict_csv_list = Find_csv(predict_dir)
predict_csv_list.sort(key=get_fps)

bias_x = mini_grid_size[1] * mini_grid_size[2]
bias_y = mini_grid_size[2]
bias_z = 1
bias_voxel = [0, bias_x, -bias_x, bias_y, -bias_y, bias_z, -bias_z,
              bias_x + bias_y, bias_x - bias_y, -bias_x + bias_y, -bias_x - bias_y,
              bias_x + bias_z, bias_x - bias_z, -bias_x + bias_z, -bias_x - bias_z,
              bias_y + bias_z, bias_y - bias_z, -bias_y + bias_z, -bias_y - bias_z,
              bias_x + bias_y + bias_z, bias_x + bias_y - bias_z, bias_x - bias_y + bias_z, bias_x - bias_y - bias_z,
              -bias_x + bias_y + bias_z, -bias_x + bias_y - bias_z, -bias_x - bias_y + bias_z,
              -bias_x - bias_y - bias_z]
