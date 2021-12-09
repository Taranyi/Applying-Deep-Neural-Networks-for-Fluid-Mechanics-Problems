from tools import *
import prepare as cfg

import matplotlib
import matplotlib.pyplot as plt

import shutil
import os

# 携带近邻网格的局部特征，逐帧计算
def build_input(files, i):
    #final_data = []
    #final_label = []
    #final_neighbor = []
    #final_voxel_index = []
    #fps = []
    #timestep = []
    #step = 0
    #for i in index:
    file = files[i]
    #_fps = get_fps(file)
    #print("choose the fps: " + str(_fps))
    #fps.append(_fps)
    #print(str(step)+'/'+str(len(index)))
    data = Load_csv(file)  # input[:, :7], output[:, 7:]
    
    _, large_voxel_index = Draw_voxels(data, cfg.voxel_size, cfg.grid_size, cfg.lidar_coord)
    data, voxel_index = Draw_voxels(_, cfg.mini_voxel_size, cfg.mini_grid_size, cfg.mini_lidar_coord)
    assert data.shape[0] == _.shape[0]

    #timestep.append(_timestep)
    final_data = data[:, :7]
    final_label = data[:, 7:]
    final_voxel_index = build_voxel(large_voxel_index, cfg.RANDOM_NUMBER, cfg.VOXEL_NUMBER, cfg.max_particles_voxel)
    final_neighbor = build_mini_voxel(cfg.MINI_VOXEL_NUMBER, cfg.mini_max_particles_voxel, cfg.mini_voxel_size, data,voxel_index, cfg.bias_voxel)
    
    #ret = [final_data, final_neighbor, final_label, final_voxel_index]
    
    
    #print(ret[0].shape)
    #print(ret[1].shape)
    #print(ret[2].shape)
    
    
    #ret[0] = ret[0][~np.all(ret[2]  == 0, axis=1)]
    #ret[1] = ret[1][~np.all(ret[2]  == 0, axis=1)]
    #ret[2]  = ret[2][~np.all(ret[2]  == 0, axis=1)]
    #ret[0] = ret[0][:,:6]
    
    return [final_data, final_neighbor, final_label, final_voxel_index]


def write_datas(feature, knn_index, y_truth, voxel_index, y_pred, trans_data_dir,id):
    #for i in range(len(fps_list)):
        #fps = fps_list[i]
        pred_out = trans_data_dir + "/all_particles_" + str(id) + ".csv"
        percent = 100 * abs(y_pred[0] - y_truth[0]) / abs(y_truth[0])

        color = ['r','g','b','c','m','y']
        if os.path.exists(trans_data_dir+ '/'+str(id)):
        	shutil.rmtree(trans_data_dir+ '/'+str(id))
        	os.mkdir(trans_data_dir+ '/'+str(id))
        else:
        	os.mkdir(trans_data_dir+ '/'+str(id))
        for i in range(0,percent.shape[1]):
        
        	plt.hist(x=percent[:,i].flatten(), bins=[0,1,2,3,4,5,6,7,8,9,10,15, 20, 30, 40, 50, 100], color=color[i],alpha=0.7, rwidth=0.85)
        	plt.savefig(trans_data_dir+ '/'+str(id) +'/histplot'+str(i)+'.png', bbox_inches='tight')
        	plt.clf()
        with open(pred_out, 'w', newline='') as f:
            writer = csv.writer(f)
            row = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'isFluid', 'ox', 'oy', 'oz', 'pred_x', 'pred_y', 'pred_z', '']
            writer.writerow(row)

            for j in range(len(feature[0])):
                row = []
                for k in range(0, 7):
                    row.append(str(feature[0][j][k]))
                for k in range(0, 3):
                    row.append(str(y_truth[0][j][k]))
                for k in range(0, 3):
                    row.append(str(y_pred[0][j][k]))
                row.append('')
                writer.writerow(row)

def write_data(fps_list, feature, knn_index, y_truth, voxel_index, y_pred, trans_data_dir):
    for i in range(len(fps_list)):
        fps = fps_list[i]
        pred_out = trans_data_dir + r"/all_particles_" + str(fps) + ".csv"
        with open(pred_out, 'w', newline='') as f:
            writer = csv.writer(f)
            row = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'isFluid', 'ox', 'oy', 'oz', 'pred_x', 'pred_y', 'pred_z', '']
            writer.writerow(row)

            for j in range(len(feature[i])):
                row = []
                for k in range(0, 7):
                    row.append(str(feature[i][j][k]))
                for k in range(0, 3):
                    row.append(str(y_truth[i][j][k]))
                for k in range(0, 3):
                    row.append(str(y_pred[i][j][k]))
                row.append('')
                writer.writerow(row)
