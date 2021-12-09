from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import prepare as cfg
from model import *
from tools import *
from tf_init import *
from build import *

# 准备数据集文件
train_list = cfg.train_csv_list
#train_list = train_list[:1]



with tf.Graph().as_default():
    with tf.Session(config=GPUInitial(cfg.GPU_MEMORY_FRACTION, cfg.GPU_AVAILABLE)) as sess:
        # 定义模型
        model = Model(
            cfg.learning_rate,
            cfg.GPU_AVAILABLE.split(','),
            cfg.mini_max_particles_voxel,
            cfg.max_particles_voxel,
            cfg.grid_size,
            cfg.VOXEL_NUMBER,
            cfg.RANDOM_NUMBER
        )

        # 初始化模型/恢复模型
        paramInitial(model, sess, cfg.save_model_dir)
        summary_writer = tf.summary.FileWriter(cfg.log_dir, sess.graph)

        for each_epoch in range(cfg.epoch):
            np.random.shuffle(train_list)
            #data = build_input(train_list,0)
            for index in range(len(train_list)):
                print('Epoch:',each_epoch)
                print('File:'+str(index+1)+'/'+str(len(train_list)))
                # [final_data, final_neighbor, final_label, final_voxel_index, fps, timestep]
                data = build_input(train_list,index)

                # train_input[idx, N, 7] & [idx, N, K, 7], train_output[idx, N, 3], voxel_index[idx, N]
                #for i in range(2500):
                ret = model.train(sess, data, train=True, summary=True)
                print("Loss:",ret[1])
                if ret[2] % 10 == 0:
                	print('saving...')
                	summary_writer.add_summary(ret[-1], global_step=ret[2])
                	model.saver.save(sess, os.path.join(cfg.save_model_dir, 'checkpoint'),global_step=model.global_step)

                # print_screen(data[4], ret[1], ret[0], data[2])
                # write_data(data[4], data[0], data[1], data[2], data[3], ret[0], ret[1], cfg.trans_data_dir)
                
                #write_data([0], data[0], data[1], data[2], data[3], ret[0], cfg.trans_data_dir)



            # 准备当前数据
            # 训练
            # 输出数据屏幕/文件
            # 保存模型/log
