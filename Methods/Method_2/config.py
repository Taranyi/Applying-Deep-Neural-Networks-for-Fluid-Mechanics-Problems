import os

epoch = 10
sample_folder = './data3'
file_list = os.listdir(sample_folder)
for item in os.listdir(sample_folder):
    if len(item.split(r'.')) > 1:
        file_list.remove(item)
csv_folders = list(map(lambda x: os.path.join(sample_folder, x), file_list))
csv_range = None

predict_dir = './pred'

save_model_dir = './save_model_dir'
log_dir = './log_dir'
trans_data_dir = './trans'

GPU_AVAILABLE = '0'
GPU_MEMORY_FRACTION = 1

learning_rate = 1e-3
RANDOM_NUMBER = 10

timex = 1.0
