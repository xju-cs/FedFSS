import os

vox1_dev_path = '/home/mengy23/data/voxceleb1/dev/wav'
vox2_dev_path = '/home/mengy23/data/voxceleb2/dev/wav'
cn1_dev_path = '/home/mengy23/data/cnceleb/cn_1/data/'
cn2_dev_path = '/home/mengy23/data/cnceleb/dev/wav'

vox1_test_path = '/home/mengy23/data/voxceleb1/test/wav'
vox2_test_path = '/home/mengy23/data/voxceleb2/test/wav'
cn1_test_path = '/home/mengy23/data/cnceleb/cn_1/eval'
cn2_test_path = '/home/mengy23/data/cnceleb/dev/wav'

trials_list = ['vox_o.txt', 'cn_test.txt']
train_lists = [
                'cn1.txt'
               ]
train_paths = [ cn1_dev_path ]
test_paths = [cn1_test_path]
test_lists = [1]
batch_size = 128
gpu = 4


for train_list, test_index, train_path, test_path in zip(train_lists, test_lists,train_paths,test_paths):
        dataset = train_list.split('/')[-1].split('.')[0]
        train_list = f'redata/{train_list}'
        test_list = f'redata/{trials_list[test_index]}'
        print('training on {}'.format(train_list))
        # 执行训练脚本
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python train_standard.py \
                            --dataset {dataset} \
                            --batch_size {batch_size} \
                            --train_path {train_path}\
                            --train_list {train_list} \
                            --embed_dim 512 \
                            --test_path {test_path}\
                            --test_list {test_list}"
        os.system(cmd)
        print("one task finish!\n\n")
print("all tasks finish!")

cmd = f"CUDA_VISIBLE_DEVICES={gpu} python train.py --batch_size 256"
os.system(cmd)
