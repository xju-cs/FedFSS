import os

config_names = [
        'vox1_hete_4',
        'vox2_hete_4'
]
batch_size = 256
gpu = 6


for config_name in config_names:
        print('training on {}'.format(config_name))
        # 执行训练脚本
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python train_fedavg.py \
                            --config_name {config_name} \
                            --batch_size {batch_size}"
        os.system(cmd)
        print("one task finish!\n\n")
print("all tasks finish!")

command = f"CUDA_VISIBLE_DEVICES={gpu} python /home/fangzh21/code/LNL-Speaker/train.py --batch_size {batch_size}"
os.system(command)
