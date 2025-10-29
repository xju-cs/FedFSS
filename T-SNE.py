
import numpy as np
from sklearn.manifold import TSNE
import datetime
import matplotlib.pyplot as plt
import argparse
import collections
import copy
import os
import time
import glob
import warnings
import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict

from utils import *
from datasets import BaseDataset

save_root = 'exp/test_t-SNE'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
train_start_time = datetime.datetime.now()
print('start time:{}'.format(train_start_time))

parser = argparse.ArgumentParser(description="AL_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_cpu', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.97)
parser.add_argument("--loss_type", type=str, default='am')
parser.add_argument("--config_name", type=str, default='')
## Model and Loss settings
parser.add_argument('--C', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)
## anchor hyperparameters
parser.add_argument('--mu',type=float,default=0.2)
parser.add_argument('--lama',type=float,default=0.9)
parser.add_argument('--temperature',type=float,default=0.01)
parser.add_argument('--train_method',type=str,default='global',help='local ,global, classfier')
## reweight hyperparameters
parser.add_argument('--reweight_method', type=str, default='exp',help="exp,com,softmax")
parser.add_argument("--gamma", type=float, default=1.0)
## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

## import config
config = import_config('config.{}'.format(args.config_name))
args.client_train_lists = config['client_train_lists']
args.client_test_lists = config['client_test_lists']
args.external_test_lists = config['external_test_lists']
args.server_test_lists = config['server_test_lists']
args.num_clients = config['num_clients']
args.round = config['round']
args.exp_name = config['exp_name']
args.fed_mode = config['fed_mode']
args.test_step = config['round']
args.client_model_path = config['client_model_path']
args.save_path = os.path.join(save_root, args.exp_name)
args = init_args(args)

# 导入配置
config = import_config('config.{}'.format(args.config_name))
args.client_train_lists = config['client_train_lists']
args.client_test_lists = config['client_test_lists']
args.external_test_lists = config['external_test_lists']
args.server_test_lists = config['server_test_lists']
args.num_clients = config['num_clients']
args.round = config['round']
args.exp_name = config['exp_name']
args.save_path = os.path.join(save_root, args.exp_name)
args = init_args(args)

logger = get_logger(args)
logger.log("args: {}\n".format(args))

cmd = 'cp ./train_fedavg.py {}'.format(args.save_path)
os.system(cmd)
logger.log('copy train_fedavg.py to {}'.format(args.save_path))


def extract_embedding(model, client_index, loader):
    embeddings = []
    labels_list = []
    model.eval()
    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        data, labels = batch
        progress.set_description("Train {}".format(client_index))
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        embeddings.extend(speaker_embedding.detach().cpu().numpy())
        labels_list.extend(labels.cpu().numpy())  # 获取类别标签
    return embeddings, labels_list


def main(args):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # 定义每种颜色对应的颜色和标记
    colors = ['red', 'green', 'blue', 'purple']
    markers = ['o', 's', '^', 'p']
    combined_embeddings = []
    all_labels = []
    unique_labels = None
    j=-1
    for key, model_path in args.client_model_path.items():
    # Build clients
        j+=1
        row = j // 2  # 整除得到行号
        col = j % 2  # 取余得到列号
        client_train_datasets = []
        client_train_loaders = []
        client_models = []

        all_embeddings = []
        all_labels = []
        client_label_mapping = {}
        for i in range(args.num_clients):
            logger.log("Building Client {}...".format(i + 1))
            client_label_mapping[i] = getlabel_mapping(args.client_train_lists[i])
            client_train_datasets.append(BaseDataset(args.client_train_lists[i], args.num_frames))
            client_train_loaders.append(DataLoader(client_train_datasets[i], batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.n_cpu, drop_last=False))
            args.n_class = client_train_datasets[i].get_speaker_number()
            client_models.append(Model(**vars(args)))

        for i in range(args.num_clients):
            client_models[i].load_parameters(model_path[i])
            embedding, labels = extract_embedding(client_models[i], i + 1, client_train_loaders[i])
            all_embeddings.extend(embedding)
            all_labels.extend(f'{i}_{client_label_mapping[i][label]}' for label in labels)

        combined_embeddings = np.vstack(all_embeddings)
        tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(combined_embeddings) - 1))  # 确保 perplexity 小于样本数
        embeddings_2d = tsne.fit_transform(combined_embeddings)

        unique_labels = np.unique(all_labels) # 使用tab10调色板
        for idx, label in enumerate(unique_labels):
            mask = np.array(all_labels) == label  # 创建布尔掩码
            axs[j].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], marker=markers[idx % len(markers)], c=[colors[int(label.split('_')[0])]],
                    label=f'{label.split("_")[-1]}', alpha=0.6)
        axs[j].set_title('{}'.format(key))
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.0), ncol=(len(unique_labels)//2), fancybox=True,
               shadow=True)
    # 调整布局以避免重叠
    plt.tight_layout()

    # 调整顶部间距以适应图例
    plt.subplots_adjust(top=0.8)
    plt.savefig(os.path.join(args.save_path, 't-SNE.png'))

    # 显示图形
    plt.show()

if __name__ == '__main__':
    main(args)