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
from config.config import Config

save_root = 'exp/fedanchor_gamma'
## record time
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
train_start_time = datetime.datetime.now()
print('start time:{}'.format(train_start_time))

parser = argparse.ArgumentParser(description="AL_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_cpu', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.97)
parser.add_argument("--loss_type", type=str, default='am')
## Model and Loss settings
parser.add_argument('--C', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)
## New hyperparameters
parser.add_argument('--mu',type=float,default=0.2)
parser.add_argument('--lama',type=float,default=0.9)
parser.add_argument('--temperature',type=float,default=0.01)
parser.add_argument('--train_method',type=str,default='global',help='local ,global, classfier')
##gamma hyperparameters
parser.add_argument("--gamma", type=float, default=0.99)
## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()

## import config
cfg = Config()
config = cfg.generate_config(num_clients=4,fed_mode= 'hete',client_datasets=['vox1','vox1','vox1','vox1'])
args.client_train_path = config['client_train_path']
args.client_train_lists = config['client_train_lists']
args.client_test_path = config['client_test_path']
args.client_test_lists = config['client_test_lists']
args.external_test_path = config['external_test_path']
args.external_test_lists = config['external_test_lists']
args.server_test_path = config['server_test_path']
args.server_test_lists = config['server_test_lists']
args.num_clients = config['num_clients']
args.round = config['round']
args.exp_name = config['exp_name']
args.fed_mode = config['fed_mode']
args.test_step = config['round']

args.exp_name='{}_anchor_{}lama_{}_mu{}_gamma{}_embeddim{}'.format(args.exp_name,args.train_method,args.lama,args.mu,args.gamma,args.embed_dim)
args.save_path = os.path.join(save_root, args.exp_name)
args = init_args(args)

logger = get_logger(args)
logger.log("args: {}\n".format(args))

cmd = 'cp ./train_fedavg.py {}'.format(args.save_path)
os.system(cmd)
logger.log('copy train_fedavg.py to {}'.format(args.save_path))


def contrastive_loss_cosine(speaker_embedding, labels, label_mapping, label_anchor, temperature=0.01):

    anchors = torch.stack([feat for feat in label_anchor.values()]).cuda()
    # 将speaker_embeddings和anchors扩展到与batch_size匹配
    batch_size = speaker_embedding.size(0)
    num_anchors = len(label_anchor)
    anchors_expanded = anchors.T  # [batch_size, num_anchors, embedding_dim]
    # 计算speaker_embeddings和所有anchors之间的余弦相似度
    dot_product = torch.matmul(speaker_embedding, anchors_expanded)
    # 计算 L2 范数
    norm_a = torch.norm(speaker_embedding, p=2, dim=1, keepdim=True)  # 形状为 [128, 1]
    norm_b = torch.norm(anchors_expanded, p=2, dim=0, keepdim=True)  # 形状为 [1, 1211]
    eps = 1e-8
    norm_product = torch.clamp(norm_a @ norm_b, min=eps)  # 结果形状为 [128, 1211]
    # 计算余弦相似度
    cos_similarities = dot_product / norm_product
    # logits = cos_similarities / temperature
    # logits = torch.sigmoid(logits)

    # 创建目标标签：初始化为全0张量，对于每个样本，只有对应于其真实标签的anchor位置设为1
    target_labels = torch.zeros((batch_size, num_anchors), device=speaker_embedding.device)

    for i, label in enumerate(labels):
        anchor_id = label_mapping[label.item()]
        anchor_index = list(label_anchor.keys()).index(anchor_id)
        target_labels[i, anchor_index] = 1.0
    # 计算交叉熵损失
    loss = F.cross_entropy(cos_similarities, target_labels)
    return loss.mean()


def local_anchor_train(model, client_index, mu, temperature, epoch, loader, label_mapping, label_anchor):
    model.train()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']

    top1, loss_value, num = 0, 0, 1e-7
    temp_collector=defaultdict(list)
    temp_labels = []
    temp_embeddings = []

    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        num+=1
        data, labels = batch
        progress.set_description("Train {}".format(client_index))
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding

        loss, pred, acc = model.speaker_loss.forward(speaker_embedding, labels)
        nloss = loss.mean()
        # 计算辅助损失
        aux_loss = contrastive_loss_cosine(speaker_embedding, labels, label_mapping, label_anchor,temperature=temperature)
        total_loss = (1-mu)*nloss + mu*aux_loss

        temp_labels.extend(labels)
        temp_embeddings.extend(speaker_embedding.detach())

        model.zero_grad()
        total_loss.backward()
        model.optim.step()

        top1 += acc
        loss_value += total_loss.detach().cpu().numpy()
        progress.update()
        progress.set_postfix(
            lr='{:.4}'.format(lr),
            loss='{:.4}'.format(loss_value / num),
            acc='{:.4}'.format(100 * top1 / num),
            anchor='{:.4}'.format(aux_loss)
        )
    progress.close()
    for i, label in enumerate(temp_labels):
        id=label_mapping[label.item()]
        temp_collector[id].append(temp_embeddings[i])

    avg_embeddings = {id: torch.mean(torch.stack(embeddings), dim=0).cuda() for id, embeddings in
                      temp_collector.items()}
    return loss_value / num, lr, 100 * top1 / num, avg_embeddings


def global_anchor_train(model, global_model, client_index, mu, temperature , epoch, loader, label_mapping, label_anchor):
    model.train()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']

    top1, loss_value, num = 0, 0, 1e-7
    temp_collector=defaultdict(list)
    temp_labels = []
    temp_embeddings = []

    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        num+=1
        data, labels = batch
        progress.set_description("Train {}".format(client_index))
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        anchor_embed= global_model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        anchor_embed = anchor_embed[-1] if isinstance(anchor_embed, tuple) else anchor_embed

        loss, pred, acc = model.speaker_loss.forward(speaker_embedding, labels)
        nloss = loss.mean()
        # 计算辅助损失
        aux_loss = contrastive_loss_cosine(speaker_embedding, labels, label_mapping, label_anchor, temperature=temperature)
        total_loss = (1-mu)*nloss + mu*aux_loss

        temp_labels.extend(labels)
        temp_embeddings.extend(anchor_embed.detach())

        model.zero_grad()
        total_loss.backward()
        model.optim.step()

        top1 += acc
        loss_value += total_loss.detach().cpu().numpy()
        progress.update()
        progress.set_postfix(
            lr='{:.4}'.format(lr),
            loss='{:.4}'.format(loss_value / num),
            acc='{:.4}'.format(100 * top1 / num),
            anchor='{:.4}'.format(aux_loss)
        )
    progress.close()
    for i, label in enumerate(temp_labels):
        id=label_mapping[label.item()]
        temp_collector[id].append(temp_embeddings[i])

    avg_embeddings = {id: torch.mean(torch.stack(embeddings), dim=0).cuda() for id, embeddings in
                      temp_collector.items()}
    return loss_value / num, lr, 100 * top1 / num, avg_embeddings

def classfier_anchor_train(model, client_index, mu, temperature, epoch, loader, label_mapping, label_anchor):
    model.train()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']

    top1, loss_value, num = 0, 0, 1e-7
    temp_collector=defaultdict(list)

    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        num+=1
        data, labels = batch
        progress.set_description("Train {}".format(client_index))
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding

        loss, pred, acc = model.speaker_loss.forward(speaker_embedding, labels)
        nloss = loss.mean()
        # 计算辅助损失
        aux_loss = contrastive_loss_cosine(speaker_embedding, labels, label_mapping, label_anchor, temperature=temperature)
        total_loss = (1-mu)*nloss + mu*aux_loss

        model.zero_grad()
        total_loss.backward()
        model.optim.step()

        top1 += acc
        loss_value += total_loss.detach().cpu().numpy()
        progress.update()
        progress.set_postfix(
            lr='{:.4}'.format(lr),
            loss='{:.4}'.format(loss_value / num),
            acc='{:.4}'.format(100 * top1 / num),
            anchor='{:.4}'.format(aux_loss)
        )
    progress.close()
    anchor_matrix=model.speaker_loss.W.data
    for i in range(anchor_matrix.size(1)):
        weight_vector = anchor_matrix[:, i]
        id=label_mapping[i]
        temp_collector[id].append(weight_vector)

    avg_embeddings = {id: torch.mean(torch.stack(embeddings), dim=0).cuda() for id, embeddings in
                      temp_collector.items()}
    return loss_value / num, lr, 100 * top1 / num, avg_embeddings


def main(args):
    # Build clients
    client_train_datasets = []
    client_train_loaders = []
    client_models = []
    client_evaluators = []
    client_label_mapping={}
    client_avgemb_collector=defaultdict(list)
    for i in range(args.num_clients):
        logger.log("Building Client {}...".format(i+1))
        client_label_mapping[i]=getlabel_mapping(args.client_train_lists[i])
        client_train_datasets.append(BaseDataset(args.client_train_lists[i],args.client_train_path[i], args.num_frames))
        client_train_loaders.append(DataLoader(client_train_datasets[i], batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.n_cpu,
                                               drop_last=True))
        args.n_class = client_train_datasets[i].get_speaker_number()
        client_models.append(Model(**vars(args)))
        client_evaluators.append(Evaluator(args.client_test_lists[i], args.client_test_path[i]))
    args.weights = compute_client_avg_weight(client_train_datasets)
    logger.log("args.weights: {}".format(args.weights))

    # Build server
    server_model = Model(**vars(args))
    for para in server_model.parameters():
        para.requires_grad=False
    server_evaluators = []
    for server_test_list, server_test_path in zip(args.server_test_lists, args.server_test_path):
        server_evaluators.append(Evaluator(server_test_list, server_test_path))

    label_anchor={}
    label_anchor=init_label_anchor(client_label_mapping, label_anchor, args.embed_dim)
    EERs = [100]
    epoch = 1

    if not os.path.exists(args.score_save_path):
        with open(args.score_save_path, "w") as f:
            f.write("Time,Epoch,Client,Trials,LR,Loss,Acc,EER,minDCF(0.01),minDCF(0.001),bestEER\n")
            f.close()
    score_file = open(args.score_save_path, "a+")

    while epoch <= args.max_epoch:
        logger.log("\nEpoch {}:".format(epoch))
        epoch_start_time = datetime.datetime.now()
        logger.log(epoch_start_time)

        # Training for each client
        for i in range(args.num_clients):
            # Update client with server model
            if epoch % args.round == 1:
                client_models[i].copy_weights(server_model, mode=args.fed_mode)
                logger.log("Updated Client {} with Server".format(i + 1))

            # Training
            if args.train_method=='local':
                loss, lr, acc, avg_embedding = local_anchor_train(client_models[i], i+1, args.mu, args.temperature, epoch, client_train_loaders[i], client_label_mapping[i], label_anchor)
            elif args.train_method=='global':
                loss, lr, acc, avg_embedding = global_anchor_train(client_models[i], server_model, i+1, args.mu, args.temperature, epoch, client_train_loaders[i], client_label_mapping[i], label_anchor)
            elif args.train_method=='classfier':
                loss, lr, acc, avg_embedding = classfier_anchor_train(client_models[i], i+1, args.mu, args.temperature, epoch, client_train_loaders[i], client_label_mapping[i], label_anchor)
            client_avgemb_collector[i].append(avg_embedding)
            # Save Model
            if epoch == 60 or epoch == args.max_epoch:
                client_models[i].save_parameters(args.model_save_path + "/model_%04d_%d.model" % (epoch, i + 1))

            # Evaluation for client
            EER, minDCF01, minDCF001 = None, None, None
            if epoch % args.test_step == 0:
                print("Client {} Evaluation on {}".format(i+1, args.client_test_lists[i]))
                EER, minDCF01, minDCF001 = client_evaluators[i].eval(client_models[i])
                EERs.append(EER)
                logger.log(
                    "{} ACC:{:.4f}, EER:{:.4f}, minDCF:{:.4f}, bestEER:{:.4f}".format(datetime.datetime.now(), acc, EER,
                                                                                      minDCF01, min(EERs)),
                    time.strftime("%Y-%m-%d %H:%M:%S"))
            score_file.write(
                "{},{},{},{},{},{},{},{},{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, i + 1,
                                                            args.client_test_lists[i], lr, loss, acc, EER,
                                                            minDCF01, minDCF001, min(EERs)))
            score_file.flush()

        # Update server with client model
        if epoch % args.round == 0:
            label_anchor = update_label_anchor(client_avgemb_collector, label_anchor, lama=args.lama)

            # Reset client_avgemb_collector for the next round
            client_avgemb_collector = defaultdict(list)
            server_model.update_parameters_gamma(client_models, args.weights, args.gamma, mode=args.fed_mode)
            logger.log("Updated Server with Clients")

            # Evaluation for server
            for j in range(len(server_evaluators)):
                logger.log("Server Evaluation on {}".format(args.server_test_lists[j]))
                EER, minDCF01, minDCF001 = server_evaluators[j].eval(server_model)
                EERs.append(EER)
                logger.log(
                    "{} EER:{:.4f}, minDCF:{:.4f}, bestEER:{:.4f}".format(datetime.datetime.now(), EER, minDCF01,
                                                                          min(EERs)),
                    time.strftime("%Y-%m-%d %H:%M:%S"))
                score_file.write(
                    "{},{},{},{},{},{},{},{},{},{},{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, "server",
                                                                args.server_test_lists[j], {}, {}, {}, EER,
                                                                minDCF01, minDCF001, min(EERs)))
                score_file.flush()

        epoch += 1
        ## record time
        epoch_end_time = datetime.datetime.now()
        logger.log('this epoch time:{}\n'.format(epoch_end_time - epoch_start_time))
    train_end_time = datetime.datetime.now()
    logger.log('total train time:{}\n'.format(train_end_time - train_start_time))


if __name__ == '__main__':
    main(args)
