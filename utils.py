import importlib
from loss import *
from datasets import *
from model.ecapa import ECAPA_TDNN
from model.resnet import ResNet34
from tools import Logger
from collections import defaultdict
import copy
import numpy as np

class Model(nn.Module):
    def __init__(self, C, n_class, m, s, loss_type, lr, lr_decay, embed_dim, **kwargs):
        super().__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C, embed_dim=embed_dim).cuda()
        self.speaker_loss = get_loss_function(loss_type, n_class, m, s, embed_dim)
        self.optim = get_optimizer(self, lr)
        self.scheduler = get_scheduler(self.optim, lr_decay)

    def forward(self, data, labels):
        speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        loss, pred, acc = self.speaker_loss.forward(speaker_embedding, labels)
        return loss, pred, acc

    def extract_embedding(self, data):
        self.speaker_encoder.forward(data.cuda(), aug=False)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    def update_parameters(self, models, weights, mode='hete'):
        speaker_encoder_state = self.speaker_encoder.state_dict()

        for name, param in speaker_encoder_state.items():
            temp_state = models[0].speaker_encoder.state_dict()
            speaker_encoder_state[name] = weights[0] * temp_state[name]
            for i in range(1, len(models)):
                temp_state = models[i].speaker_encoder.state_dict()
                speaker_encoder_state[name] += weights[i] * temp_state[name]

        self.speaker_encoder.load_state_dict(speaker_encoder_state)

        if mode == 'homo':
            speaker_loss_state = self.speaker_loss.state_dict()
            for name, param in speaker_loss_state.items():
                temp_state = models[0].speaker_loss.state_dict()
                speaker_loss_state[name] = weights[0] * temp_state[name]
                for i in range(1, len(models)):
                    temp_state = models[i].speaker_loss.state_dict()
                    speaker_loss_state[name] += weights[i] * temp_state[name]
            self.speaker_loss.load_state_dict(speaker_loss_state)

    def update_parameters_gamma(self, models, weights, gamma, mode='hete'):
        speaker_encoder_state = self.speaker_encoder.state_dict()

        for name, param in speaker_encoder_state.items():
            temp_state = models[0].speaker_encoder.state_dict()
            speaker_encoder_state[name] = gamma * weights[0] * temp_state[name]
            for i in range(1, len(models)):
                temp_state = models[i].speaker_encoder.state_dict()
                speaker_encoder_state[name] += gamma * weights[i] * temp_state[name]

        self.speaker_encoder.load_state_dict(speaker_encoder_state)

        if mode == 'homo':
            speaker_loss_state = self.speaker_loss.state_dict()
            for name, param in speaker_loss_state.items():
                temp_state = models[0].speaker_loss.state_dict()
                speaker_loss_state[name] = gamma * weights[0] * temp_state[name]
                for i in range(1, len(models)):
                    temp_state = models[i].speaker_loss.state_dict()
                    speaker_loss_state[name] += gamma * weights[i] * temp_state[name]
            self.speaker_loss.load_state_dict(speaker_loss_state)

    def copy_weights(self, server_model, mode='hete'):
        speaker_encoder_state = self.speaker_encoder.state_dict()

        for name, param in speaker_encoder_state.items():
            global_state = server_model.speaker_encoder.state_dict()
            speaker_encoder_state[name] = global_state[name]

        self.speaker_encoder.load_state_dict(speaker_encoder_state)

        if mode == 'homo':
            speaker_loss_state = self.speaker_loss.state_dict()
            for name, param in speaker_loss_state.items():
                global_state = server_model.speaker_loss.state_dict()
                speaker_loss_state[name] = global_state[name]
            self.speaker_loss.load_state_dict(speaker_loss_state)

    def upload_parameters(self, mode='hete'):
        speaker_encoder_para = [param.clone().detach() for param in self.speaker_encoder.parameters()]

        if mode == 'homo':
            if isinstance(self.speaker_loss, torch.nn.Module):
                speaker_loss_para = [param.clone().detach() for param in self.speaker_loss.parameters()]
                return {'speaker_encoder': speaker_encoder_para, 'speaker_loss': speaker_loss_para}
            else:
                raise ValueError("For 'homo' mode, speaker_loss must be an instance of torch.nn.Module.")
        else:  # Assume 'hete' as default mode
            return {'speaker_encoder': speaker_encoder_para}

    def compute_fedprox_regularizer(self, global_w):
        differences=0
        for key, value in global_w.items():
            if key == 'speaker_encoder':
                for param_index, param in enumerate(self.speaker_encoder.parameters()):
                    differences += torch.norm((param - value[param_index])) ** 2
            if key == 'speaker_loss':
                for param_index, param in enumerate(self.speaker_loss.parameters()):
                    differences += torch.norm((param - value[param_index])) ** 2
        return differences


class ConModel(nn.Module):
    def __init__(self, C, n_class, m, s, loss_type, lr, lr_decay, embed_dim, **kwargs):
        super().__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C, embed_dim=embed_dim).cuda()
        self.speaker_loss = get_loss_function(loss_type, n_class, m, s, embed_dim)
        self.optim = get_optimizer(self, lr)
        self.scheduler = get_scheduler(self.optim, lr_decay)
        self.Header=nn.Linear(embed_dim,256).cuda()

    def forward(self, data, labels):
        speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        loss, pred, acc = self.speaker_loss.forward(speaker_embedding, labels)
        return loss, pred, acc

    def extract_embedding(self, data, aug):
        x=self.speaker_encoder.forward(data.cuda(), aug)
        con_x=self.Header(x)
        return x,con_x


    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    def update_parameters(self, models, weights, mode='hete'):
        speaker_encoder_state = self.speaker_encoder.state_dict()

        for name, param in speaker_encoder_state.items():
            temp_state = models[0].speaker_encoder.state_dict()
            speaker_encoder_state[name] = weights[0] * temp_state[name]
            for i in range(1, len(models)):
                temp_state = models[i].speaker_encoder.state_dict()
                speaker_encoder_state[name] += weights[i] * temp_state[name]
        self.speaker_encoder.load_state_dict(speaker_encoder_state)

        header_state=self.Header.state_dict()

        for name, param in header_state.items():
            temp_state = models[0].Header.state_dict()
            header_state[name] = weights[0] * temp_state[name]
            for i in range(1, len(models)):
                temp_state = models[i].Header.state_dict()
                header_state[name] += weights[i] * temp_state[name]
        self.Header.load_state_dict(header_state)

        if mode == 'homo':
            speaker_loss_state = self.speaker_loss.state_dict()
            for name, param in speaker_loss_state.items():
                temp_state = models[0].speaker_loss.state_dict()
                speaker_loss_state[name] = weights[0] * temp_state[name]
                for i in range(1, len(models)):
                    temp_state = models[i].speaker_loss.state_dict()
                    speaker_loss_state[name] += weights[i] * temp_state[name]
            self.speaker_loss.load_state_dict(speaker_loss_state)

    def update_parameters_gamma(self, models, weights, gamma, mode='hete'):
        speaker_encoder_state = self.speaker_encoder.state_dict()

        for name, param in speaker_encoder_state.items():
            temp_state = models[0].speaker_encoder.state_dict()
            speaker_encoder_state[name] = gamma * weights[0] * temp_state[name]
            for i in range(1, len(models)):
                temp_state = models[i].speaker_encoder.state_dict()
                speaker_encoder_state[name] += gamma * weights[i] * temp_state[name]

        self.speaker_encoder.load_state_dict(speaker_encoder_state)

        if mode == 'homo':
            speaker_loss_state = self.speaker_loss.state_dict()
            for name, param in speaker_loss_state.items():
                temp_state = models[0].speaker_loss.state_dict()
                speaker_loss_state[name] = gamma * weights[0] * temp_state[name]
                for i in range(1, len(models)):
                    temp_state = models[i].speaker_loss.state_dict()
                    speaker_loss_state[name] += gamma * weights[i] * temp_state[name]
            self.speaker_loss.load_state_dict(speaker_loss_state)

    def copy_weights(self, server_model, mode='hete'):
        speaker_encoder_state = self.speaker_encoder.state_dict()

        for name, param in speaker_encoder_state.items():
            global_state = server_model.speaker_encoder.state_dict()
            speaker_encoder_state[name] = global_state[name]

        self.speaker_encoder.load_state_dict(speaker_encoder_state)

        header_state = self.Header.state_dict()

        for name, param in header_state.items():
            global_state = server_model.Header.state_dict()
            header_state[name] = global_state[name]
        self.Header.load_state_dict(header_state)

        if mode == 'homo':
            speaker_loss_state = self.speaker_loss.state_dict()
            for name, param in speaker_loss_state.items():
                global_state = server_model.speaker_loss.state_dict()
                speaker_loss_state[name] = global_state[name]
            self.speaker_loss.load_state_dict(speaker_loss_state)

    def upload_parameters(self, mode='hete'):
        speaker_encoder_para = [param.clone().detach() for param in self.speaker_encoder.parameters()]

        if mode == 'homo':
            if isinstance(self.speaker_loss, torch.nn.Module):
                speaker_loss_para = [param.clone().detach() for param in self.speaker_loss.parameters()]
                return {'speaker_encoder': speaker_encoder_para, 'speaker_loss': speaker_loss_para}
            else:
                raise ValueError("For 'homo' mode, speaker_loss must be an instance of torch.nn.Module.")
        else:  # Assume 'hete' as default mode
            return {'speaker_encoder': speaker_encoder_para}


def compute_client_avg_weight(client_train_datasets):
    weights = []
    for i in range(len(client_train_datasets)):
        weights.append(client_train_datasets[i].get_speaker_number())
    weights = [x / sum(weights) for x in weights]
    return weights

def compute_client_eer_weight(eers):
    weights = []
    for i in range(len(eers)):
        weights.append(1 - eers[i])
    weights = [i / sum(weights) for i in weights]
    return weights

def getlabel_mapping(train_list):
    if train_list is None:
        print('create a empty dataset')
    else:
        lines=open(train_list).read().splitlines()
        ids = list(set([x.split()[0] for x in lines]))
        ids.sort()
        label_dict={key: id for key, id in enumerate(ids)}
        return label_dict


def init_label_anchor(client_label_mapping, label_anchor, feature_dim=256):
    """
    初始化 label_anchor 字典，对于每个客户端的标签映射，
    如果该 id 不在 label_anchor 中，则随机初始化一个 feature_dim 维的张量。

    参数:
        client_label_mapping (dict): 客户端标签到 ID 的映射字典。
        label_anchor (dict): 锚点字典，键为 id，值为对应的特征向量。
        feature_dim (int): 特征维度，默认 128。

    返回:
        dict: 更新后的 label_anchor 字典。
    """
    all_ids = set()

    # 收集所有客户端的 id
    for client_id, mapping in client_label_mapping.items():
        all_ids.update(mapping.values())

    # 对于不在 label_anchor 中的 id，随机初始化 feature_dim 维的张量
    for id in all_ids:
        if id not in label_anchor:
            label_anchor[id] = torch.randn(feature_dim)

    return label_anchor

def init_nums_label_anchor(anchor_nums, label_anchor, feature_dim=256):
    """
    初始化 label_anchor 字典，对于每个客户端的标签映射，
    如果该 id 不在 label_anchor 中，则随机初始化一个 feature_dim 维的张量。

    参数:
        label_anchor (dict): 锚点字典，键为 id，值为对应的特征向量。
        feature_dim (int): 特征维度，默认 128。

    返回:
        dict: 更新后的 label_anchor 字典。
    """

    # 对于不在 label_anchor 中的 id，随机初始化 feature_dim 维的张量
    for id in range(anchor_nums):
            label_anchor[id] = torch.randn(feature_dim)

    return label_anchor


def assign_anchors_by_client_ratio(weights, anchors):
    """
    按照客户端数据比例分配锚点索引

    参数:
        client_data_counts (list): 各客户端的数据量列表，如[100, 200, 300]
        anchors (list or dict): 全部锚点(可以是列表或字典)

    返回:
        dict: {客户端ID: [分配的锚点索引列表]}
    """
    # 确定总锚点数
    total_anchors = len(anchors) if isinstance(anchors, list) else len(anchors.keys())

    # 计算各客户端应分配的锚点数(至少分配1个)
    anchor_nums = [max(1, int(weight * total_anchors)) for weight in weights]
    anchor_nums[-1] = total_anchors - sum(anchor_nums[:-1])  # 确保总数正确

    # 随机打乱所有锚点索引
    all_indices = np.random.permutation(total_anchors)

    # 按比例分配索引
    client_anchor_indices = {}
    start = 0
    for client_id, num in enumerate(anchor_nums):
        end = start + num
        client_anchor_indices[client_id] = all_indices[start:end].tolist()
        start = end

    return client_anchor_indices
def update_label_anchor(client_avgemb_collector, label_anchor, lama):

    # Collect embeddings for each id across all clients and rounds
    id_embeddings = defaultdict(list)
    for client_id, round_embeddings in client_avgemb_collector.items():
        for round_emb in round_embeddings:  # 遍历每一轮的 avg_embedding
            for id, emb in round_emb.items():
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb).cuda()  # 确保嵌入是 PyTorch 张量并移到 GPU 上
                id_embeddings[id].append(emb)

    # Compute the average embedding for each id and apply weighted update
    for id, embeddings in id_embeddings.items():
        avg_emb = torch.mean(torch.stack(embeddings), dim=0).cuda()
        if id in label_anchor:
            label_anchor[id]=label_anchor[id].cuda()
            label_anchor[id] = (1 - lama) * label_anchor[id] + lama * avg_emb
        else:
            label_anchor[id] = avg_emb  # 如果是新的ID，则直接赋值

    return label_anchor


def update_anchors(client_avgemb_collector, anchors):

    # Collect embeddings for each id across all clients and rounds
    id_embeddings = defaultdict(list)
    for client_id, round_embeddings in client_avgemb_collector.items():
        for round_emb in round_embeddings:  # 遍历每一轮的 avg_embedding
            for id, emb in round_emb.items():  # 确保嵌入是 PyTorch 张量并移到 GPU 上
                id_embeddings[id].append(emb.cuda())

    # Compute the average embedding for each id and apply weighted update
    for id, embeddings in id_embeddings.items():
        avg_emb = torch.mean(torch.stack(embeddings), dim=0).cuda()
        anchors[id] = avg_emb
    return anchors# 如果是新的ID，则直接赋值
def compute_client_rw_weights(eers,method,scale=1.0,temprature=0.1):
    eers=torch.tensor(eers)
    eers=eers/100
    softmax = nn.Softmax(dim=0)
    weights=[]
    if method=='com':
        for i in range(len(eers)):
            weights.append(1-eers[i].item())
        weights=[i/sum(weights) for i in weights]
        return weights
    if method=='exp':
        log_eers = -torch.log(eers)  # 取对数并取负数
        exp_log_eers = torch.exp(log_eers * scale)  # 使用指数函数，调整系数 scale 可以改变权重分布
        weights = exp_log_eers / exp_log_eers.sum()  # 归一化 # 将张量转换为权重列表in weights]
        return weights.tolist()
    if method=='softmax':
        eers= eers / temprature
        for eer in eers:
            weights.append(1-eer)
        weights=torch.tensor(weights)
        weights=softmax(weights)
        return weights.tolist()

def import_config(config_file):
    config_module = importlib.import_module(config_file)
    config = config_module.config
    return config


def get_logger(args):
    return Logger(args.log_file)


def get_loss_function(loss_type, n_class, m, s, embed_dim):
    if loss_type == 'ce':
        return Softmax(n_class=n_class, embed_dim=embed_dim).cuda()
    elif loss_type == 'am':
        return AMSoftmax(n_class=n_class, m=m, s=s, embed_dim=embed_dim).cuda()
    elif loss_type == 'aam':
        return AAMSoftmax(n_class=n_class, m=m, s=s, embed_dim=embed_dim).cuda()
    else:
        print("Loss type not supported!")
        exit(0)


def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)


def get_scheduler(optimizer, lr_decay):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)


if __name__ == '__main__':
    model = Model(C=1024, n_class=5994, m=0.2, s=30, loss_type='ce', lr=0.001, lr_decay=0.97, embed_dim=192)
    print(model)

