import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def __init__(self, n_class, embed_dim):
        super(Softmax, self).__init__()
        self.fc = nn.Linear(embed_dim, n_class)
        self.criertion = nn.CrossEntropyLoss(reduction='none')
        print('Use the cross-entropy loss function')

    def forward(self, x, label=None):
        x = self.fc(x)
        pred = torch.softmax(x, -1)
        loss = self.criertion(x, label.to(torch.long))
        # acc = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        acc = torch.argmax(pred, dim=1).eq(label).sum().item() / float(label.size(0))

        return loss, pred, acc

class AMSoftmax(nn.Module):
    def __init__(self, n_class, m, s, embed_dim):
        super(AMSoftmax, self).__init__()

        self.m = m
        self.s = s
        self.W = torch.nn.Parameter(torch.randn(embed_dim, n_class), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(reduction='none')
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AM-Softmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        # 求L2范数,clamp:将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # 逐元素除法，得到规范化的x
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        # 矩阵乘法，[batch_size,192] x [192,n_class] = [batch_size,n_class]
        costh = torch.mm(x_norm, w_norm)
        # 将标签变为一条竖线
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()

        # 将m在分散放入对应标签的位置，如[[0,m,0,0,0],[0,0,0,m,0],[m,0,0,0,0]]
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.to(costh.device)

        costh_m = costh - delt_costh
        output = self.s * costh_m

        loss = self.ce(output, label.to(torch.long))
        acc = torch.argmax(torch.mm(x, self.W), dim=1).eq(label).sum().item() / float(label.size(0))

        return loss, torch.softmax(torch.mm(x, self.W), -1), acc


class AAMSoftmax(nn.Module):
    def __init__(self, n_class, m, s, embed_dim):
        super(AAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, embed_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        acc = torch.argmax(torch.mm(x, self.weight.T), dim=1).eq(label).sum().item() / float(label.size(0))

        return loss, torch.softmax(torch.mm(x, self.weight.T), -1), acc



if __name__ == '__main__':
    pass
