import argparse
import collections
import time
import glob
import warnings
import datetime
from torch.utils.data import DataLoader

from utils import *
from datasets import BaseDataset


## record time
train_start_time = datetime.datetime.now()
print('start time:{}'.format(train_start_time))

parser = argparse.ArgumentParser(description="AL_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--max_epoch', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_cpu', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.97)
parser.add_argument("--loss_type", type=str, default='am')
parser.add_argument("--dataset", type=str, default='vox2')
parser.add_argument("--test_step", type=int, default=5)
parser.add_argument("--train_list", type=str, default='data/vox2.txt')
parser.add_argument('--train_path', type=str,default='/home/mengy23/data/voxceleb2/dev/aac')

## Model and Loss settings
parser.add_argument('--C', type=int, default=1024)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()


def train(model, epoch, loader):
    model.train()
    # Update the learning rate based on the current epcoh
    model.scheduler.step(epoch - 1)
    lr = model.optim.param_groups[0]['lr']

    top1, loss_value, num = 0, 0, 1e-7

    progress = tqdm(loader, mininterval=10, ncols=120)
    for batch in progress:
        data, labels = batch
        progress.set_description("train")
        labels = labels.cuda()
        speaker_embedding = model.speaker_encoder.forward(data.cuda(), aug=True)
        speaker_embedding = speaker_embedding[-1] if isinstance(speaker_embedding, tuple) else speaker_embedding
        loss, pred, acc = model.speaker_loss.forward(speaker_embedding, labels)
        nloss = loss.mean()

        model.zero_grad()
        nloss.backward()
        model.optim.step()

        num += 1
        top1 += acc
        loss_value += nloss.detach().cpu().numpy()
        progress.update()
        progress.set_postfix(
            lr='{:.4}'.format(lr),
            loss='{:.4}'.format(loss_value / num),
            acc='{:.4}'.format(100 * top1 / num),
        )
    progress.close()

    return loss_value / num, lr, 100 * top1 / num


def main(args):
    train_dataset = BaseDataset(args.train_list,args.train_path, args.num_frames)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_cpu,
                              drop_last=True)
    args.n_class = train_dataset.get_speaker_number()
    epoch = 1
    model = Model(**vars(args))

    while True:
        print("\nEpoch {}:".format(epoch))
        # Training
        train(model, epoch, train_loader)


if __name__ == '__main__':
    main(args)
